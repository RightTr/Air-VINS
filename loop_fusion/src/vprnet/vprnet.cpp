#include "vprnet.h"

#include <NvOnnxParser.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
constexpr float kImageMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kImageStd[3] = {0.229f, 0.224f, 0.225f};
}  // namespace

VPRNet::TrtLogger::TrtLogger(bool verbose) : verbose_(verbose) {}

void VPRNet::TrtLogger::log(Severity severity, const char* msg) noexcept
{
  if (!msg) {
    return;
  }
  if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
    std::cerr << "[VPRNet][TRT] " << msg << std::endl;
  } else if (verbose_ && severity <= Severity::kWARNING) {
    std::cout << "[VPRNet][TRT] " << msg << std::endl;
  }
}

VPRNet::VPRNet()
    : VPRNet(Options{})
{
}

VPRNet::VPRNet(const Options& options)
    : options_(options),
      logger_(options.verbose),
      runtime_(nullptr),
      input_binding_index_(-1),
      output_binding_index_(-1),
      input_layout_nhwc_(false),
      input_layout_nchw_(false),
      device_input_(nullptr),
      device_output_(nullptr),
      input_elements_(0),
      output_elements_(0),
      descriptor_dim_(0),
      stream_(nullptr),
      stream_created_(false)
{
  input_dims_.nbDims = 0;
  output_dims_.nbDims = 0;
}

VPRNet::VPRNet(const std::string& engine_path, int input_width, int input_height)
    : VPRNet(Options{})
{
  options_.engine_path = engine_path;
  options_.input_width = input_width;
  options_.input_height = input_height;
}

VPRNet::~VPRNet()
{
  release_buffers();
  context_.reset();
  engine_.reset();
  runtime_.reset();
  if (stream_created_) {
    cudaStreamDestroy(stream_);
    stream_created_ = false;
  }
}

bool VPRNet::build()
{
  release_buffers();
  context_.reset();
  engine_.reset();
  runtime_.reset();
  if (stream_created_) {
    cudaStreamDestroy(stream_);
    stream_created_ = false;
  }

  if (std::filesystem::exists(options_.engine_path)) {
    std::ifstream file(options_.engine_path, std::ios::binary | std::ios::ate);
    if (file.is_open()) {
      const std::streamsize size = file.tellg();
      if (size > 0) {
        std::vector<char> engine_data(static_cast<size_t>(size));
        file.seekg(0, std::ios::beg);
        if (file.read(engine_data.data(), size)) {
          runtime_.reset(nvinfer1::createInferRuntime(logger_));
          if (!runtime_) {
            std::cerr << "[VPRNet] failed to create TensorRT runtime" << std::endl;
            return false;
          }
          nvinfer1::ICudaEngine* raw_engine = runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size());
          if (raw_engine) {
            engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(raw_engine, TrtDeleter{});
          } else {
            std::cerr << "[VPRNet] failed to deserialize engine from " << options_.engine_path << std::endl;
          }
        }
      }
    }
  }

  if (!engine_) {
    std::cout << "[VPRNet] engine missing or invalid, trying to build from ONNX: "
              << options_.onnx_path << std::endl;
    if (!std::filesystem::exists(options_.onnx_path)) {
      std::cerr << "[VPRNet] onnx file not found: " << options_.onnx_path << std::endl;
      return false;
    }

    auto builder = std::unique_ptr<nvinfer1::IBuilder, TrtDeleter>(nvinfer1::createInferBuilder(logger_));
    if (!builder) {
      std::cerr << "[VPRNet] failed to create TensorRT builder" << std::endl;
      return false;
    }
    const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition, TrtDeleter>(builder->createNetworkV2(explicit_batch));
    if (!network) {
      std::cerr << "[VPRNet] failed to create network" << std::endl;
      return false;
    }
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig, TrtDeleter>(builder->createBuilderConfig());
    if (!config) {
      std::cerr << "[VPRNet] failed to create builder config" << std::endl;
      return false;
    }
    auto parser = std::unique_ptr<nvonnxparser::IParser, TrtDeleter>(nvonnxparser::createParser(*network, logger_));
    if (!parser) {
      std::cerr << "[VPRNet] failed to create ONNX parser" << std::endl;
      return false;
    }
    if (!parser->parseFromFile(options_.onnx_path.c_str(), static_cast<int>(options_.verbose ? nvinfer1::ILogger::Severity::kINFO
                                                                                             : nvinfer1::ILogger::Severity::kERROR))) {
      std::cerr << "[VPRNet] failed to parse onnx: " << options_.onnx_path << std::endl;
      return false;
    }
    if (builder->platformHasFastFp16()) {
      config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    auto input = network->getInput(0);
    if (!input) {
      std::cerr << "[VPRNet] network has no input" << std::endl;
      return false;
    }

    nvinfer1::Dims input_dims = input->getDimensions();
    bool need_profile = false;
    for (int i = 0; i < input_dims.nbDims; ++i) {
      if (input_dims.d[i] < 0) {
        need_profile = true;
        break;
      }
    }
    if (need_profile) {
      auto profile = builder->createOptimizationProfile();
      if (!profile) {
        std::cerr << "[VPRNet] failed to create optimization profile" << std::endl;
        return false;
      }

      nvinfer1::Dims min_dims = input_dims;
      nvinfer1::Dims opt_dims = input_dims;
      nvinfer1::Dims max_dims = input_dims;
      if (input_dims.nbDims == 4) {
        if (input_dims.d[1] == 3 || input_dims.d[1] < 0) {
          min_dims.d[0] = opt_dims.d[0] = max_dims.d[0] = 1;
          min_dims.d[1] = opt_dims.d[1] = max_dims.d[1] = 3;
          min_dims.d[2] = max_dims.d[2] = std::max(1, options_.input_height / 2);
          opt_dims.d[2] = std::max(1, options_.input_height);
          min_dims.d[3] = max_dims.d[3] = std::max(1, options_.input_width / 2);
          opt_dims.d[3] = std::max(1, options_.input_width);
        }
      }
      if (!profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, min_dims) ||
          !profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, opt_dims) ||
          !profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, max_dims)) {
        std::cerr << "[VPRNet] failed to set optimization profile" << std::endl;
        return false;
      }
      config->addOptimizationProfile(profile);
    }

    std::unique_ptr<nvinfer1::IHostMemory, TrtDeleter> plan(builder->buildSerializedNetwork(*network, *config));
    if (!plan) {
      std::cerr << "[VPRNet] failed to build serialized network from onnx: " << options_.onnx_path << std::endl;
      return false;
    }

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
      std::cerr << "[VPRNet] failed to create TensorRT runtime" << std::endl;
      return false;
    }
    nvinfer1::ICudaEngine* raw_engine = runtime_->deserializeCudaEngine(plan->data(), plan->size());
    if (!raw_engine) {
      std::cerr << "[VPRNet] failed to deserialize built engine" << std::endl;
      return false;
    }
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(raw_engine, TrtDeleter{});
    if (!save_engine()) {
      std::cerr << "[VPRNet] failed to save built engine to: " << options_.engine_path << std::endl;
      return false;
    }
  }

  if (!engine_) {
    return false;
  }

  if (!context_) {
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
      std::cerr << "[VPRNet] failed to create execution context" << std::endl;
      return false;
    }
  }

  input_binding_index_ = -1;
  output_binding_index_ = -1;
  const int nb_bindings = engine_->getNbBindings();
  for (int i = 0; i < nb_bindings; ++i) {
    if (engine_->bindingIsInput(i)) {
      input_binding_index_ = i;
      input_dims_ = engine_->getBindingDimensions(i);
    } else if (output_binding_index_ < 0) {
      output_binding_index_ = i;
      output_dims_ = engine_->getBindingDimensions(i);
    }
  }

  if (input_binding_index_ < 0 || output_binding_index_ < 0) {
    std::cerr << "[VPRNet] failed to locate input/output bindings" << std::endl;
    return false;
  }

  input_layout_nhwc_ = is_nhwc_layout(input_dims_);
  input_layout_nchw_ = is_nchw_layout(input_dims_);
  if (!input_layout_nhwc_ && !input_layout_nchw_) {
    std::cerr << "[VPRNet] unsupported input layout, expect NHWC or NCHW" << std::endl;
    return false;
  }

  if (input_dims_.nbDims <= 0 || output_dims_.nbDims <= 0) {
    std::cerr << "[VPRNet] invalid binding dimensions" << std::endl;
    return false;
  }

  if (input_dims_.d[0] <= 0) {
    input_dims_.d[0] = 1;
  }
  if (input_layout_nhwc_) {
    if (input_dims_.d[1] <= 0) {
      input_dims_.d[1] = options_.input_height;
    }
    if (input_dims_.d[2] <= 0) {
      input_dims_.d[2] = options_.input_width;
    }
    if (input_dims_.d[3] <= 0) {
      input_dims_.d[3] = 3;
    }
  } else {
    if (input_dims_.d[1] <= 0) {
      input_dims_.d[1] = 3;
    }
    if (input_dims_.d[2] <= 0) {
      input_dims_.d[2] = options_.input_height;
    }
    if (input_dims_.d[3] <= 0) {
      input_dims_.d[3] = options_.input_width;
    }
  }

  if (!context_->setBindingDimensions(input_binding_index_, input_dims_)) {
    std::cerr << "[VPRNet] failed to set binding dimensions" << std::endl;
    return false;
  }

  output_dims_ = context_->getBindingDimensions(output_binding_index_);
  input_elements_ = dims_volume(input_dims_);
  output_elements_ = dims_volume(output_dims_);
  descriptor_dim_ = static_cast<int>(output_elements_);
  if (input_elements_ == 0 || output_elements_ == 0) {
    std::cerr << "[VPRNet] invalid tensor size" << std::endl;
    return false;
  }

  host_input_.assign(input_elements_, 0.0f);
  host_output_.assign(output_elements_, 0.0f);

  if (cudaMalloc(&device_input_, input_elements_ * sizeof(float)) != cudaSuccess) {
    std::cerr << "[VPRNet] cudaMalloc failed for input" << std::endl;
    device_input_ = nullptr;
    return false;
  }
  if (cudaMalloc(&device_output_, output_elements_ * sizeof(float)) != cudaSuccess) {
    std::cerr << "[VPRNet] cudaMalloc failed for output" << std::endl;
    cudaFree(device_input_);
    device_input_ = nullptr;
    device_output_ = nullptr;
    return false;
  }

  bindings_.assign(engine_->getNbBindings(), nullptr);
  bindings_[input_binding_index_] = device_input_;
  bindings_[output_binding_index_] = device_output_;

  if (!stream_created_) {
    if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) {
      std::cerr << "[VPRNet] failed to create CUDA stream" << std::endl;
      release_buffers();
      return false;
    }
    stream_created_ = true;
  }

  std::cout << "[VPRNet] loaded engine: " << options_.engine_path
            << ", descriptor_dim=" << descriptor_dim_
            << ", input_elements=" << input_elements_
            << ", output_elements=" << output_elements_ << std::endl;
  return true;
}

bool VPRNet::infer(const cv::Mat& image, Eigen::VectorXf& descriptor)
{
  std::vector<float> input;
  if (!preprocess(image, input)) {
    return false;
  }
  return infer_internal(input, descriptor);
}

bool VPRNet::save_engine(const std::string& dst_path) const
{
  if (!engine_) {
    return false;
  }

  const std::string& path = dst_path.empty() ? options_.engine_path : dst_path;
  std::shared_ptr<nvinfer1::IHostMemory> serialized(engine_->serialize(), TrtDeleter{});
  if (!serialized) {
    return false;
  }

  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    return false;
  }

  file.write(reinterpret_cast<const char*>(serialized->data()), serialized->size());
  return static_cast<bool>(file);
}

bool VPRNet::loaded() const
{
  return static_cast<bool>(engine_);
}

int VPRNet::descriptor_dim() const
{
  return descriptor_dim_;
}

const std::string& VPRNet::engine_path() const
{
  return options_.engine_path;
}

bool VPRNet::preprocess(const cv::Mat& image, std::vector<float>& input) const
{
  if (image.empty()) {
    return false;
  }

  cv::Mat rgb = to_rgb_mat(image);
  cv::Mat resized;
  cv::resize(rgb, resized, cv::Size(options_.input_width, options_.input_height));
  resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

  input.resize(static_cast<size_t>(options_.input_width) * static_cast<size_t>(options_.input_height) * 3U);
  const int plane_size = options_.input_width * options_.input_height;
  for (int y = 0; y < options_.input_height; ++y) {
    const cv::Vec3f* row = resized.ptr<cv::Vec3f>(y);
    for (int x = 0; x < options_.input_width; ++x) {
      const cv::Vec3f& pixel = row[x];
      const int idx = y * options_.input_width + x;
      input[idx] = (pixel[0] - kImageMean[0]) / kImageStd[0];
      input[plane_size + idx] = (pixel[1] - kImageMean[1]) / kImageStd[1];
      input[2 * plane_size + idx] = (pixel[2] - kImageMean[2]) / kImageStd[2];
    }
  }

  return true;
}

bool VPRNet::infer_internal(const std::vector<float>& input, Eigen::VectorXf& descriptor)
{
  if (!engine_ || !context_ || device_input_ == nullptr || device_output_ == nullptr) {
    return false;
  }
  if (input.size() != input_elements_ || descriptor_dim_ <= 0) {
    return false;
  }

  host_input_ = input;
  if (cudaMemcpyAsync(device_input_, host_input_.data(), host_input_.size() * sizeof(float),
                      cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
    return false;
  }

  if (!context_->executeV2(bindings_.data())) {
    return false;
  }

  if (cudaMemcpyAsync(host_output_.data(), device_output_, host_output_.size() * sizeof(float),
                      cudaMemcpyDeviceToHost, stream_) != cudaSuccess) {
    return false;
  }
  if (cudaStreamSynchronize(stream_) != cudaSuccess) {
    return false;
  }

  descriptor.resize(descriptor_dim_);
  for (int i = 0; i < descriptor_dim_; ++i) {
    descriptor(i) = host_output_[static_cast<size_t>(i)];
  }
  return true;
}

size_t VPRNet::dims_volume(const nvinfer1::Dims& dims)
{
  if (dims.nbDims <= 0) {
    return 0;
  }
  size_t volume = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] <= 0) {
      return 0;
    }
    volume *= static_cast<size_t>(dims.d[i]);
  }
  return volume;
}

bool VPRNet::is_nhwc_layout(const nvinfer1::Dims& dims)
{
  return dims.nbDims == 4 && dims.d[3] == 3;
}

bool VPRNet::is_nchw_layout(const nvinfer1::Dims& dims)
{
  return dims.nbDims == 4 && dims.d[1] == 3;
}

cv::Mat VPRNet::to_rgb_mat(const cv::Mat& image)
{
  cv::Mat rgb;
  if (image.channels() == 3) {
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
  } else if (image.channels() == 4) {
    cv::cvtColor(image, rgb, cv::COLOR_BGRA2RGB);
  } else if (image.channels() == 1) {
    cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
  } else {
    rgb = image.clone();
  }
  return rgb;
}

void VPRNet::release_buffers()
{
  if (device_input_) {
    cudaFree(device_input_);
    device_input_ = nullptr;
  }
  if (device_output_) {
    cudaFree(device_output_);
    device_output_ = nullptr;
  }
  bindings_.clear();
  host_input_.clear();
  host_output_.clear();
  input_elements_ = 0;
  output_elements_ = 0;
  descriptor_dim_ = 0;
}
