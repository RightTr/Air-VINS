#include "netvlad_trt.h"

#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {
constexpr float kImageMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kImageStd[3] = {0.229f, 0.224f, 0.225f};
}  // namespace

NetVLADTRT::TrtLogger::TrtLogger(bool verbose) : verbose_(verbose) {}

void NetVLADTRT::TrtLogger::log(Severity severity, const char* msg) noexcept
{
  if (!msg) {
    return;
  }
  if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
    std::cerr << "[NetVLADTRT][TRT] " << msg << std::endl;
  } else if (verbose_ && severity <= Severity::kWARNING) {
    std::cout << "[NetVLADTRT][TRT] " << msg << std::endl;
  }
}

NetVLADTRT::NetVLADTRT(const Options& options)
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

NetVLADTRT::NetVLADTRT(const std::string& engine_path, int input_width, int input_height)
    : NetVLADTRT(Options{})
{
  options_.engine_path = engine_path;
  options_.input_width = input_width;
  options_.input_height = input_height;
}

NetVLADTRT::~NetVLADTRT()
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

bool NetVLADTRT::build()
{
  return deserialize_engine();
}

bool NetVLADTRT::load_engine_file(std::vector<char>& data) const
{
  std::ifstream file(options_.engine_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::cerr << "[NetVLADTRT] failed to open engine file: " << options_.engine_path << std::endl;
    return false;
  }

  const std::streamsize size = file.tellg();
  if (size <= 0) {
    std::cerr << "[NetVLADTRT] engine file is empty: " << options_.engine_path << std::endl;
    return false;
  }

  data.resize(static_cast<size_t>(size));
  file.seekg(0, std::ios::beg);
  if (!file.read(data.data(), size)) {
    std::cerr << "[NetVLADTRT] failed to read engine file: " << options_.engine_path << std::endl;
    return false;
  }
  return true;
}

bool NetVLADTRT::initialize_runtime()
{
  if (runtime_) {
    return true;
  }
  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (!runtime_) {
    std::cerr << "[NetVLADTRT] failed to create TensorRT runtime" << std::endl;
    return false;
  }
  return true;
}

bool NetVLADTRT::deserialize_engine()
{
  release_buffers();
  context_.reset();
  engine_.reset();

  if (!initialize_runtime()) {
    return false;
  }

  std::vector<char> engine_data;
  if (!load_engine_file(engine_data)) {
    return false;
  }

  nvinfer1::ICudaEngine* raw_engine = runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size());
  if (!raw_engine) {
    std::cerr << "[NetVLADTRT] failed to deserialize engine from " << options_.engine_path << std::endl;
    return false;
  }
  engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(raw_engine, TrtDeleter{});

  nvinfer1::IExecutionContext* raw_context = engine_->createExecutionContext();
  if (!raw_context) {
    std::cerr << "[NetVLADTRT] failed to create execution context" << std::endl;
    engine_.reset();
    return false;
  }
  context_ = std::shared_ptr<nvinfer1::IExecutionContext>(raw_context, TrtDeleter{});

  if (!resolve_bindings()) {
    context_.reset();
    engine_.reset();
    return false;
  }

  if (!allocate_buffers()) {
    context_.reset();
    engine_.reset();
    return false;
  }

  if (!stream_created_) {
    if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) {
      std::cerr << "[NetVLADTRT] failed to create CUDA stream" << std::endl;
      release_buffers();
      context_.reset();
      engine_.reset();
      return false;
    }
    stream_created_ = true;
  }

  std::cout << "[NetVLADTRT] loaded engine: " << options_.engine_path
            << ", descriptor_dim=" << descriptor_dim_
            << ", input_elements=" << input_elements_
            << ", output_elements=" << output_elements_ << std::endl;
  return true;
}

bool NetVLADTRT::resolve_bindings()
{
  if (!engine_ || !context_) {
    return false;
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
    std::cerr << "[NetVLADTRT] failed to locate input/output bindings" << std::endl;
    return false;
  }

  input_layout_nhwc_ = is_nhwc_layout(input_dims_);
  input_layout_nchw_ = is_nchw_layout(input_dims_);
  if (!input_layout_nhwc_ && !input_layout_nchw_) {
    std::cerr << "[NetVLADTRT] unsupported input layout, expect NHWC or NCHW" << std::endl;
    return false;
  }

  if (input_dims_.nbDims <= 0 || output_dims_.nbDims <= 0) {
    std::cerr << "[NetVLADTRT] invalid binding dimensions" << std::endl;
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
    std::cerr << "[NetVLADTRT] failed to set binding dimensions" << std::endl;
    return false;
  }

  output_dims_ = context_->getBindingDimensions(output_binding_index_);
  input_elements_ = dims_volume(input_dims_);
  output_elements_ = dims_volume(output_dims_);
  descriptor_dim_ = static_cast<int>(output_elements_);

  if (input_elements_ == 0 || output_elements_ == 0) {
    std::cerr << "[NetVLADTRT] invalid tensor size" << std::endl;
    return false;
  }
  return true;
}

bool NetVLADTRT::allocate_buffers()
{
  release_buffers();
  host_input_.assign(input_elements_, 0.0f);
  host_output_.assign(output_elements_, 0.0f);

  if (cudaMalloc(&device_input_, input_elements_ * sizeof(float)) != cudaSuccess) {
    std::cerr << "[NetVLADTRT] cudaMalloc failed for input" << std::endl;
    device_input_ = nullptr;
    return false;
  }
  if (cudaMalloc(&device_output_, output_elements_ * sizeof(float)) != cudaSuccess) {
    std::cerr << "[NetVLADTRT] cudaMalloc failed for output" << std::endl;
    cudaFree(device_input_);
    device_input_ = nullptr;
    device_output_ = nullptr;
    return false;
  }

  bindings_.assign(engine_->getNbBindings(), nullptr);
  bindings_[input_binding_index_] = device_input_;
  bindings_[output_binding_index_] = device_output_;
  return true;
}

void NetVLADTRT::release_buffers()
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

bool NetVLADTRT::preprocess(const cv::Mat& image, std::vector<float>& input) const
{
  if (image.empty()) {
    return false;
  }

  cv::Mat rgb = to_rgb_mat(image);
  cv::Mat resized;
  cv::resize(rgb, resized, cv::Size(options_.input_width, options_.input_height), 0, 0, cv::INTER_LINEAR);
  cv::Mat float_image;
  resized.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

  input.assign(input_elements_, 0.0f);

  if (input_layout_nhwc_) {
    size_t idx = 0;
    for (int h = 0; h < float_image.rows; ++h) {
      const cv::Vec3f* row_ptr = float_image.ptr<cv::Vec3f>(h);
      for (int w = 0; w < float_image.cols; ++w) {
        const cv::Vec3f& pix = row_ptr[w];
        input[idx++] = (pix[0] - kImageMean[0]) / kImageStd[0];
        input[idx++] = (pix[1] - kImageMean[1]) / kImageStd[1];
        input[idx++] = (pix[2] - kImageMean[2]) / kImageStd[2];
      }
    }
  } else {
    const int channel_stride = options_.input_height * options_.input_width;
    for (int h = 0; h < float_image.rows; ++h) {
      const cv::Vec3f* row_ptr = float_image.ptr<cv::Vec3f>(h);
      for (int w = 0; w < float_image.cols; ++w) {
        const cv::Vec3f& pix = row_ptr[w];
        const int base = h * options_.input_width + w;
        input[0 * channel_stride + base] = (pix[0] - kImageMean[0]) / kImageStd[0];
        input[1 * channel_stride + base] = (pix[1] - kImageMean[1]) / kImageStd[1];
        input[2 * channel_stride + base] = (pix[2] - kImageMean[2]) / kImageStd[2];
      }
    }
  }

  return true;
}

bool NetVLADTRT::infer_internal(const std::vector<float>& input, Eigen::VectorXf& descriptor)
{
  if (!context_ || bindings_.empty() || !stream_created_) {
    return false;
  }

  if (input.size() != input_elements_) {
    std::cerr << "[NetVLADTRT] input size mismatch" << std::endl;
    return false;
  }

  if (cudaMemcpyAsync(device_input_, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
    std::cerr << "[NetVLADTRT] cudaMemcpyAsync H2D failed" << std::endl;
    return false;
  }

  if (!context_->enqueueV2(bindings_.data(), stream_, nullptr)) {
    std::cerr << "[NetVLADTRT] enqueueV2 failed" << std::endl;
    return false;
  }

  if (cudaMemcpyAsync(host_output_.data(), device_output_, host_output_.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_) != cudaSuccess) {
    std::cerr << "[NetVLADTRT] cudaMemcpyAsync D2H failed" << std::endl;
    return false;
  }
  if (cudaStreamSynchronize(stream_) != cudaSuccess) {
    std::cerr << "[NetVLADTRT] cudaStreamSynchronize failed" << std::endl;
    return false;
  }

  descriptor.resize(static_cast<Eigen::Index>(host_output_.size()));
  for (size_t i = 0; i < host_output_.size(); ++i) {
    descriptor(static_cast<Eigen::Index>(i)) = host_output_[i];
  }

  const float norm = descriptor.norm();
  if (norm > 1e-8f) {
    descriptor /= norm;
  }
  return true;
}

bool NetVLADTRT::infer(const cv::Mat& image, Eigen::VectorXf& descriptor)
{
  if (!loaded()) {
    return false;
  }

  std::vector<float> input;
  if (!preprocess(image, input)) {
    return false;
  }
  return infer_internal(input, descriptor);
}

bool NetVLADTRT::save_engine(const std::string& dst_path) const
{
  if (!engine_) {
    return false;
  }

  std::unique_ptr<nvinfer1::IHostMemory, TrtDeleter> serialized(engine_->serialize());
  if (!serialized) {
    std::cerr << "[NetVLADTRT] failed to serialize engine" << std::endl;
    return false;
  }

  const std::string output_path = dst_path.empty() ? options_.engine_path : dst_path;
  std::ofstream out(output_path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    std::cerr << "[NetVLADTRT] failed to open output file: " << output_path << std::endl;
    return false;
  }

  out.write(static_cast<const char*>(serialized->data()), serialized->size());
  return static_cast<bool>(out);
}

bool NetVLADTRT::loaded() const
{
  return engine_ != nullptr && context_ != nullptr;
}

int NetVLADTRT::descriptor_dim() const
{
  return descriptor_dim_;
}

const std::string& NetVLADTRT::engine_path() const
{
  return options_.engine_path;
}

size_t NetVLADTRT::dims_volume(const nvinfer1::Dims& dims)
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

bool NetVLADTRT::is_nhwc_layout(const nvinfer1::Dims& dims)
{
  return dims.nbDims == 4 && dims.d[3] == 3;
}

bool NetVLADTRT::is_nchw_layout(const nvinfer1::Dims& dims)
{
  return dims.nbDims == 4 && dims.d[1] == 3;
}

cv::Mat NetVLADTRT::to_rgb_mat(const cv::Mat& image)
{
  cv::Mat rgb;
  if (image.channels() == 1) {
    cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
  } else if (image.channels() == 3) {
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
  } else if (image.channels() == 4) {
    cv::cvtColor(image, rgb, cv::COLOR_BGRA2RGB);
  } else {
    throw std::runtime_error("unsupported image channel count");
  }
  return rgb;
}
