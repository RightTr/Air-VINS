#pragma once

#include <NvInfer.h>

#include <Eigen/Core>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#include <memory>
#include <string>
#include <vector>

class NetVLADTRT {
 public:
  struct Options {
    std::string engine_path = "/home/pi/netvlad_tensorrt/models/netvlad.trt";
    std::string debug_output_path;
    int input_width = 640;
    int input_height = 480;
    bool verbose = false;
  };

  explicit NetVLADTRT(const Options& options = Options());
  explicit NetVLADTRT(const std::string& engine_path,
                      int input_width = 640,
                      int input_height = 480);
  ~NetVLADTRT();

  bool build();
  bool deserialize_engine();
  bool infer(const cv::Mat& image, Eigen::VectorXf& descriptor);
  bool save_engine(const std::string& dst_path = "") const;

  bool loaded() const;
  int descriptor_dim() const;
  const std::string& engine_path() const;

 private:
  struct TrtDeleter {
    template <typename T>
    void operator()(T* ptr) const
    {
      if (ptr) {
        ptr->destroy();
      }
    }
  };

  class TrtLogger : public nvinfer1::ILogger {
   public:
    explicit TrtLogger(bool verbose = false);
    void log(Severity severity, const char* msg) noexcept override;

   private:
    bool verbose_;
  };

  bool load_engine_file(std::vector<char>& data) const;
  bool initialize_runtime();
  bool resolve_bindings();
  bool allocate_buffers();
  void release_buffers();

  bool preprocess(const cv::Mat& image, std::vector<float>& input) const;
  bool infer_internal(const std::vector<float>& input, Eigen::VectorXf& descriptor);

  static size_t dims_volume(const nvinfer1::Dims& dims);
  static bool is_nhwc_layout(const nvinfer1::Dims& dims);
  static bool is_nchw_layout(const nvinfer1::Dims& dims);
  static cv::Mat to_rgb_mat(const cv::Mat& image);

  Options options_;
  TrtLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime, TrtDeleter> runtime_;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IExecutionContext> context_;

  int input_binding_index_;
  int output_binding_index_;
  nvinfer1::Dims input_dims_;
  nvinfer1::Dims output_dims_;
  bool input_layout_nhwc_;
  bool input_layout_nchw_;

  std::vector<void*> bindings_;
  void* device_input_;
  void* device_output_;
  std::vector<float> host_input_;
  std::vector<float> host_output_;
  size_t input_elements_;
  size_t output_elements_;
  int descriptor_dim_;

  cudaStream_t stream_;
  bool stream_created_;
};

