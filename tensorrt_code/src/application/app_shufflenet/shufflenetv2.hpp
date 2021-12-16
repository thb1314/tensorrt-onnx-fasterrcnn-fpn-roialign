#ifndef SHUFFLENETV2
#define SHUFFLENETV2

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>
#include <common/trt_tensor.hpp>


namespace ShuffleNetV2 {

    using namespace std;

    void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, int ibatch);

    class Infer{
        public:
            virtual shared_future<vector<float>> commit(const cv::Mat& image) = 0;
            virtual vector<shared_future<vector<float>>> commits(const vector<cv::Mat>& images) = 0;
    };

    shared_ptr<Infer> create_infer(const string& engine_file,int gpu_id = 0);

}; // namespace Yolo

#endif // YOLO_HPP