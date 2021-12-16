#include "app_shufflenet/shufflenetv2.hpp"
#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <fstream>


using namespace std;


bool requires(const char* name);

vector<string> get_classes_from_file(const string& filepath) {
    vector<string> ret;
    ifstream file(filepath);
    if(!file.is_open()) return ret;
    string line;
    while(getline(file, line))
        ret.push_back(line);
    return ret;
}

static void inference_and_performance(int deviceid, const string& engine_file, TRT::Mode mode, const string& model_name){

    auto engine = ShuffleNetV2::create_infer(engine_file, deviceid);
    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    auto files = iLogger::find_files("classification", "*.jpg;*.jpeg;*.png;*.gif;*.tif");
    vector<cv::Mat> images;
    vector<string> filepath_list;
    for(int i = 0; i < files.size(); ++i) {
        auto image = cv::imread(files[i]);
        filepath_list.push_back(files[i]);
        images.emplace_back(image);
    }

    // warmup
    vector<shared_future<vector<float>>> classification_array;
    for(int i = 0; i < 10; ++i)
        classification_array = engine->commits(images);
    classification_array.back().get();
    classification_array.clear();
    
    /////////////////////////////////////////////////////////
    const int ntest = 100;
    auto begin_timer = iLogger::timestamp_now_float();

    for(int i  = 0; i < ntest; ++i)
        classification_array = engine->commits(images);
    
    // wait all result
    classification_array.back().get();

    float inference_average_time = (iLogger::timestamp_now_float() - begin_timer) / ntest / images.size();
    const char* type_name = "ShuffleNetV2";
    auto mode_name = TRT::mode_string(mode);
    INFO("%s[%s] average: %.2f ms / image, FPS: %.2f", engine_file.c_str(), type_name, inference_average_time, 1000 / inference_average_time);
    
    vector<string> index2classification = get_classes_from_file("labels.imagenet.txt");
    int i = 0;
    for(auto& vec_future: classification_array)
    {
        auto& vec = vec_future.get();
        int index = (max_element(vec.begin(), vec.end()) - vec.begin());
        INFO("%s, out max_index: %d %s", filepath_list[i++].c_str(), index, index2classification[index].c_str());
    }
}

static void test(TRT::Mode mode, const string& model){

    int deviceid = 0;
    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(deviceid);

    auto int8process = [=](int current, int count, const vector<string>& files, shared_ptr<TRT::Tensor>& tensor){

        INFO("Int8 %d / %d", current, count);

        for(int i = 0; i < files.size(); ++i){
            auto image = cv::imread(files[i]);
            ShuffleNetV2::image_to_tensor(image, tensor, i);
        }
    };

    const char* name = model.c_str();
    INFO("===================== test %s %s ==================================", mode_name, name);

    if(not requires(name))
        return;

    string onnx_file = iLogger::format("%s.onnx", name);
    string model_file = iLogger::format("%s.%s.trtmodel", name, mode_name);
    int test_batch_size = 16;
    
    if(not iLogger::exists(model_file)){
        TRT::compile(
            mode,                       // FP32、FP16、INT8
            test_batch_size,            // max batch size
            onnx_file,                  // source 
            model_file,                 // save to
            {},
            int8process,
            "inference"
        );
    }

    inference_and_performance(deviceid, model_file, mode, name);
}

int app_shufflenetv2(){

    //iLogger::set_log_level(iLogger::LogLevel::Debug);
    test(TRT::Mode::FP32, "shufflenet_v2_x0_5");

    return 0;
}