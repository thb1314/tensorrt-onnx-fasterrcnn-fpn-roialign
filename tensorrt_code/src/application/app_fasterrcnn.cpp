
#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include "app_fasterrcnn/fasterrcnn.hpp"

using namespace std;

static const char* cocolabels[] = {
    "person","bicycle","car","motorcycle","airplane",
    "bus","train","truck","boat","traffic_light","fire_hydrant","",
    "stop_sign","parking_meter","bench","bird",
    "cat","dog","horse","sheep","cow","elephant","bear",
    "zebra","giraffe","","backpack","umbrella","","","handbag",
    "tie","suitcase","frisbee","skis","snowboard","sports_ball","kite","baseball_bat",
    "baseball_glove","skateboard","surfboard",
    "tennis_racket","bottle","","wine_glass","cup","fork","knife","spoon",
    "bowl","banana","apple","sandwich","orange","broccoli",
    "carrot","hot_dog","pizza","donut","cake","chair","couch",
    "potted_plant","bed","","dining_table","","","toilet","","tv",
    "laptop","mouse","remote","keyboard","cell_phone","microwave",
    "oven","toaster","sink","refrigerator","","book","clock","vase",
    "scissors","teddy_bear","hair_drier","toothbrush"
};

bool requires(const char* name);

static void append_to_file(const string& file, const string& data){
    FILE* f = fopen(file.c_str(), "a+");
    if(f == nullptr){
        INFOE("Open %s failed.", file.c_str());
        return;
    }

    fprintf(f, "%s\n", data.c_str());
    fclose(f);
}

static void inference_and_performance(int deviceid, const vector<string>& engine_files, TRT::Mode mode, FasterRCNN::Type type, const string& model_name){
    assert(2 == engine_files.size());
    auto engine = FasterRCNN::create_infer(engine_files, type, deviceid, 0.5f, 0.3f);
    auto engine_file = engine_files[0];
    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    auto files = iLogger::find_files("inference_fasterrcnn", "*.jpg;*.jpeg;*.png;*.gif;*.tif");
    vector<cv::Mat> images;
    for(int i = 0; i < files.size(); ++i){
        auto image = cv::imread(files[i]);
        images.emplace_back(image);
    }

    // warmup
    vector<shared_future<FasterRCNN::BoxArray>> boxes_array;
    for(int i = 0; i < 10; ++i)
        boxes_array = engine->commits(images);
    boxes_array.back().get();
    boxes_array.clear();
    
    /////////////////////////////////////////////////////////
    const int ntest = 100;
    auto begin_timer = iLogger::timestamp_now_float();

    for(int i  = 0; i < ntest; ++i)
        boxes_array = engine->commits(images);
    
    // wait all result
    boxes_array.back().get();

    float inference_average_time = (iLogger::timestamp_now_float() - begin_timer) / ntest / images.size();
    auto type_name = FasterRCNN::type_name(type);
    auto mode_name = TRT::mode_string(mode);
    INFO("%s[%s] average: %.2f ms / image, FPS: %.2f", engine_file.c_str(), type_name, inference_average_time, 1000 / inference_average_time);
    append_to_file("perf.result.log", iLogger::format("%s,%s,%s,%f", model_name.c_str(), type_name, mode_name, inference_average_time));

    string root = iLogger::format("%s_%s_%s_result", model_name.c_str(), type_name, mode_name);
    iLogger::rmtree(root);
    iLogger::mkdir(root);

    for(int i = 0; i < boxes_array.size(); ++i){

        auto& image = images[i];
        auto boxes  = boxes_array[i].get();
        
        for(auto& obj : boxes) {
            uint8_t b, g, r;
            tie(b, g, r) = iLogger::random_color(obj.class_label);
            cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 1);
            auto name    = cocolabels[obj.class_label];
            auto caption = iLogger::format("%s %.2f", name, obj.confidence);
            int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(image, cv::Point(obj.left-3, obj.top-33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
            cv::putText(image, caption, cv::Point(obj.left, obj.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
        }

        string file_name = iLogger::file_name(files[i], false);
        string save_path = iLogger::format("%s/%s.jpg", root.c_str(), file_name.c_str());
        INFO("Save to %s, %d object, average time %.2f ms", save_path.c_str(), boxes.size(), inference_average_time);
        cv::imwrite(save_path, image);
    }
}

static void test(FasterRCNN::Type type, TRT::Mode mode, const string& model, const string& sub_model){

    int deviceid = 0;
    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(deviceid);

    auto int8process = [=](int current, int count, const vector<string>& files, shared_ptr<TRT::Tensor>& tensor){

        INFO("Int8 %d / %d", current, count);

        for(int i = 0; i < files.size(); ++i){
            auto image = cv::imread(files[i]);
            FasterRCNN::image_to_tensor(image, tensor, type, i);
        }
    };

    const char* name = model.c_str();
    INFO("===================== test %s %s %s ==================================", FasterRCNN::type_name(type), mode_name, name);

    if(not requires(name))
        return;

    string onnx_file = iLogger::format("%s.onnx", name);
    string model_file = iLogger::format("%s.%s.trtmodel", name, mode_name);
    int test_batch_size = 1;
    
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
    const char* subname = sub_model.c_str();
    string sub_onnx_file = iLogger::format("%s.onnx", subname);
    string sub_model_file = iLogger::format("%s.%s.trtmodel", subname, mode_name);
    if(not iLogger::exists(sub_model_file)){
        TRT::compile(
            mode,                       // FP32、FP16、INT8
            1024 * 5,            // max batch size
            sub_onnx_file,                  // source 
            sub_model_file,                 // save to
            {},
            int8process,
            "inference"
        );
    }

    inference_and_performance(deviceid, {model_file, sub_model_file}, mode, type, name);
}

int app_fasterrcnn(){

    //iLogger::set_log_level(iLogger::LogLevel::Debug);
    // test(FasterRCNN::Type::FasterRCNN, TRT::Mode::FP32, "fasterrpn_resnet50_fpn");
    test(FasterRCNN::Type::FasterRCNN, TRT::Mode::FP32, "rpn_backbone_resnet50", "new_header");

    return 0;
}