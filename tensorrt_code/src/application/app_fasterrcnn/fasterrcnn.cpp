#include "fasterrcnn.hpp"
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include <common/infer_controller.hpp>
#include <common/preprocess_kernel.cuh>
#include <common/monopoly_allocator.hpp>
#include <common/cuda_tools.hpp>


namespace FasterRCNN{
    using namespace cv;
    using namespace std;

    const char* type_name(Type type){
        switch(type){
        case Type::FasterRCNN: return "FasterRCNN";
        default: return "Unknow";
        }
    }

    void decode_kernel_invoker(
        float* predict, float* scores, int num_bboxes, int num_classes, float confidence_threshold, 
        float nms_threshold, float* invert_affine_matrix, float* parray,
        int max_objects, cudaStream_t stream
    );

    void rpn_decode_kernel_invoker(float* predict, int num_bboxes, float nms_threshold, 
                                   float* invert_affine_matrix, float* parray, int max_objects, int batch_index, cudaStream_t stream);
    
    void RoiAlignImpl_Float(
  cudaStream_t stream,
  const int64_t nthreads,
  const float** bottom_data_list,
  const float* spatial_scales,
  const int64_t channels,
  const int64_t* heights,
  const int64_t* widths,
  const int64_t pooled_height,
  const int64_t pooled_width,
  const int64_t sampling_ratio,
  const float* bottom_rois,
  int64_t roi_cols,
  float* top_data,
  float* proposals,
  const bool is_mode_avg);

    struct AffineMatrix{
        float i2d[6];       // image to dst(network), 2x3 matrix
        float d2i[6];       // dst to image, 2x3 matrix

        void compute(const cv::Size& from, const cv::Size& to){
            float scale_x = to.width / (float)from.width;
            float scale_y = to.height / (float)from.height;

            // 这里取min的理由是
            // 1. M矩阵是 M @ from = to的方式进行映射，因此scale的分母一定是from
            // 2. 取最小，即根据宽高比，算出最小的比例，如果取最大，则势必有一部分超出图像范围而被裁剪掉，这不是我们要的
            // **
            // 保证 to/from越小越好 from越大to越小
            float scale = std::min(scale_x, scale_y);

            /**
            这里的仿射变换矩阵实质上是2x3的矩阵，具体实现是
            scale, 0, -scale * from.width * 0.5 + to.width * 0.5
            0, scale, -scale * from.height * 0.5 + to.height * 0.5
            
            这里可以想象成，是经历过缩放、平移、平移三次变换后的组合，M = TPS
            例如第一个S矩阵，定义为把输入的from图像，等比缩放scale倍，到to尺度下
            S = [
            scale,     0,      0
            0,     scale,      0
            0,         0,      1
            ]
            
            P矩阵定义为第一次平移变换矩阵，将图像的原点，从左上角，移动到缩放(scale)后图像的中心上
            P = [
            1,        0,      -scale * from.width * 0.5
            0,        1,      -scale * from.height * 0.5
            0,        0,                1
            ]
            T矩阵定义为第二次平移变换矩阵，将图像从原点移动到目标（to）图的中心上
            T = [
            1,        0,      to.width * 0.5,
            0,        1,      to.height * 0.5,
            0,        0,            1
            ]
            通过将3个矩阵顺序乘起来，即可得到下面的表达式：
            M = [
            scale,    0,     -scale * from.width * 0.5 + to.width * 0.5
            0,     scale,    -scale * from.height * 0.5 + to.height * 0.5
            0,        0,                     1
            ]
            去掉第三行就得到opencv需要的输入2x3矩阵
            **/

            i2d[0] = scale;  i2d[1] = 0;  i2d[2] = -scale * from.width  * 0.5  + to.width * 0.5;
            i2d[3] = 0;  i2d[4] = scale;  i2d[5] = -scale * from.height * 0.5 + to.height * 0.5;

            cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
            cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
            cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
        }

        cv::Mat i2d_mat(){
            return cv::Mat(2, 3, CV_32F, i2d);
        }
    };

    using ControllerImpl = InferController
    <
        Mat,                    // input
        BoxArray,         // output
        tuple<string, int>,     // start param
        AffineMatrix            // additional
    >;
    class InferImpl : public Infer, public ControllerImpl{
    public:

        /** 要求在InferImpl里面执行stop，而不是在基类执行stop **/
        virtual ~InferImpl(){
            stop();
        }

        virtual bool startup(const vector<string>& files, Type type, int gpuid, float confidence_threshold, float nms_threshold){

            if(type == Type::FasterRCNN){
                float mean[] = {0.485, 0.456, 0.406};
                float std[]  = {0.229, 0.224, 0.225};
                normalize_ = CUDAKernel::Norm::mean_std(mean, std, 1/255.0f, CUDAKernel::ChannelType::Invert);
            }else{
                INFOE("Unsupport type %d", type);
            }
            
            confidence_threshold_ = confidence_threshold;
            nms_threshold_        = nms_threshold;
            string file_str = files[0];
            for(int i = 1; i < (int)files.size(); ++i)
            {
                file_str += "|";
                file_str += files[i];
            }
            return ControllerImpl::startup(make_tuple(file_str, gpuid));
        }

        virtual void worker(promise<bool>& result) override{

            string files = get<0>(start_param_);
            int gpuid   = get<1>(start_param_);

            int split_index = files.find('|');
            assert(split_index != files.npos);

            string file = files.substr(0, split_index);
            string sub_file = files.substr(split_index + 1);

            TRT::set_device(gpuid);
            auto engine = TRT::load_infer(file);
            auto sub_engine = TRT::load_infer(sub_file);
            if(engine == nullptr){
                INFOE("Engine %s load failed", file.c_str());
                result.set_value(false);
                return;
            }
            if(sub_engine == nullptr){
                INFOE("Engine %s load failed", sub_file.c_str());
                result.set_value(false);
                return;
            }

            engine->print();
            sub_engine->print();

            const int MAX_IMAGE_BBOX  = 1024 * 5;
            const int RPN_NUM_BOX_ELEMENT = 9;      // left, top, right, bottom, score, level, keepflag, fpn_level, batch_index
            const int NUM_BOX_ELEMENT = 9;      // left, top, right, bottom, score, level, keepflag, not set, not set
            TRT::Tensor affin_matrix_device(TRT::DataType::Float);
            TRT::Tensor output_array_device(TRT::DataType::Float);
            
            int max_batch_size = min(engine->get_max_batch_size(), sub_engine->get_max_batch_size());
            
            auto input         = engine->tensor("input");
            auto output        = engine->tensor("rpn_boxes");
            auto feature_0     = engine->tensor("feature_0");
            auto feature_1     = engine->tensor("feature_1");
            auto feature_2     = engine->tensor("feature_2");
            auto feature_3     = engine->tensor("feature_3");
            // int num_classes    = output->size(2) - 5;

            input_width_       = input->size(3);
            input_height_      = input->size(2);
            tensor_allocator_  = make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
            stream_            = engine->get_stream();
            sub_engine->set_stream(stream_);
            cudaStream_t substream = sub_engine->get_stream();
            assert(substream == stream_);
            gpu_               = gpuid;
            

            auto proposals = sub_engine->tensor("proposals");
            auto roialigned_feature = sub_engine->tensor("roialigned_feature");
            auto boxes = sub_engine->tensor("boxes");
            auto scores = sub_engine->tensor("scores");
            
            result.set_value(true);

            input->resize_single_dim(0, max_batch_size).to_gpu();
            affin_matrix_device.set_stream(stream_);

            // 这里8个值的目的是保证 8 * sizeof(float) % 32 == 0
            affin_matrix_device.resize(max_batch_size, 8).to_gpu();

            // 这里的 1 + MAX_IMAGE_BBOX 结构是，counter + bboxes ...
            output_array_device.resize(max_batch_size, 1 + MAX_IMAGE_BBOX * RPN_NUM_BOX_ELEMENT).to_gpu();
            
            
            
            vector<Job> fetch_jobs;

            
            while(get_jobs_and_wait(fetch_jobs, max_batch_size)){

                int infer_batch_size = fetch_jobs.size();
                input->resize_single_dim(0, infer_batch_size);

                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    auto& job  = fetch_jobs[ibatch];
                    auto& mono = job.mono_tensor->data();
                    affin_matrix_device.copy_from_gpu(affin_matrix_device.offset(ibatch), mono->get_workspace()->gpu(), 6);
                    input->copy_from_gpu(input->offset(ibatch), mono->gpu(), mono->count());
                    job.mono_tensor->release();
                }

                engine->forward(false);
                engine->synchronize();
                output_array_device.to_gpu(false);
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    
                    auto& job                 = fetch_jobs[ibatch];
                    float* image_based_output = output->gpu<float>(ibatch);
                    float* output_array_ptr   = output_array_device.gpu<float>(ibatch);
                    auto affine_matrix        = affin_matrix_device.gpu<float>(ibatch);
                    checkCudaRuntime(cudaMemsetAsync(output_array_ptr, 0, sizeof(float), stream_));
                    rpn_decode_kernel_invoker(image_based_output, output->size(1), 0.7, affine_matrix, output_array_ptr, MAX_IMAGE_BBOX, ibatch, stream_);
                }

                // 使用 roi align
                const float* bottom_data_list[] = {(const float*)feature_0->gpu(), (const float*)feature_1->gpu(), (const float*)feature_2->gpu(), (const float*)feature_3->gpu()};
                TRT::Tensor bottom_data_list_tensor(TRT::DataType::Float);
                bottom_data_list_tensor.resize(sizeof(bottom_data_list) / sizeof(const float*), sizeof(const float*) / sizeof(float));
                memcpy(bottom_data_list_tensor.cpu(), &bottom_data_list[0], sizeof(bottom_data_list));
                bottom_data_list_tensor.to_gpu(true);

                float spatial_scales[] = {1 / 4.0, 1 / 8.0, 1 / 16.0, 1 / 32.0};
                TRT::Tensor spatial_scales_tensor(TRT::DataType::Float);
                spatial_scales_tensor.resize(4);
                memcpy(spatial_scales_tensor.cpu(), spatial_scales, 4 * sizeof(float));
                spatial_scales_tensor.to_gpu(true);

                int64_t heights[] = {feature_0->size(2), feature_1->size(2), feature_2->size(2), feature_3->size(2)}; 
                TRT::Tensor heights_tensor(TRT::DataType::Float);
                heights_tensor.resize(4, sizeof(int64_t) / sizeof(float));
                memcpy(heights_tensor.cpu(), heights, 4 * sizeof(int64_t));
                heights_tensor.to_gpu(true);

                int64_t widths[] = {feature_0->size(3), feature_1->size(3), feature_2->size(3), feature_3->size(3)}; 
                TRT::Tensor widths_tensor(TRT::DataType::Float);
                widths_tensor.resize(4, sizeof(int64_t) / sizeof(float));
                memcpy(widths_tensor.cpu(), widths, 4 * sizeof(int64_t));
                widths_tensor.to_gpu(true);

                TRT::Tensor roi_align_inputs(TRT::DataType::Float);
                roi_align_inputs.resize(infer_batch_size * MAX_IMAGE_BBOX * 6);
                roi_align_inputs.to_cpu(false);
                
                
                output_array_device.to_cpu(true);
                int roi_align_inputs_index = 0;
                float* roi_align_inputs_cpu_ptr = (float*)roi_align_inputs.cpu();
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    float* parray = output_array_device.cpu<float>(ibatch);
                    int count     = min(MAX_IMAGE_BBOX, (int)*parray);
  
                    for(int i = 0; i < count; ++i) {
                        float* pbox  = parray + 1 + i * RPN_NUM_BOX_ELEMENT;
                        int keepflag = pbox[6];
                        if(keepflag == 1) {
                            // left, top, right, bottom, score, level, keepflag, fpn_level, batch_index
                            roi_align_inputs_cpu_ptr[roi_align_inputs_index++] = pbox[0];
                            roi_align_inputs_cpu_ptr[roi_align_inputs_index++] = pbox[1];
                            roi_align_inputs_cpu_ptr[roi_align_inputs_index++] = pbox[2];
                            roi_align_inputs_cpu_ptr[roi_align_inputs_index++] = pbox[3];
                            roi_align_inputs_cpu_ptr[roi_align_inputs_index++] = pbox[7];
                            roi_align_inputs_cpu_ptr[roi_align_inputs_index++] = pbox[8];
                        }
                    }
                }
                
                int number_anchors = roi_align_inputs_index / 6;
                
                int feature_channel = feature_0->size(1);
                // output_channel == input_channel
                assert(feature_channel == roialigned_feature->size(1));
                TRT::Tensor roi_align_result(TRT::DataType::Float);
                TRT::Tensor proposals_result(TRT::DataType::Float);
                roi_align_result.resize(number_anchors, feature_channel, roialigned_feature->size(2), roialigned_feature->size(3)).to_gpu(false);
                proposals_result.resize(number_anchors, 4).to_gpu(false);
                roi_align_inputs.to_gpu(true);
                
                RoiAlignImpl_Float(stream_, 
                            (const int64_t)(number_anchors * feature_channel * roi_align_result.size(2) * roi_align_result.size(3)), 
                            (const float**)bottom_data_list_tensor.gpu(), 
                            (const float*)spatial_scales_tensor.gpu(), 
                            (const int64_t)feature_channel,
                            /* heights */ (const int64_t*)heights_tensor.gpu(),
                            /* widths */  (const int64_t*)widths_tensor.gpu(),
                            /* pooled_height */ (const int64_t)roi_align_result.size(2),
                            /* pooled_width */ (const int64_t)roi_align_result.size(3),
                            /* sampling_ratio */ (const int64_t)2,
                            /* bottom_rois */ (const float*)roi_align_inputs.gpu(),
                            6,
                            /* top_data */ (float*)roi_align_result.gpu(),
                            /* proposals */ (float*)proposals_result.gpu(),
                            true);

                // 进行另外一个onnx的推理
                int input_number_anchor = min(MAX_IMAGE_BBOX, number_anchors);

                proposals->resize_single_dim(0, input_number_anchor).to_gpu(false);
                proposals->copy_from_gpu(0, proposals_result.gpu(), input_number_anchor * 4);
                
                roialigned_feature->resize_single_dim(0, input_number_anchor).to_gpu(false);
                roialigned_feature->copy_from_gpu(0, roi_align_result.gpu(), input_number_anchor * roialigned_feature->size(1) * roialigned_feature->size(2) * roialigned_feature->size(3));
                sub_engine->forward(false);
                
                boxes->to_gpu(false);
                scores->to_gpu(false);
                
                
                number_anchors = scores->count();
                int number_classes = scores->size(1);

                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch) {
                    float* boxes_output       = (float*)boxes->gpu();
                    float* scores_output      = (float*)scores->gpu();
                    float* output_array_ptr   = output_array_device.gpu<float>(ibatch);
                    auto affine_matrix        = affin_matrix_device.gpu<float>(ibatch);
                    checkCudaRuntime(cudaMemsetAsync(output_array_ptr, 0, sizeof(float), substream));
                    decode_kernel_invoker(boxes_output, scores_output, number_anchors, number_classes, 
                                          confidence_threshold_, nms_threshold_, affine_matrix, output_array_ptr, MAX_IMAGE_BBOX, substream);
                }
                output_array_device.to_cpu(true);
                // checkCudaRuntime(cudaStreamSynchronize(stream_));
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch) {
                    float* parray = output_array_device.cpu<float>(ibatch);
                    int count     = min(MAX_IMAGE_BBOX, (int)*parray);
                    auto& job     = fetch_jobs[ibatch];
                    auto& image_based_boxes = job.output;
     
                    for(int i = 0; i < count; ++i) {
                        float* pbox  = parray + 1 + i * RPN_NUM_BOX_ELEMENT;
                        int keepflag = pbox[6];
                        if(keepflag == 1) {
                            image_based_boxes.emplace_back(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], pbox[5]);
                        }
                    }
                    job.pro->set_value(image_based_boxes);
                }

                fetch_jobs.clear();
            }
            INFO("Engine destroy.");
        }

        virtual bool preprocess(Job& job, const Mat& image) override{
            
            job.mono_tensor = tensor_allocator_->query();
            if(job.mono_tensor == nullptr){
                INFOE("Tensor allocator query failed.");
                return false;
            }

            CUDATools::AutoDevice auto_device(gpu_);
            auto& tensor = job.mono_tensor->data();
            if(tensor == nullptr){
                // not init
                tensor = make_shared<TRT::Tensor>();
                tensor->set_workspace(make_shared<TRT::MixMemory>());
            }

            Size input_size(input_width_, input_height_);
            job.additional.compute(image.size(), input_size);
            
            tensor->set_stream(stream_);
            tensor->resize(1, 3, input_height_, input_width_);

            size_t size_image      = image.cols * image.rows * 3;
            size_t size_matrix     = iLogger::upbound(sizeof(job.additional.d2i), 32);
            auto workspace         = tensor->get_workspace();
            uint8_t* gpu_workspace        = (uint8_t*)workspace->gpu(size_matrix + size_image);
            float*   affine_matrix_device = (float*)gpu_workspace;
            uint8_t* image_device         = size_matrix + gpu_workspace;

            uint8_t* cpu_workspace        = (uint8_t*)workspace->cpu(size_matrix + size_image);
            float* affine_matrix_host     = (float*)cpu_workspace;
            uint8_t* image_host           = size_matrix + cpu_workspace;

            //checkCudaRuntime(cudaMemcpyAsync(image_host,   image.data, size_image, cudaMemcpyHostToHost,   stream_));
            // speed up
            memcpy(image_host, image.data, size_image);
            memcpy(affine_matrix_host, job.additional.d2i, sizeof(job.additional.d2i));
            checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_));
            checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(job.additional.d2i), cudaMemcpyHostToDevice, stream_));
            
            CUDAKernel::warp_affine_bilinear_and_normalize_plane(
                image_device,         image.cols * 3,       image.cols,       image.rows, 
                tensor->gpu<float>(), input_width_,         input_height_, 
                affine_matrix_device, 114, 
                normalize_, stream_
            );

            return true;
        }

        virtual vector<shared_future<BoxArray>> commits(const vector<Mat>& images) override{
            return ControllerImpl::commits(images);
        }

        virtual std::shared_future<BoxArray> commit(const Mat& image) override{
            return ControllerImpl::commit(image);
        }

    private:
        int input_width_            = 0;
        int input_height_           = 0;
        int gpu_                    = 0;
        float confidence_threshold_ = 0;
        float nms_threshold_        = 0;
        TRT::CUStream stream_       = nullptr;
        CUDAKernel::Norm normalize_;
    };

    shared_ptr<Infer> create_infer(const vector<string>& engine_files, Type type, int gpuid, float confidence_threshold, float nms_threshold){
        shared_ptr<InferImpl> instance(new InferImpl());
        if(!instance->startup(engine_files, type, gpuid, confidence_threshold, nms_threshold)){
            instance.reset();
        }
        return instance;
    }

    void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, Type type, int ibatch){
        
        CUDAKernel::Norm normalize;
        if(type == Type::FasterRCNN) {
            float mean[] = {0.485, 0.456, 0.406};
            float std[]  = {0.229, 0.224, 0.225};
            normalize = CUDAKernel::Norm::mean_std(mean, std, 1/255.0f, CUDAKernel::ChannelType::Invert);
        } else {
            INFOE("Unsupport type %d", type);
        }
        
        Size input_size(tensor->size(3), tensor->size(2));
        AffineMatrix affine;
        affine.compute(image.size(), input_size);

        size_t size_image      = image.cols * image.rows * 3;
        size_t size_matrix     = iLogger::upbound(sizeof(affine.d2i), 32);
        auto workspace         = tensor->get_workspace();
        uint8_t* gpu_workspace        = (uint8_t*)workspace->gpu(size_matrix + size_image);
        float*   affine_matrix_device = (float*)gpu_workspace;
        uint8_t* image_device         = size_matrix + gpu_workspace;

        uint8_t* cpu_workspace        = (uint8_t*)workspace->cpu(size_matrix + size_image);
        float* affine_matrix_host     = (float*)cpu_workspace;
        uint8_t* image_host           = size_matrix + cpu_workspace;
        auto stream                   = tensor->get_stream();

        memcpy(image_host, image.data, size_image);
        memcpy(affine_matrix_host, affine.d2i, sizeof(affine.d2i));
        checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream));
        checkCudaRuntime(cudaMemcpyAsync(affine_matrix_device, affine_matrix_host, sizeof(affine.d2i), cudaMemcpyHostToDevice, stream));

        CUDAKernel::warp_affine_bilinear_and_normalize_plane(
            image_device,               image.cols * 3,       image.cols,       image.rows, 
            tensor->gpu<float>(ibatch), input_size.width,     input_size.height, 
            affine_matrix_device, 114, 
            normalize, stream
        );
        tensor->synchronize();
    }
};