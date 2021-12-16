#include "shufflenetv2.hpp"
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

namespace ShuffleNetV2 {
    using namespace cv;
    using namespace std;

    using ControllerImpl = InferController
    <
        Mat,                    // input
        vector<float>,         // output
        tuple<string, int>     // start param
    >;
    class InferImpl : public Infer, public ControllerImpl{
        
        private:
            int input_width_            = 0;
            int input_height_           = 0;
            int gpu_                    = 0;
            float confidence_threshold_ = 0;
            float nms_threshold_        = 0;
            TRT::CUStream stream_       = nullptr;
            CUDAKernel::Norm normalize_;
        public:
            /** 要求在InferImpl里面执行stop，而不是在基类执行stop **/
            virtual ~InferImpl(){
                stop();
            }

            virtual bool startup(const string& file, int gpuid = 0){

                float mean[] = {0.485, 0.456, 0.406};
                float std[]  = {0.229, 0.224, 0.225};
                normalize_ = CUDAKernel::Norm::mean_std(mean, std, 1/255.0f, CUDAKernel::ChannelType::Invert);
                return ControllerImpl::startup(make_tuple(file, gpuid));
            }

            virtual void worker(promise<bool>& result) override {

                string file = get<0>(start_param_);
                int gpuid   = get<1>(start_param_);

                TRT::set_device(gpuid);
                auto engine = TRT::load_infer(file);
                if(engine == nullptr){
                    INFOE("Engine %s load failed", file.c_str());
                    result.set_value(false);
                    return;
                }

                engine->print();

                TRT::Tensor output_array_device(TRT::DataType::Float);

                int max_batch_size = engine->get_max_batch_size();
                auto input         = engine->tensor("input");
                auto output        = engine->tensor("output");
                int num_classes    = output->size(1);

                input_width_       = input->size(3);
                input_height_      = input->size(2);
                tensor_allocator_  = make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
                stream_            = engine->get_stream();
                gpu_               = gpuid;
                result.set_value(true);

                input->resize_single_dim(0, max_batch_size).to_gpu();
                // batch_size * num_classes
                output_array_device.resize(max_batch_size, num_classes).to_gpu();

                vector<Job> fetch_jobs;
                while(get_jobs_and_wait(fetch_jobs, max_batch_size)){

                    int infer_batch_size = fetch_jobs.size();
                    input->resize_single_dim(0, infer_batch_size);

                    for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                        auto& job  = fetch_jobs[ibatch];
                        auto& mono = job.mono_tensor->data();
                        input->copy_from_gpu(input->offset(ibatch), mono->gpu(), mono->count());
                        job.mono_tensor->release();
                    }

                    engine->forward(false);
                    output_array_device.to_gpu(false);
                    for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                        // auto& job                 = fetch_jobs[ibatch];
                        float* image_based_output = output->gpu<float>(ibatch);
                        float* output_array_ptr   = output_array_device.gpu<float>(ibatch);
                        checkCudaRuntime(cudaMemcpyAsync(output_array_ptr, image_based_output, num_classes * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
                    }
   

                    output_array_device.to_cpu();
                    for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                        float* parray = output_array_device.cpu<float>(ibatch);
                        auto& job     = fetch_jobs[ibatch];
                        auto& job_output = job.output;
                        // job_output.resize(num_classes);
                        for(int i = 0; i < num_classes; ++i){
                            job_output.push_back(parray[i]);
                        }
                        job.pro->set_value(job_output);
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
                
                tensor->set_stream(stream_);
                tensor->resize(1, 3, input_height_, input_width_);

                size_t size_image      = image.cols * image.rows * 3;
                auto workspace         = tensor->get_workspace();
                uint8_t* gpu_workspace        = (uint8_t*)workspace->gpu(size_image);
                uint8_t* image_device         = gpu_workspace;

                uint8_t* cpu_workspace        = (uint8_t*)workspace->cpu(size_image);
                uint8_t* image_host           = cpu_workspace;
                
                // speed up
                memcpy(image_host, image.data, size_image);
                checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream_));

                CUDAKernel::resize_bilinear_and_normalize(
                    image_device,               image.cols * 3,       image.cols,       image.rows, 
                    tensor->gpu<float>(), input_width_,     input_height_, 
                    114, normalize_, stream_
                );
                
                return true;
            }

            virtual vector<shared_future<vector<float>>> commits(const vector<Mat>& images) override{
                return ControllerImpl::commits(images);
            }
            
            virtual shared_future<vector<float>> commit(const Mat& image) override{
                return ControllerImpl::commit(image);
            }

    };

    shared_ptr<Infer> create_infer(const string& engine_file, int gpu_id){
        shared_ptr<InferImpl> instance(new InferImpl());
        if(!instance->startup(engine_file, gpu_id)) {
            instance.reset();
        }
        return instance;
    }

    void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, int ibatch){

        CUDAKernel::Norm normalize;

        float mean[] = {0.485, 0.456, 0.406};
        float std[]  = {0.229, 0.224, 0.225};
        normalize = CUDAKernel::Norm::mean_std(mean, std, 1/255.0f, CUDAKernel::ChannelType::Invert);
        
        Size input_size(tensor->size(3), tensor->size(2));

        size_t size_image      = image.cols * image.rows * 3;
        auto workspace         = tensor->get_workspace();
        uint8_t* gpu_workspace        = (uint8_t*)workspace->gpu(size_image);
        uint8_t* image_device         = gpu_workspace;

        uint8_t* cpu_workspace        = (uint8_t*)workspace->cpu(size_image);
        uint8_t* image_host           = cpu_workspace;
        auto stream                   = tensor->get_stream();

        memcpy(image_host, image.data, size_image);
        checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream));

        CUDAKernel::resize_bilinear_and_normalize(
            image_device,               image.cols * 3,       image.cols,       image.rows, 
            tensor->gpu<float>(ibatch), input_size.width,     input_size.height, 
            114, normalize, stream
        );
        tensor->synchronize();
    }
};