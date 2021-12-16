
#include <common/cuda_tools.hpp>


namespace FasterRCNN{

    const int NUM_BOX_ELEMENT = 9;      // left, top, right, bottom, confidence, label, keepflag, not use, not use
    const int RPN_NUM_BOX_ELEMENT = 9;      // left, top, right, bottom, confidence, level, keepflag, fpn_level, batch_index
    static __device__ void affine_project(float* matrix, float x, float y, float* ox, float* oy){
        *ox = matrix[0] * x + matrix[1] * y + matrix[2];
        *oy = matrix[3] * x + matrix[4] * y + matrix[5];
    }

    static __global__ void decode_kernel(float* predict, float* score, int num_bboxes, int num_classes, float confidence_threshold, float* invert_affine_matrix, float* parray, int max_objects){  

        int position = blockDim.x * blockIdx.x + threadIdx.x;
		if (position >= num_bboxes) return;
        
        float* pitem     = predict + 4 * position;
        float confidence = score[position];

        if(confidence < confidence_threshold)
            return;
        
        
        float left       = *pitem++;
        float top        = *pitem++;
        float right      = *pitem++;
        float bottom     = *pitem;
        float width = right - left;
        float height = bottom - top;
        if(width < 1e-3 || height < 1e-3)
            return;
        
        int index = atomicAdd(parray, 1);
        if(index >= max_objects)
            return;

        int label = position % num_classes;

        affine_project(invert_affine_matrix, left,  top,    &left,  &top);
        affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

        float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
        *pout_item++ = left;
        *pout_item++ = top;
        *pout_item++ = right;
        *pout_item++ = bottom;
        *pout_item++ = confidence;
        *pout_item++ = label;
        *pout_item++ = 1; // 1 = keep, 0 = ignore
        
    }

    static __device__ float box_iou(
        float aleft, float atop, float aright, float abottom, 
        float bleft, float btop, float bright, float bbottom
    ){

        float cleft 	= max(aleft, bleft);
        float ctop 		= max(atop, btop);
        float cright 	= min(aright, bright);
        float cbottom 	= min(abottom, bbottom);
        
        float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
        if(c_area == 0.0f)
            return 0.0f;
        
        float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
        float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
        return c_area / (a_area + b_area - c_area);
    }

    static __global__ void nms_kernel(float* bboxes, int max_objects, float threshold){

        int position = (blockDim.x * blockIdx.x + threadIdx.x);
        int count = min((int)*bboxes, max_objects);
        if (position >= count)
            return;
        
        // left, top, right, bottom, confidence
        float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
        
        for(int i = 0; i < count; ++i){
            float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
            if(i == position || pcurrent[5] != pitem[5]) continue;
            
            if(pitem[4] >= pcurrent[4]) {
                if(pitem[4] == pcurrent[4] && i < position)
                    continue;

                float iou = box_iou(
                    pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                    pitem[0],    pitem[1],    pitem[2],    pitem[3]
                );

                if(iou > threshold) {
                    pcurrent[6] = 0;  // 1=keep, 0=ignore
                    return;
                }
            }
        }
    }

    void decode_kernel_invoker(float* predict, float* scores, int num_bboxes, int num_classes, float confidence_threshold, 
        float nms_threshold, float* invert_affine_matrix, float* parray, int max_objects, cudaStream_t stream){
        

        auto grid = CUDATools::grid_dims(num_bboxes);
        auto block = CUDATools::block_dims(num_bboxes);
        checkCudaKernel(decode_kernel<<<grid, block, 0, stream>>>(predict, scores, num_bboxes, num_classes, confidence_threshold, invert_affine_matrix, parray, max_objects));

        grid = CUDATools::grid_dims(max_objects);
        block = CUDATools::block_dims(max_objects);
        checkCudaKernel(nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold));
    }

    static __global__ void rpn_nms_kernel(float* bboxes, int num_bboxes, int max_objects, float threshold) {

        int position = (blockDim.x * blockIdx.x + threadIdx.x);
        int count = min(num_bboxes, max_objects);
        if (position >= count) 
            return;
        

        // left, top, right, bottom, confidence, keep
        float* pcurrent = bboxes + 1 + position * RPN_NUM_BOX_ELEMENT;
        
        for(int i = 0; i < count; ++i) {
            float* pitem = bboxes + 1 + i * RPN_NUM_BOX_ELEMENT;
            if(i == position || pcurrent[5] != pitem[5]) continue;
            
            if(pitem[4] >= pcurrent[4]) {
                if(pitem[4] == pcurrent[4] && i < position)
                    continue;

                float iou = box_iou(
                    pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                    pitem[0],    pitem[1],    pitem[2],    pitem[3]
                );

                if(iou > threshold) {
                    pcurrent[6] = 0;  // 1=keep, 0=ignore
                    return;
                }
            }
        }
    }

    static __global__ void rpn_decode_kernel(float* predict, int num_bboxes, float* invert_affine_matrix, float* parray, int max_objects, int batch_index){  

        int position = blockDim.x * blockIdx.x + threadIdx.x;
		if (position >= num_bboxes) return;

        float* pitem     = predict + (6) * position;
        

        float objectness = *pitem++;
        float left       = *pitem++;
        float top        = *pitem++;
        float right      = *pitem++;
        float bottom     = *pitem++;
        float level     = *pitem;
        // filter small box w >= 1e-3 && h >= 1e-3
        
        float width = right - left;
        float height = bottom - top;
        if(width < 1e-3 || height < 1e-3)
            return;
        int index = atomicAdd(parray, 1);
        if(index >= max_objects)
            return;
        // affine_project(invert_affine_matrix, left,  top,    &left,  &top);
        // affine_project(invert_affine_matrix, right, bottom, &right, &bottom);
        
        float* pout_item = parray + 1 + index * RPN_NUM_BOX_ELEMENT;
        *pout_item++ = left;
        *pout_item++ = top;
        *pout_item++ = right;
        *pout_item++ = bottom;
        *pout_item++ = objectness;
        *pout_item++ = level;
        *pout_item++ = 1; // 1 = keep, 0 = ignore

        float area = width * height;
        // 硬编码
        int fpn_lvl = floorf(4 + log2(sqrt(area) / 224) + 1e-6) - 2;

        fpn_lvl = fpn_lvl > 3 ? 3 : fpn_lvl;
        fpn_lvl = fpn_lvl < 0 ? 0 : fpn_lvl;
        *pout_item++ = fpn_lvl;
        *pout_item = batch_index;
    }

    void rpn_decode_kernel_invoker(float* predict, int num_bboxes, float nms_threshold, float* invert_affine_matrix, 
                                float* parray, 
                                int max_objects,
                                int batch_index,
                                cudaStream_t stream) {
        
        auto grid = CUDATools::grid_dims(num_bboxes);
        auto block = CUDATools::block_dims(num_bboxes);
        checkCudaKernel(rpn_decode_kernel<<<grid, block, 0, stream>>>(predict, num_bboxes, invert_affine_matrix, parray, max_objects, batch_index));

        grid = CUDATools::grid_dims(max_objects);
        block = CUDATools::block_dims(max_objects);
        checkCudaKernel(rpn_nms_kernel<<<grid, block, 0, stream>>>(parray, num_bboxes, max_objects, nms_threshold));
    }
};