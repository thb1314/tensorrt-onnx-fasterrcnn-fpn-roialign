#ifndef __ROIALIGN__
#define __ROIALIGN__


#include <common/cuda_tools.hpp>

namespace FasterRCNN{

// We would like to use 64-bit integer to support large matrices. However, CUDA seems to support only 32-bit integer
// For now, use int32_t to ensure that both Linux and Windows see this as 32 bit integer type.
#ifndef CUDA_LONG
#define CUDA_LONG int32_t
#endif


/**
 * bottom_data 有 height x width 的形状
 * 求取 (row y, col x)坐标的插值
 * 
 * */
template <typename T>
__device__ T bilinear_interpolate(
    const T* bottom_data,
    const int height,
    const int width,
    T y,
    T x,
    const bool is_mode_avg,
    const int index /* index for debug only*/) 
    {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }

  if (y <= 0) {
    y = 0;
  }
  if (x <= 0) {
    x = 0;
  }

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = bottom_data[y_low * width + x_low];
  T v2 = bottom_data[y_low * width + x_high];
  T v3 = bottom_data[y_high * width + x_low];
  T v4 = bottom_data[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = is_mode_avg
            ? (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4)  // mode Avg
            : max(max(max(w1 * v1, w2 * v2), w3 * v3), w4 * v4);  // mode Max

  return val;
}

struct GridDim {
  enum : CUDA_LONG {
    maxThreadsPerBlock = 256,  // max threads per block
    maxElementsPerThread = 4,  // max element processed per thread
  };
};


template <typename T>
__device__ __inline__ T _Ceil(T a);

template <>
__device__ __inline__ float _Ceil(float a) { return ceilf(a); }

template <>
__device__ __inline__ double _Ceil(double a) { return ceil(a); }


/**
 * nthreads: 线程数
 * bottom_data: BxCxHxW的数据
 * bottom_data_list
 * spatial_scale: 乘法性质空间标尺因子，池化时，将RoI坐标变换至运算采用的标度，默认值为1.0。由于proposal是对应MXN尺度的，
 * spatial_scales
 * 所以首先使用 spatial_scale 参数将其映射回 (M/spatial_scale) X (N/spatial_scale) 大小的feature map尺度
 * sampling_ratio: 插值格中采样点的数目。 如果它 <=0, 它们将自适应 roi_width 和 pooled_w , 在高度上也是同样的道理。默认值为-1
 * bottom_rois: Nx(roi_cols)的坐标, N通过 blockIdx 计算得来
 * batch_indices_ptr: N维向量
 * top_data: write_data
 * 
 * */
template <typename T>
__global__ void RoIAlignForward(
    const int64_t nthreads,
    const T** bottom_data_list,
    const T* spatial_scales,
    const int64_t channels,
    const int64_t* heights,
    const int64_t* widths,
    const int64_t pooled_height,
    const int64_t pooled_width,
    const int64_t sampling_ratio,
    const T* bottom_rois,
    int64_t roi_cols,
    T* top_data,
    T* proposals,
    const bool is_mode_avg
    ) {
  // 起始 blockIdx.x * blockDim.x + threadIdx.x 步长 blockDim.x * gridDim.x
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;


    // RoI could have 4 or 5 columns
    // last column index == level_index
    const T* offset_bottom_rois = bottom_rois + n * roi_cols;
    
    
    bool continuous_coordinate = false;
    // Do not using rounding; this implementation detail is critical
    T roi_offset = continuous_coordinate ? T(0.5) : T(0);
    const int level_index = int(offset_bottom_rois[roi_cols - 2]);
    
    const T* bottom_data = bottom_data_list[level_index];
    const T spatial_scale = spatial_scales[level_index];
    T roi_start_w = offset_bottom_rois[0] * spatial_scale - roi_offset;
    T roi_start_h = offset_bottom_rois[1] * spatial_scale - roi_offset;
    T roi_end_w = offset_bottom_rois[2] * spatial_scale - roi_offset;
    T roi_end_h = offset_bottom_rois[3] * spatial_scale - roi_offset;
    proposals[n * 4 + 0] = offset_bottom_rois[0];
    proposals[n * 4 + 1] = offset_bottom_rois[1];
    proposals[n * 4 + 2] = offset_bottom_rois[2];
    proposals[n * 4 + 3] = offset_bottom_rois[3];

    const int roi_batch_ind = int(offset_bottom_rois[roi_cols - 1]);

    T roi_width = roi_end_w - roi_start_w;
    T roi_height = roi_end_h - roi_start_h;
    if (!continuous_coordinate) { // backward compatiblity
      // Force malformed ROIs to be 1x1
      roi_width = max(roi_width, (T)1.);
      roi_height = max(roi_height, (T)1.);
    }
    // roi的高度除以pooled_height的高度（都是特征维度上的单位）
    // 计算每一块bin的高度与宽度 用当前特征的维度单位表示 计算每一个格子的高度和宽度
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    const int64_t width = widths[level_index];
    const int64_t height = heights[level_index];
    
    // bottom_data BxCxHxW 维度
    const T* offset_bottom_data =
        bottom_data + static_cast<int64_t>((roi_batch_ind * channels + c) * height * width);

    // roi_bin_grid_x 表示 每一个格子在x方向上采集多少个采样点
    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : _Ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : _Ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4
    // 针对 n,c,ph,pw 的点进行插值
    T output_val = 0.;
    bool max_flag = false;
    // roi_start_h + ph * bin_size_h 表示ph格子位置的索引起始值
    // roi_start_h + ph * bin_size_h + (roi_bin_grid_h - 0.5) / roi_bin_grid_h * bin_size_h 
    // 表示ph格子位置的索引终止值
    // 也就是 针对n,c,ph,pw 的点的值映射回原图划分为roi_bin_grid_h x roi_bin_grid_w个小块
    // 计算每一个小块的值

    // bin_size_h 每一个网格占用的特征上的像素个数(在row这个维度上
    
    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const T y = roi_start_h + ph * bin_size_h +
          static_cast<T>(iy + .5f) * bin_size_h /
              static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + pw * bin_size_w +
            static_cast<T>(ix + .5f) * bin_size_w /
                static_cast<T>(roi_bin_grid_w);

        T val = bilinear_interpolate(
            offset_bottom_data, height, width, y, x, is_mode_avg, index);
        
        if (is_mode_avg) {
          output_val += val;
        } else {
          if (!max_flag) {
            output_val = val;
            max_flag = true;
          } else {
            output_val = max(output_val, val);
          }
        }
      }
    }
    if (is_mode_avg) {
      output_val /= count;
    }

    top_data[index] = output_val;
  }
}

/**
 * nthreads: 线程数
 * bottom_data: BxCxHxW的数据
 * bottom_data_list 对应 {P_k, ...}
 * spatial_scale: 乘法性质空间标尺因子，池化时，将RoI坐标变换至运算采用的标度，默认值为1.0。由于proposal是对应MXN尺度的，
 * spatial_scales: 将roi中的坐标转换为对应特征图上的坐标
 * 所以首先使用 spatial_scale 参数将其映射回 (M/spatial_scale) X (N/spatial_scale) 大小的feature map尺度
 * sampling_ratio: 插值格中采样点的数目。 如果它 <=0, 它们将自适应 roi_width 和 pooled_w , 在高度上也是同样的道理。默认值为-1
 * bottom_rois: Nx(roi_cols)的坐标, N通过 blockIdx 计算得来
 * batch_indices_ptr: N维向量
 * top_data: write_data
 * 
 * */
template <typename T>
void RoiAlignImpl(
  cudaStream_t stream,
  const int64_t nthreads,
  const T** bottom_data_list,
  const T* spatial_scales,
  const int64_t channels,
  const int64_t* heights,
  const int64_t* widths,
  const int64_t pooled_height,
  const int64_t pooled_width,
  const int64_t sampling_ratio,
  const T* bottom_rois,
  int64_t roi_cols,
  T* top_data,
  T* proposals,
  const bool is_mode_avg) {
    int blocksPerGrid = (int)(ceil(static_cast<float>(nthreads) / GridDim::maxThreadsPerBlock)); 
    RoIAlignForward<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      nthreads,
      bottom_data_list,
      spatial_scales,
      channels,
      heights,
      widths,
      pooled_height,
      pooled_width,
      sampling_ratio,
      bottom_rois,
      roi_cols,
      top_data,
      proposals,
      is_mode_avg
    );    
}

#define SPECIALIZED_IMPL(T)                     \
  template void RoiAlignImpl<T>(                \
        cudaStream_t stream,              \
        const int64_t nthreads,                 \
        const T** bottom_data_list,                   \
        const T* spatial_scales,                  \
        const int64_t channels,                 \
        const int64_t* heights,                   \
        const int64_t* widths,                    \
        const int64_t pooled_height,            \
        const int64_t pooled_width,             \
        const int64_t sampling_ratio,           \
        const T* bottom_rois,                   \
        int64_t roi_cols,                       \
        T* top_data,                            \
        T* proposals,                            \
        const bool is_mode_avg                 \
        );


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
  const bool is_mode_avg) {

    int blocksPerGrid = (int)(ceil(static_cast<float>(nthreads) / GridDim::maxThreadsPerBlock)); 
    RoIAlignForward<float><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      nthreads,
      bottom_data_list,
      spatial_scales,
      channels,
      heights,
      widths,
      pooled_height,
      pooled_width,
      sampling_ratio,
      bottom_rois,
      roi_cols,
      top_data,
      proposals,
      is_mode_avg
    );    
}

SPECIALIZED_IMPL(double)

}
#endif
