#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void forward_kernel(
    const int nthreads,
    const int num_pairs,
    const int num_channels,
    const int num_layer_channels,
    const int bottom_h,
    const int bottom_w,
    const int channel_offset,
    const Dtype *pairs_data,
    const Dtype *layer_data,
    Dtype *left_data,
    Dtype *right_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // n: sample, p: pair, c: channel
    int n = index / num_pairs / num_layer_channels;
    int p = (index / num_layer_channels) % num_pairs;
    int c = index % num_layer_channels;

    int pairs_offset = (n * num_pairs + p) * 4;
    int y1 = static_cast<int>(pairs_data[pairs_offset + 0]);
    int x1 = static_cast<int>(pairs_data[pairs_offset + 1]);
    int y2 = static_cast<int>(pairs_data[pairs_offset + 2]);
    int x2 = static_cast<int>(pairs_data[pairs_offset + 3]);

    int layer_offset = (n * num_pairs + p) * num_channels;
    int base_offset = (n * num_layer_channels + c) * bottom_h * bottom_w;

    left_data[layer_offset + channel_offset + c] =
        layer_data[base_offset + y1 * bottom_w + x1];
    right_data[layer_offset + channel_offset + c] =
        layer_data[base_offset + y2 * bottom_w + x2];
  }
}

template <typename Dtype>
void HypercolumnPairsLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype *pairs_data = bottom[0]->gpu_data();
  Dtype *left_data = top[0]->mutable_gpu_data();
  Dtype *right_data = top[1]->mutable_gpu_data();

  for (int l = 0; l < num_layers_; ++l) {
    int num_layer_channels = bottom[1 + l]->shape(1);
    int total_count = bottom[0]->shape(0) * num_pairs_ * num_layer_channels;
    const Dtype* layer_data = bottom[1 + l]->gpu_data();

    forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(total_count), CAFFE_CUDA_NUM_THREADS>>>(
        total_count,
        num_pairs_,
        num_channels_,
        num_layer_channels,
        bottom_h_,
        bottom_w_,
        channel_offsets_[l],
        pairs_data,
        layer_data,
        left_data,
        right_data);
  }
}

__global__ void backward_kernel(
    const int nthreads,
    const int num_pairs,
    const int num_channels,
    const int num_layer_channels,
    const int bottom_h,
    const int bottom_w,
    const int channel_offset,
    const float *pairs_data,
    const float *left_diff,
    const float *right_diff,
    float *layer_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // n: sample, p: pair, c: channel
    int n = index / num_pairs / num_layer_channels;
    int p = (index / num_layer_channels) % num_pairs;
    int c = index % num_layer_channels;

    int pairs_offset = (n * num_pairs + p) * 4;
    int y1 = static_cast<int>(pairs_data[pairs_offset + 0]);
    int x1 = static_cast<int>(pairs_data[pairs_offset + 1]);
    int y2 = static_cast<int>(pairs_data[pairs_offset + 2]);
    int x2 = static_cast<int>(pairs_data[pairs_offset + 3]);

    int layer_offset = (n * num_pairs + p) * num_channels;
    int base_offset = (n * num_layer_channels + c) * bottom_h * bottom_w;

    // Multiple threads could increment the same values, so it has to be atomic
#define ATOMIC
#ifdef ATOMIC
    atomicAdd(&layer_diff[base_offset + y1 * bottom_w + x1],
              left_diff[layer_offset + channel_offset + c]);
    atomicAdd(&layer_diff[base_offset + y2 * bottom_w + x2],
              right_diff[layer_offset + channel_offset + c]);
#else
    layer_diff[base_offset + y1 * bottom_w + x1] = left_diff[layer_offset + channel_offset + c];
    layer_diff[base_offset + y2 * bottom_w + x2] = right_diff[layer_offset + channel_offset + c];
#endif
  }
}

// Note: The general templated version will fall-back to the CPU version.
// This will happen only for 64-bit floats, since the 32-bit version is defined below
template <typename Dtype>
void HypercolumnPairsLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Backward_cpu(top, propagate_down, bottom);
}

template <>
void HypercolumnPairsLayer<float>::Backward_gpu(const vector<Blob<float>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<float>*>& bottom) {
  if (propagate_down[0]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to pairs inputs.";
  }

  // Reset gradients
  for (int l = 0; l < num_layers_; ++l) {
    float *layer_diff = bottom[1 + l]->mutable_gpu_diff();
    caffe_gpu_set(bottom[1 + l]->count(), 0.0f, layer_diff);
  }

  const float *pairs_data = bottom[0]->gpu_data();
  const float *left_diff = top[0]->gpu_diff();
  const float *right_diff = top[1]->gpu_diff();

  for (int l = 0; l < num_layers_; ++l) {
    if (propagate_down[1 + l]) {
      int num_layer_channels = bottom[1 + l]->shape(1);
      int total_count = bottom[0]->shape(0) * num_pairs_ * num_layer_channels;
      float* layer_diff = bottom[1 + l]->mutable_gpu_diff();

      backward_kernel<<<CAFFE_GET_BLOCKS(total_count), CAFFE_CUDA_NUM_THREADS>>>(
          total_count,
          num_pairs_,
          num_channels_,
          num_layer_channels,
          bottom_h_,
          bottom_w_,
          channel_offsets_[l],
          pairs_data,
          left_diff,
          right_diff,
          layer_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(HypercolumnPairsLayer);

}  // namespace caffe

