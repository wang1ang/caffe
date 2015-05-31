#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

void strided_gpu_memcpy(void *dst, const void *src, int dst_stride, int src_stride, int count, int size) {
  cudaMemcpy2D(
    dst,
    size * dst_stride,
    src,
    size * src_stride,
    size,
    count,
    cudaMemcpyDefault);
}

template <typename Dtype>
void strided_gpu_memadd(Dtype *dst, const Dtype *src, int dst_stride, int src_stride, int count) {
  for (int i = 0; i < count; ++i) {
    dst[i * dst_stride] += src[i * src_stride];
  }
}


template <typename Dtype>
void HypercolumnPairsLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype *pairs_data = bottom[0]->cpu_data();
  void *left_data = top[0]->mutable_gpu_data();
  void *right_data = top[1]->mutable_gpu_data();

  for (int n = 0; n < bottom[0]->shape(0); ++n) {
    for (int p = 0; p < num_pairs_; ++p) {
      int off = n * num_pairs_ * 4 + p * 4;
      int y1 = static_cast<int>(pairs_data[off + 0]);
      int x1 = static_cast<int>(pairs_data[off + 1]);
      int y2 = static_cast<int>(pairs_data[off + 2]);
      int x2 = static_cast<int>(pairs_data[off + 3]);

      int offset1 = n * num_pairs_ * num_channels_ + p * num_channels_;

      int cc = 0;
      for (int l = 0; l < num_layers_; ++l) {
        const void *layer_data = bottom[1 + l]->gpu_data();
        int num_layer_channels = bottom[1 + l]->shape(1);

        strided_gpu_memcpy(left_data + sizeof(Dtype) * (offset1 + cc),
                       layer_data + sizeof(Dtype) * (n * num_layer_channels * bottom_h_ * bottom_w_ +
                                         y1 * bottom_w_ + x1),
                       1, bottom_h_ * bottom_w_, num_layer_channels, sizeof(Dtype));

        strided_gpu_memcpy(right_data + sizeof(Dtype) * (offset1 + cc),
                       layer_data + sizeof(Dtype) * (n * num_layer_channels * bottom_h_ * bottom_w_ +
                                         y2 * bottom_w_ + x2),
                       1, bottom_h_ * bottom_w_, num_layer_channels, sizeof(Dtype));

        cc += num_layer_channels;
      }
    }
  }
}

/*
template <typename Dtype>
__global__ void back_helper(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}
*/

template <typename Dtype>
void HypercolumnPairsLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to pairs inputs.";
  }
  // Not sure if I need to reset these values
  for (int l = 0; l < num_layers_; ++l) {
    Dtype *layer_diff = bottom[1 + l]->mutable_cpu_diff();
    caffe_set(bottom[1 + l]->count(), Dtype(0), layer_diff);
  }

  //back_helper<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      //N, alpha, Y);

  const Dtype *pairs_data = bottom[0]->cpu_data();
  const Dtype *left_diff = top[0]->mutable_cpu_diff();
  const Dtype *right_diff = top[1]->mutable_cpu_diff();

  for (int n = 0; n < bottom[0]->shape(0); ++n) {
    for (int p = 0; p < num_pairs_; ++p) {
      int off = n * num_pairs_ * 4 + p * 4;
      int y1 = static_cast<int>(pairs_data[off + 0]);
      int x1 = static_cast<int>(pairs_data[off + 1]);
      int y2 = static_cast<int>(pairs_data[off + 2]);
      int x2 = static_cast<int>(pairs_data[off + 3]);

      int offset1 = n * num_pairs_ * num_channels_ + p * num_channels_;

      int cc = 0;
      for (int l = 0; l < num_layers_; ++l) {
        int num_layer_channels = bottom[1 + l]->shape(1);
        if (propagate_down[1 + l]) {
          Dtype* layer_diff = bottom[1 + l]->mutable_cpu_diff();

          strided_gpu_memadd(layer_diff + (n * num_layer_channels * bottom_h_ * bottom_w_ +
                                       y1 * bottom_w_ + x1),
                         left_diff + offset1 + cc,
                         bottom_h_ * bottom_w_,
                         1,
                         num_layer_channels);

          strided_gpu_memadd(layer_diff + (n * num_layer_channels * bottom_h_ * bottom_w_ +
                                       y2 * bottom_w_ + x2),
                         right_diff + offset1 + cc,
                         bottom_h_ * bottom_w_,
                         1,
                         num_layer_channels);
        }
        cc += num_layer_channels;
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(HypercolumnPairsLayer);

}  // namespace caffe

