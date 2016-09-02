#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void strided_memcpy(Dtype *dst, const Dtype *src, int dst_stride, int src_stride, int count) {
  for (int i = 0; i < count; ++i) {
    dst[i * dst_stride] = src[i * src_stride];
  }
}

template <typename Dtype>
void strided_memadd(Dtype *dst, const Dtype *src, int dst_stride, int src_stride, int count) {
  for (int i = 0; i < count; ++i) {
    dst[i * dst_stride] += src[i * src_stride];
  }
}

template <typename Dtype>
void HypercolumnPairsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void HypercolumnPairsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    num_channels_ = 0;
    num_pairs_ = bottom[0]->shape(1);
    int num = bottom[0]->shape(0);

    // Centroids should be specified with four dimesions (y1, x1, y2, x2)
    CHECK_EQ(bottom[0]->shape(2), 4);

    num_layers_ = 0;
    for (int i = 1; i < bottom.size(); ++i) {
        if (i == 1) {
            bottom_h_ = bottom[i]->shape(2);
            bottom_w_ = bottom[i]->shape(3);
        } else {
            CHECK_EQ(bottom[i]->shape(2), bottom_h_);
            CHECK_EQ(bottom[i]->shape(3), bottom_w_);
        }

        CHECK_EQ(bottom[i]->shape(0), num);

        channel_offsets_.push_back(num_channels_);
        num_channels_ += bottom[i]->shape(1);
        num_layers_ += 1;
    }

    vector<int> top_shape;
    top_shape.push_back(num * num_pairs_);
    top_shape.push_back(num_channels_);

    top[0]->Reshape(top_shape);
    top[1]->Reshape(top_shape);
}

template <typename Dtype>
void HypercolumnPairsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype *pairs_data = bottom[0]->cpu_data();
  Dtype *left_data = top[0]->mutable_cpu_data();
  Dtype *right_data = top[1]->mutable_cpu_data();

  for (int n = 0; n < bottom[0]->shape(0); ++n) {
    for (int p = 0; p < num_pairs_; ++p) {
      int off = n * num_pairs_ * 4 + p * 4;
      int y1 = static_cast<int>(pairs_data[off + 0]);
      int x1 = static_cast<int>(pairs_data[off + 1]);
      int y2 = static_cast<int>(pairs_data[off + 2]);
      int x2 = static_cast<int>(pairs_data[off + 3]);

      int offset1 = n * num_pairs_ * num_channels_ + p * num_channels_;

      for (int l = 0; l < num_layers_; ++l) {
        const Dtype *layer_data = bottom[1 + l]->cpu_data();
        int num_layer_channels = bottom[1 + l]->shape(1);
        int cc = channel_offsets_[l];

        strided_memcpy(left_data + offset1 + cc,
                       layer_data + (n * num_layer_channels * bottom_h_ * bottom_w_ +
                                     y1 * bottom_w_ + x1),
                       1, bottom_h_ * bottom_w_, num_layer_channels);

        strided_memcpy(right_data + offset1 + cc,
                       layer_data + (n * num_layer_channels * bottom_h_ * bottom_w_ +
                                     y2 * bottom_w_ + x2),
                       1, bottom_h_ * bottom_w_, num_layer_channels);
      }
    }
  }
}

template <typename Dtype>
void HypercolumnPairsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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

      for (int l = 0; l < num_layers_; ++l) {
        int num_layer_channels = bottom[1 + l]->shape(1);
        if (propagate_down[1 + l]) {
          int cc = channel_offsets_[l];
          Dtype* layer_diff = bottom[1 + l]->mutable_cpu_diff();

          strided_memadd(layer_diff + (n * num_layer_channels * bottom_h_ * bottom_w_ +
                                       y1 * bottom_w_ + x1),
                         left_diff + offset1 + cc,
                         bottom_h_ * bottom_w_,
                         1,
                         num_layer_channels);

          strided_memadd(layer_diff + (n * num_layer_channels * bottom_h_ * bottom_w_ +
                                       y2 * bottom_w_ + x2),
                         right_diff + offset1 + cc,
                         bottom_h_ * bottom_w_,
                         1,
                         num_layer_channels);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(HypercolumnPairsLayer);
#endif

INSTANTIATE_CLASS(HypercolumnPairsLayer);
REGISTER_LAYER_CLASS(HypercolumnPairs);

}  // namespace caffe
