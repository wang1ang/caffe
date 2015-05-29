#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void HypercolumnPairsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void HypercolumnPairsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    //LOG(INFO) << "Reshaping";
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
    
    // Run through the pairs
    //for (int i
    /*
    for (int i = 0; i < bottom.size(); ++i) {
      if (!propagate_down[i]) { continue; }
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
      const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
      for (int n = 0; n < num_concats_; ++n) {
        caffe_copy(bottom_concat_axis * concat_input_size_, top_diff +
            (n * top_concat_axis + offset_concat_axis) * concat_input_size_,
            bottom_diff + n * bottom_concat_axis * concat_input_size_);
      }
      offset_concat_axis += bottom_concat_axis;
    }
    */


  const Dtype *pairs_data = bottom[0]->cpu_data();
  Dtype *left_data = top[0]->mutable_cpu_data();
  Dtype *right_data = top[1]->mutable_cpu_data();

  for (int n = 0; n < bottom[0]->shape(0); ++n) {
    int offset = n * num_pairs_ * num_layers_ * num_channels_;
    for (int p = 0; p < num_pairs_; ++p) {
      int off = n * num_pairs_ * 4 + p * 4;
      int y1 = static_cast<int>(pairs_data[off + 0]);
      int x1 = static_cast<int>(pairs_data[off + 1]);
      int y2 = static_cast<int>(pairs_data[off + 2]);
      int x2 = static_cast<int>(pairs_data[off + 3]);

      int offset1 = offset + p * num_channels_;

      int cc = 0;
      for (int l = 0; l < num_layers_; ++l) {
        const Dtype *layer_data = bottom[l + 1]->cpu_data();
        int layer_channels = bottom[l + 1]->shape(1);
        for (int c = 0; c < layer_channels; ++c) {
          /*LOG(INFO) << "Setting " << l << " (" << n << ", " << c << ", " << y1 << ", " << x1 << ") to " << "(" << n << ", " << p << ") value " << layer_data[
            n * layer_channels * bottom_h_ * bottom_w_ +
            c * bottom_h_ * bottom_w_ + 
            y1 * bottom_w_ +
            x1];
          */
          left_data[offset1 + cc] = layer_data[
            n * layer_channels * bottom_h_ * bottom_w_ +
            c * bottom_h_ * bottom_w_ + 
            y1 * bottom_w_ +
            x1];

          right_data[offset1 + cc] = layer_data[
            n * layer_channels * bottom_h_ * bottom_w_ +
            c * bottom_h_ * bottom_w_ + 
            y2 * bottom_w_ +
            x2];
          cc += 1;
        }
      }
    }
  }
}

template <typename Dtype>
void HypercolumnPairsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  // Not sure if I need to reset these values
  for (int l = 0; l < num_layers_; ++l) {
    Dtype *layer_diff = bottom[1 + l]->mutable_cpu_diff();
    caffe_memset(bottom[1 + l]->count(), 0, layer_diff);
  }

  const Dtype *pairs_data = bottom[0]->cpu_data();
  const Dtype *left_diff = top[0]->mutable_cpu_diff();
  const Dtype *right_diff = top[1]->mutable_cpu_diff();

  for (int n = 0; n < bottom[0]->shape(0); ++n) {
    int offset = n * num_pairs_ * num_layers_ * num_channels_;

    for (int p = 0; p < num_pairs_; ++p) {
      int off = n * num_pairs_ * 4 + p * 4;
      int y1 = static_cast<int>(pairs_data[off + 0]);
      int x1 = static_cast<int>(pairs_data[off + 1]);
      int y2 = static_cast<int>(pairs_data[off + 2]);
      int x2 = static_cast<int>(pairs_data[off + 3]);

      int offset1 = offset + p * num_channels_;

      int cc = 0;
      for (int l = 0; l < num_layers_; ++l) {
        Dtype* layer_diff = bottom[1 + l]->mutable_cpu_diff();
        int layer_channels = bottom[l + 1]->shape(1);

        for (int c = 0; c < layer_channels; ++c) {
          layer_diff[
            n * layer_channels * bottom_h_ * bottom_w_ +
            c * bottom_h_ * bottom_w_ +
            y1 * bottom_w_ +
            x1] += left_diff[offset1 + cc];

          layer_diff[
            n * layer_channels * bottom_h_ * bottom_w_ +
            c * bottom_h_ * bottom_w_ +
            y2 * bottom_w_ +
            x2] += right_diff[offset1 + cc];

          cc += 1;
        }
      }
    }
  }
}

#ifdef CPU_ONLY
//STUB_GPU(HypercolumnPairsLayer);
#endif

INSTANTIATE_CLASS(HypercolumnPairsLayer);
REGISTER_LAYER_CLASS(HypercolumnPairs);

}  // namespace caffe
