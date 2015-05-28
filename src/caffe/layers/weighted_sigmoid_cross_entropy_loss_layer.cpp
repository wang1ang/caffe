#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void WeightedSigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void WeightedSigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "WEIGHTED_SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void WeightedSigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  const Dtype* pos_weight = bottom[2]->cpu_data();
  Dtype loss = 0;
  for (int i = 0; i < count; ++i) {
    Dtype xpos = (Dtype)(input_data[i] >= 0);
    Dtype log_term = log(1 + exp((1 - 2 * xpos) * input_data[i]));
    Dtype log_one_minus_sig_x = -xpos * input_data[i] - log_term;
    Dtype log_sig_x = (1 - xpos) * input_data[i] - log_term;

    //loss -= sample_weight[i] * (input_data[i] * (target[i] - (input_data[i] >= 0)) -
    //    log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0))));
    loss -= 2 * (pos_weight[i] * target[i] * log_sig_x + (1 - pos_weight[i]) * (1 - target[i]) * log_one_minus_sig_x);
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void WeightedSigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    const Dtype* pos_weight = bottom[2]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = -2 * ((1 - sigmoid_output_data[i]) * pos_weight[i] * target[i] - sigmoid_output_data[i] * (1 - pos_weight[i]) * (1 - target[i]));
    }

    /*
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);
    // Multiple by weights
    caffe_mul(count, sample_weight, bottom_diff, bottom_diff);
    */
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(WeightedSigmoidCrossEntropyLossLayer, Backward);
#endif

INSTANTIATE_CLASS(WeightedSigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(WeightedSigmoidCrossEntropyLoss);

}  // namespace caffe
