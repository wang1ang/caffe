#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"


namespace caffe {

template <typename Dtype>
void make_pair(Dtype *v, int n, int y1, int x1, int y2, int x2) {
    v[4*n  ] = y1;
    v[4*n+1] = x1;
    v[4*n+2] = y2;
    v[4*n+3] = x2;
}

template <typename TypeParam>
class HypercolumnPairsLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  HypercolumnPairsLayerTest()
      : blob_bottom_pairs_(new Blob<Dtype>(2, 3, 4, 1)),
        blob_bottom_0_(new Blob<Dtype>(2, 7, 10, 5)),
        blob_bottom_1_(new Blob<Dtype>(2, 3, 10, 5)),
        blob_bottom_2_(new Blob<Dtype>(2, 10, 10, 5)),
        blob_top_left_(new Blob<Dtype>()),
        blob_top_right_(new Blob<Dtype>()) {
  }
  virtual void SetUp() {
    Dtype *pairs_data = blob_bottom_pairs_->mutable_cpu_data();

    int c = 0;
    make_pair(pairs_data, c++, 0, 0, 0, 1);
    make_pair(pairs_data, c++, 0, 2, 9, 4);
    make_pair(pairs_data, c++, 2, 3, 4, 2);

    make_pair(pairs_data, c++, 1, 0, 2, 0);
    make_pair(pairs_data, c++, 1, 2, 1, 1);
    make_pair(pairs_data, c++, 2, 3, 4, 2);

    vector<int> shape;
    shape.push_back(blob_bottom_pairs_->shape(0));
    shape.push_back(blob_bottom_pairs_->shape(1));
    shape.push_back(blob_bottom_pairs_->shape(2));
    blob_bottom_pairs_->Reshape(shape);

    FillerParameter filler_param;
    filler_param.set_min(-3);
    filler_param.set_max(3);
    UniformFiller<Dtype> filler0(filler_param);
    filler0.Fill(this->blob_bottom_0_);

    filler_param.set_min(-2);
    filler_param.set_max(2);
    UniformFiller<Dtype> filler1(filler_param);
    filler1.Fill(this->blob_bottom_1_);

    filler_param.set_min(-10);
    filler_param.set_max(10);
    UniformFiller<Dtype> filler2(filler_param);
    filler2.Fill(this->blob_bottom_2_);

    blob_bottom_vec_0_.push_back(blob_bottom_pairs_);
    blob_bottom_vec_0_.push_back(blob_bottom_0_);
    blob_bottom_vec_0_.push_back(blob_bottom_1_);
    blob_bottom_vec_0_.push_back(blob_bottom_2_);
    blob_top_vec_.push_back(blob_top_left_);
    blob_top_vec_.push_back(blob_top_right_);

    num_channels_0_ = 0;
    for (int l = 1; l < this->blob_bottom_vec_0_.size(); ++l) {
      num_channels_0_ += this->blob_bottom_vec_0_[l]->shape(1);
    }
  }

  virtual ~HypercolumnPairsLayerTest() {
    delete blob_bottom_pairs_;
    delete blob_bottom_0_;
    delete blob_bottom_1_;
    delete blob_bottom_2_;
    delete blob_top_left_;
    delete blob_top_right_;
  }

  Blob<Dtype>* blob_bottom_pairs_;
  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_left_;
  Blob<Dtype>* const blob_top_right_;
  vector<Blob<Dtype>*> blob_bottom_vec_0_;
  vector<Blob<Dtype>*> blob_top_vec_;
  int num_channels_0_;
};

TYPED_TEST_CASE(HypercolumnPairsLayerTest, TestDtypesAndDevices);

TYPED_TEST(HypercolumnPairsLayerTest, TestSetupNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  HypercolumnPairsLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_0_, this->blob_top_vec_);

  int N = this->blob_bottom_pairs_->shape(0);
  int P = this->blob_bottom_pairs_->shape(1);
  EXPECT_EQ(this->blob_top_left_->shape(0), N * P);
  EXPECT_EQ(this->blob_top_left_->shape(1), this->num_channels_0_);
  EXPECT_EQ(this->blob_top_right_->shape(0), N * P);
  EXPECT_EQ(this->blob_top_right_->shape(1), this->num_channels_0_);
}

TYPED_TEST(HypercolumnPairsLayerTest, TestGradient0) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  HypercolumnPairsLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-1, 1e-2);
  for (int l = 1; l < this->blob_bottom_vec_0_.size(); ++l) {
    // Test gradient for one input layer
    checker.CheckGradient(&layer, this->blob_bottom_vec_0_, this->blob_top_vec_, l);
  }
}

}  // namespace caffe
