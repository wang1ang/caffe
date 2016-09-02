#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include <iostream>  // TODO

namespace caffe {

template <typename Dtype>
void make_location(Dtype *v, int n, int y, int x) {
    v[2*n  ] = y;
    v[2*n+1] = x;
}

template <typename TypeParam>
class HypercolumnExtractorLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  HypercolumnExtractorLayerTest()
      : blob_bottom_locations_(new Blob<Dtype>(2, 3, 2, 1)),
        blob_bottom_0_(new Blob<Dtype>(2, 7, 20, 16)),
        blob_bottom_1_(new Blob<Dtype>(2, 3, 10, 8)),
        blob_bottom_2_(new Blob<Dtype>(2, 10, 5, 4)),
        blob_top_(new Blob<Dtype>()) {
  }
  virtual void SetUp() {
    Dtype *locations_data = blob_bottom_locations_->mutable_cpu_data();

    this->scales_0_.push_back(1);
    this->scales_0_.push_back(2);
    this->scales_0_.push_back(4);
    this->offsets_height_0_.push_back(0);
    this->offsets_height_0_.push_back(0);
    this->offsets_height_0_.push_back(0);
    this->offsets_width_0_.push_back(0);
    this->offsets_width_0_.push_back(0);
    this->offsets_width_0_.push_back(0);

    int c = 0;
    make_location(locations_data, c++, 0, 0);
    make_location(locations_data, c++, 0, 0);
    make_location(locations_data, c++, 0, 0);

    make_location(locations_data, c++, 0, 0);
    make_location(locations_data, c++, 0, 0);
    make_location(locations_data, c++, 0, 0);

    vector<int> shape;
    shape.push_back(blob_bottom_locations_->shape(0));
    shape.push_back(blob_bottom_locations_->shape(1));
    shape.push_back(blob_bottom_locations_->shape(2));
    blob_bottom_locations_->Reshape(shape);

    FillerParameter filler_param;
    filler_param.set_min(-3);
    filler_param.set_max(3);
    UniformFiller<Dtype> filler0(filler_param);
    filler0.Fill(this->blob_bottom_0_);

    filler_param.set_min(-2);
    filler_param.set_max(2);
    UniformFiller<Dtype> filler1(filler_param);
    filler1.Fill(this->blob_bottom_1_);

    filler_param.set_min(-1);
    filler_param.set_max(1);
    UniformFiller<Dtype> filler2(filler_param);
    filler2.Fill(this->blob_bottom_2_);

    blob_bottom_vec_0_.push_back(blob_bottom_locations_);
    blob_bottom_vec_0_.push_back(blob_bottom_0_);
    blob_bottom_vec_0_.push_back(blob_bottom_1_);
    blob_bottom_vec_0_.push_back(blob_bottom_2_);
    blob_top_vec_.push_back(blob_top_);

    num_channels_0_ = 0;
    for (int l = 1; l < this->blob_bottom_vec_0_.size(); ++l) {
      num_channels_0_ += this->blob_bottom_vec_0_[l]->shape(1);
    }
  }

  virtual ~HypercolumnExtractorLayerTest() {
    delete blob_bottom_locations_;
    delete blob_bottom_0_;
    delete blob_bottom_1_;
    delete blob_bottom_2_;
    delete blob_top_;
  }

  Blob<Dtype>* blob_bottom_;
  Blob<Dtype>* const blob_bottom_locations_;
  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_0_;
  vector<Blob<Dtype>*> blob_top_vec_;
  int num_channels_0_;

  vector<float> scales_0_;
  vector<float> offsets_height_0_;
  vector<float> offsets_width_0_;
};

TYPED_TEST_CASE(HypercolumnExtractorLayerTest, TestDtypesAndDevices);

TYPED_TEST(HypercolumnExtractorLayerTest, TestSetupNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  for (int i = 0; i < this->scales_0_.size(); ++i) {
    layer_param.mutable_hypercolumn_extractor_param()->add_scale(this->scales_0_[i]);
    layer_param.mutable_hypercolumn_extractor_param()->add_offset_height(this->offsets_height_0_[i]);
    layer_param.mutable_hypercolumn_extractor_param()->add_offset_width(this->offsets_width_0_[i]);
  }
  HypercolumnExtractorLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_0_, this->blob_top_vec_);

  int N = this->blob_bottom_locations_->shape(0);
  int P = this->blob_bottom_locations_->shape(1);
  EXPECT_EQ(this->blob_top_->shape(0), N * P);
  EXPECT_EQ(this->blob_top_->shape(1), this->num_channels_0_);
}

TYPED_TEST(HypercolumnExtractorLayerTest, TestGradient0) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  for (int i = 0; i < this->scales_0_.size(); ++i) {
    layer_param.mutable_hypercolumn_extractor_param()->add_scale(this->scales_0_[i]);
    layer_param.mutable_hypercolumn_extractor_param()->add_offset_height(this->offsets_height_0_[i]);
    layer_param.mutable_hypercolumn_extractor_param()->add_offset_width(this->offsets_width_0_[i]);
  }
  HypercolumnExtractorLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  for (int l = 1; l < this->blob_bottom_vec_0_.size(); ++l) {
    // Test gradient for one input layer
    checker.CheckGradient(&layer, this->blob_bottom_vec_0_, this->blob_top_vec_, l);
  }
}

}  // namespace caffe
