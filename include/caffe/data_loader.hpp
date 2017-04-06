#pragma once

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/sparse_blob.hpp"
#include "caffe/common.hpp"
#include "caffe/text_data_transformer.hpp"

namespace caffe {

    template<class Dtype>
    class IDataLoader {
    public:
        ~IDataLoader() {}
        virtual void Init(const DataDescriptionParameter& param, Phase phase) = 0;
        virtual void Load(const vector<string>& input_data, Blob<Dtype>& out_blob) = 0;
        virtual void Reshape(int batch_size, string input_sample, Blob<Dtype>& out_blob) = 0;
    };

    template<class Dtype>
    class TextDataLoader : public IDataLoader < Dtype > {
        virtual void Init(const DataDescriptionParameter& param, Phase phase);
        virtual void Load(const vector<string>& input_data, Blob<Dtype>& out_blob);
        virtual void Reshape(int batch_size, string input_sample, Blob<Dtype>& out_blob);
    protected:
        Phase phase_;
        DataDescriptionParameter param_;
        SparseDataDimension text_data_dim_;
        shared_ptr<ITextHashing<Dtype>> text_hashing_algo_;
    };

    template<class Dtype>
    class FeatureDataLoader : public IDataLoader < Dtype > {
        virtual void Init(const DataDescriptionParameter& param, Phase phase);
        virtual void Load(const vector<string>& input_data, Blob<Dtype>& out_blob);
        virtual void Reshape(int batch_size, string input_sample, Blob<Dtype>& out_blob);
    protected:
        Phase phase_;
        DataDescriptionParameter param_;
    };

    template<class Dtype>
    class UnifiedDataLoader : public IDataLoader<Dtype> {
    public:
        virtual ~UnifiedDataLoader() {}
        virtual void Init(const DataDescriptionParameter& param, Phase phase);
        virtual void Load(const vector<string>& input_data, Blob<Dtype>& out_blob);
        //virtual void Load(const vector<ImageData>& input_image_data, Blob<Dtype>& out_blob);
        virtual void Reshape(int batch_size, string input_sample, Blob<Dtype>& out_blob);
    protected:
        shared_ptr<IDataLoader<Dtype>> m_data_loader;
    };
}