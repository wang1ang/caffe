#include "caffe/data_loader.hpp"
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include "caffe/util/benchmark.hpp"
#include <boost/timer/timer.hpp>

namespace caffe {

    template<class Dtype>
    void TextDataLoader<Dtype>::Init(const DataDescriptionParameter& param, Phase phase){
        param_ = param;
        phase_ = phase;
        CHECK(param_.type() == DataDescriptionParameter::TEXT) << "data loader type mismatch";
        text_hashing_algo_ = ITextHashing<Dtype>::CreateInstanceAndInit(param_.text_hashing_param().algo_name(), param_.text_hashing_param());
    }


    template<class Dtype>
    void TextDataLoader<Dtype>::Load(const vector<string>& input_data, Blob<Dtype>& out_blob){
        if (out_blob.sparse_blob().get() == nullptr) {
            out_blob.sparse_blob().reset(new SparseBlob<Dtype>);
        }
        text_hashing_algo_->Extract(input_data, *out_blob.sparse_blob());
    }

    template<class Dtype>
    void TextDataLoader<Dtype>::Reshape(int batch_size, string input_sample, Blob<Dtype>& out_blob){
        if (out_blob.sparse_blob().get() == nullptr) {
            out_blob.sparse_blob().reset(new SparseBlob<Dtype>);
        }
        out_blob.sparse_blob()->Reshape(text_hashing_algo_->GetDataDimension(batch_size));
    }

    template<class Dtype>
    void FeatureDataLoader<Dtype>::Init(const DataDescriptionParameter& param, Phase phase){
        param_ = param;
        phase_ = phase;
        CHECK(param_.type() == DataDescriptionParameter::FEATURE) << "data loader type mismatch";
    }

    template<class Dtype>
    void FeatureDataLoader<Dtype>::Load(const vector<string>& input_data, Blob<Dtype>& out_blob){
        int dim = 0;
        vector<int> idx(2);

        // reshape
        vector<string> ps;
        string feature_string = input_data[0];
        boost::trim(feature_string);
        boost::split(ps, feature_string, boost::is_any_of(" "));
        out_blob.Reshape(vector < int > {(int)input_data.size(), (int)ps.size()});
        dim = ps.size();
        #pragma omp parallel for
        for (int i = 0; i < input_data.size(); ++i){
            vector<string> ps;
            string feature_string = input_data[i];
            boost::trim(feature_string);
            boost::split(ps, feature_string, boost::is_any_of(" "));
            CHECK(ps.size() == dim) << "feature dimension mismatch, prev: " << dim << "; cur: " << ps.size();
            idx[0] = (int)i; idx[1] = 0;
            Dtype* p_data = out_blob.mutable_cpu_data() + out_blob.offset(idx);

            for (size_t j = 0; j < ps.size(); ++j){
              p_data[j] = std::stof(ps[j]);
            }
        }
    }

    template<class Dtype>
    void FeatureDataLoader<Dtype>::Reshape(int batch_size, string input_sample, Blob<Dtype>& out_blob){
        Blob<Dtype> tmp_blob;
        vector<int> dim;
        Load(vector < string > {input_sample}, tmp_blob);
        dim = tmp_blob.shape();
        dim[0] = batch_size;
        out_blob.Reshape(dim);
    }

    template<class Dtype>
    void UnifiedDataLoader<Dtype>::Init(const DataDescriptionParameter& param, Phase phase){
        switch (param.type()) {
        case DataDescriptionParameter::TEXT:
            m_data_loader.reset(new TextDataLoader<Dtype>);
            break;
        case DataDescriptionParameter::FEATURE:
            m_data_loader.reset(new FeatureDataLoader<Dtype>);
            break;
        default:
            LOG(FATAL) << "non-supported type";
        }
        m_data_loader->Init(param, phase);
    }

    template<class Dtype>
    void UnifiedDataLoader<Dtype>::Load(const vector<string>& input_data, Blob<Dtype>& out_blob){
        CHECK(m_data_loader.get() != nullptr) << "data loader not initialized";
        m_data_loader->Load(input_data, out_blob);
    }

    template<class Dtype>
    void UnifiedDataLoader<Dtype>::Reshape(int batch_size, string input_sample, Blob<Dtype>& out_blob){
        CHECK(m_data_loader.get() != nullptr) << "data loader not initialized";
        m_data_loader->Reshape(batch_size, input_sample, out_blob);
    }

    INSTANTIATE_CLASS(UnifiedDataLoader);
}