#include<caffe/util/math_funs_ext.hpp>
#ifdef _WIN64
#include "caffe/util/FastOperations.h"
#endif

namespace caffe {
    template <typename Dtype>
    Dtype caffe_cpu_norm(const int dim, const Dtype* data){
        Dtype sum(0);
        for (int k = 0; k < dim; k++){
            sum += data[k] * data[k];
        }
        return sqrt(sum);
    }
    template float caffe_cpu_norm<float>(const int dim, const float* data);
    template double caffe_cpu_norm<double>(const int dim, const double* data);

    template <typename Dtype>
    Dtype caffe_cpu_norm(const int num, const int dim, const Dtype* data, Dtype sub_norm, Dtype* norm) {
        Dtype sum(0);
        for (int i = 0; i < num; i++){
            Dtype n = caffe_cpu_norm<Dtype>(dim, data + i * dim);
            Dtype sub = n - sub_norm;
            if (sub > 0){
                sum += sub * sub;
            }
            if (norm != NULL){
                norm[i] = n;
            }
        }
        return sum / Dtype(num);
    }
    template float caffe_cpu_norm<float>(const int num, const int dim, const float* data, float sub_norm, float* norm);
    template double caffe_cpu_norm<double>(const int num, const int dim, const double* data, double sub_norm, double* norm);

#ifdef _WIN64
    template<> float caffe_cpu_l2dist<float>(const int n, const float* x, const float* y){
        return ns_base::ns_sse::EuclideanDistA(x, y, n);
    }

    template<> double caffe_cpu_l2dist<double>(const int n, const double* x, const double* y){
        return ns_base::ns_sse::EuclideanDistA(x, y, n);
    }
#else
    template <typename Dtype>
    Dtype caffe_cpu_l2dist(const int dim, const Dtype* v1, const Dtype* v2){
        Dtype res = Dtype(0);
        for (int i = 0; i < dim; i++){
            Dtype d = v1[i] - v2[i];
            res += d * d;
        }
        return res;
    }
    template float caffe_cpu_l2dist<float>(const int dim, const float* v1, const float* v2);
    template double caffe_cpu_l2dist<double>(const int dim, const double* v1, const double* v2);
#endif
    template <typename Dtype>
    Dtype caffe_cpu_euclidean(const int n, const Dtype* x, const Dtype* y){
        return sqrt(caffe_cpu_l2dist<Dtype>(n, x, y));
    }
    template float caffe_cpu_euclidean<float>(const int n, const float* x, const float* y);
    template double caffe_cpu_euclidean<double>(const int n, const double* x, const double* y);

    template <typename Dtype>
    Dtype caffe_cpu_cosine(const int n, const Dtype* x, const Dtype* y){
        return caffe_cpu_dot(n, x, y) / (caffe_cpu_norm(n, x) * caffe_cpu_norm(n, y));
    }
    template float caffe_cpu_cosine(const int n, const float* x, const float* y);
    template double caffe_cpu_cosine(const int n, const double* x, const double* y);

    template<> float neg_log_sigmoid<float>(float x) {
        if (x <= -14.6f){
            return -x;
        }
        if (x >= 16.7){
            return 0.f;
        }
        return log(1.f + exp(-x));
    }

    template<> double neg_log_sigmoid<double>(double x) {
        if (x <= -34){
            return -x;
        }
        if (x >= 36.8){
            return 0.0;
        }
        return log(1.0 + exp(-x));
    }

    template<> float neg_log_sigmoid_diff<float>(float x) {
        if (x <= -14.6f){
            return -1.f;
        }
        if (x >= 16.7){
            return 0.f;
        }
        return -1.f / (1.f + exp(x));
    }

    template<> double neg_log_sigmoid_diff<double>(double x) {
        if (x <= -34){
            return -1.0;
        }
        if (x >= 36.8){
            return 0.0;
        }
        return -1.0 / (1.0 + exp(x));
    }

    template <typename Dtype>
    void caffe_cpu_pairwise_euclidean(const int row, const int col, const Dtype* data, Dtype* dist){
        for (int i = 0; i < row; i++){
            for (int j = i + 1; j < row; j++){
                dist[j * row + i] = dist[i * row + j] = sqrt(caffe_cpu_l2dist(col, data + i * col, data + j * col));
            }
            dist[i * row + i] = Dtype(0);
        }
    }
    template void caffe_cpu_pairwise_euclidean<float>(const int row, const int col, const float* data, float* dist);
    template void caffe_cpu_pairwise_euclidean<double>(const int row, const int col, const double* data, double* dist);

    template <typename Dtype>
    void caffe_cpu_pairwise_l2dist(const int row, const int col, const Dtype* data, Dtype* dist) {
        for (int i = 0; i < row; i++){
            for (int j = i + 1; j < row; j++){
                dist[j * row + i] = dist[i * row + j] = caffe_cpu_l2dist(col, data + i * col, data + j * col); // exp(-sum)
            }
            dist[i * row + i] = Dtype(0);
        }
    }
    template void caffe_cpu_pairwise_l2dist<float>(const int row, const int col, const float* data, float* dist);
    template void caffe_cpu_pairwise_l2dist<double>(const int row, const int col, const double* data, double* dist);
}