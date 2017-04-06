#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/math_funs_ext.hpp"

namespace caffe {
    template <typename Dtype>
    __global__ void caffe_gpu_batch_euclidean_cuda(const int num_pairs, const int dim, const Dtype* data, Dtype* dist) {
        CUDA_KERNEL_LOOP(t, num_pairs) {
            const Dtype* x = data + t * 2 * dim;
            const Dtype* y = data + (t * 2 + 1) * dim;
            Dtype sum = 0;
            for (int j = 0; j < dim; ++j) {
                Dtype d = x[j] - y[j];
                sum += d*d;
            }
            dist[t] = sqrt(sum);
        }
    }

    template <typename Dtype>
    void caffe_gpu_batch_euclidean(const int num_pairs, const int dim, const Dtype* data, Dtype* dist)
    {
        CUDA_CHECK(cudaMemset(dist, 0, num_pairs * sizeof(Dtype)));
        caffe_gpu_batch_euclidean_cuda<Dtype> << <CAFFE_GET_BLOCKS(num_pairs), CAFFE_CUDA_NUM_THREADS >> >(
            num_pairs, dim, data, dist);
        CUDA_POST_KERNEL_CHECK;
    }

    template void caffe_gpu_batch_euclidean<float>(const int num_pairs, const int dim, const float* data, float* dist);
    template void caffe_gpu_batch_euclidean<double>(const int num_pairs, const int dim, const double* data, double* dist);


    template <typename Dtype>
    __global__ void caffe_gpu_pairwise_euclidean_cuda(const int row, const int num2, const int num22, int col, const Dtype* data, Dtype* dist) {
        CUDA_KERNEL_LOOP(t, (num2 + 1) * num22) {
            int b0 = (t / num22), b1 = (t % num22);
            if (b0 > b1) {
                if (2 * b1 == num2 - 1) return;
                b0 = num2 - b0;
                b1 = num2 - 1 - b1;
            }
            b0 *= 2; b1 *= 2;
            int e0 = (b0 + 2 < row) ? (b0 + 2) : row, e1 = (b1 + 2 < row) ? (b1 + 2) : row;
            for (int k0 = b0; k0 < e0; ++k0) {
                const Dtype* bottom_row0 = data + k0 * col;
                for (int k1 = (b1 > k0 + 1) ? b0 : (k0 + 1); k1 < e1; ++k1) {
                    const Dtype* bottom_row1 = data + k1 * col;
                    Dtype sum = 0;
                    for (int j = 0; j < col; ++j) {
                        Dtype d = bottom_row0[j] - bottom_row1[j];
                        sum += d*d;
                    }
                    dist[k1 * row + k0] = dist[k0 * row + k1] = sqrt(sum);// exp(-sum);
                }
            }
        }
    }

    template <typename Dtype>
    void caffe_gpu_pairwise_euclidean(const int row, const int col, const Dtype* data, Dtype* dist)
    {
        CUDA_CHECK(cudaMemset(dist, 0, row * row * sizeof(Dtype)));
        int num2 = (row + 1) / 2;
        int num22 = (num2 + 1) / 2;
        caffe_gpu_pairwise_euclidean_cuda<Dtype> << <CAFFE_GET_BLOCKS((num2 + 1) * num22), CAFFE_CUDA_NUM_THREADS >> >(
            row, num2, num22, col, data, dist);
        CUDA_POST_KERNEL_CHECK;
    }

    template void caffe_gpu_pairwise_euclidean<float>(const int row, const int col, const float* data, float* dist);
    template void caffe_gpu_pairwise_euclidean<double>(const int row, const int col, const double* data, double* dist);


    template <typename Dtype>
    __global__ void caffe_gpu_pairwise_euclidean_cuda(const int row_a, const int row_b, const int col, const Dtype* data_a, const Dtype* data_b, Dtype* dist) {
        CUDA_KERNEL_LOOP(t, row_a * row_b) {
            const Dtype* x = data_a + t / row_b * col;
            const Dtype* y = data_b + t % row_b * col;
            Dtype sum = 0;
            for (int j = 0; j < col; ++j) {
                Dtype d = x[j] - y[j];
                sum += d*d;
            }
            dist[t] = sqrt(sum);
        }
    }

    template <typename Dtype>
    void caffe_gpu_pairwise_euclidean(const int row_a, const int row_b, const int col, const Dtype* data_a, const Dtype* data_b, Dtype* dist){
        CUDA_CHECK(cudaMemset(dist, 0, row_a * row_b * sizeof(Dtype)));
        caffe_gpu_pairwise_euclidean_cuda<Dtype> << <CAFFE_GET_BLOCKS(row_a * row_b), CAFFE_CUDA_NUM_THREADS >> >(
            row_a, row_b, col, data_a, data_b, dist);
        CUDA_POST_KERNEL_CHECK;
    }

    template void caffe_gpu_pairwise_euclidean<float>(const int row_a, const int row_b, const int col, const float* data_a, const float* data_b, float* dist);
    template void caffe_gpu_pairwise_euclidean<double>(const int row_a, const int row_b, const int col, const double* data_a, const double* data_b, double* dist);


    template <typename Dtype>
    __global__ void caffe_gpu_pairwise_l2dist_cuda(const int num, const int num2, const int num22, int dim, const Dtype* data, Dtype* dist) {
        CUDA_KERNEL_LOOP(t, (num2 + 1) * num22) {
            int b0 = (t / num22), b1 = (t % num22);
            if (b0 > b1) {
                if (2 * b1 == num2 - 1) return;
                b0 = num2 - b0;
                b1 = num2 - 1 - b1;
            }
            b0 *= 2; b1 *= 2;
            int e0 = (b0 + 2 < num) ? (b0 + 2) : num, e1 = (b1 + 2 < num) ? (b1 + 2) : num;
            for (int k0 = b0; k0 < e0; ++k0) {
                const Dtype* bottom_row0 = data + k0 * dim;
                for (int k1 = (b1 > k0 + 1) ? b0 : (k0 + 1); k1 < e1; ++k1) {
                    const Dtype* bottom_row1 = data + k1 * dim;
                    Dtype sum = 0;
                    for (int j = 0; j < dim; ++j) {
                        Dtype d = bottom_row0[j] - bottom_row1[j];
                        sum += d*d;
                    }
                    dist[k1 * num + k0] = dist[k0 * num + k1] = sum;// exp(-sum);
                }
            }
        }
    }

    template <typename Dtype>
    void caffe_gpu_pairwise_l2dist(const int num, const int dim, const Dtype* data, Dtype* dist)
    {
        CUDA_CHECK(cudaMemset(dist, 0, num * num * sizeof(Dtype)));
        int num2 = (num + 1) / 2;
        int num22 = (num2 + 1) / 2;
        caffe_gpu_pairwise_l2dist_cuda<Dtype> << <CAFFE_GET_BLOCKS((num2 + 1) * num22), CAFFE_CUDA_NUM_THREADS >> >(
            num, num2, num22, dim, data, dist);
        CUDA_POST_KERNEL_CHECK;
    }
    template void caffe_gpu_pairwise_l2dist<float>(const int num, const int dim, const float* data, float* dist);
    template void caffe_gpu_pairwise_l2dist<double>(const int num, const int dim, const double* data, double* dist);

    template <typename Dtype>
    __global__ void caffe_gpu_norm_cuda(const int num, const int dim, const Dtype* data, Dtype* norm) {
        CUDA_KERNEL_LOOP(t, num) {
            const Dtype* bottom_row = data + t * dim;
            Dtype row_norm = Dtype(0);
            for (int i = 0; i < dim; i++){
                row_norm += bottom_row[i] * bottom_row[i];
            }
            norm[t] = sqrt(row_norm);
        }
    }

    template <typename Dtype>
    void caffe_gpu_norm(const int num, const int dim, const Dtype* data, Dtype* norm)
    {
        caffe_gpu_norm_cuda<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> >(
            num, dim, data, norm);
        CUDA_POST_KERNEL_CHECK;
    }
    template void caffe_gpu_norm(const int num, const int dim, const float* data, float* norm);
    template void caffe_gpu_norm(const int num, const int dim, const double* data, double* norm);

    template <typename Dtype>
    __global__ void caffe_gpu_norm_backward_cuda(const int num, const int dim, const Dtype* data, Dtype* diff, const Dtype* norm, const double min_norm, const double alpha) {
        CUDA_KERNEL_LOOP(t, num) {
            Dtype d = norm[t] - min_norm;
            if (d <= 0){
                return;
            }
            Dtype m = d * alpha / (Dtype(num) * norm[t]);
            const Dtype* row_data = data + t * dim;
            Dtype* row_diff = diff + t * dim;
            // caffe_gpu_axpy(dim, m, row_data, row_diff); //__host__ function
            for (int j = 0; j < dim; j++){
                row_diff[j] += m * row_data[j];
            }
        }
    }

    template
        __global__ void caffe_gpu_norm_backward_cuda<float>(const int num, const int dim, const float* data, float* diff, const float* norm, const double min_norm, const double alpha);
    template
        __global__ void caffe_gpu_norm_backward_cuda<double>(const int num, const int dim, const double* data, double* diff, const double* norm, const double min_norm, const double alpha);

    template <typename Dtype>
    void caffe_gpu_norm_backward(const int num, const int dim, const Dtype* data, Dtype* diff, Dtype* norm, const double min_norm, const double alpha) {
        caffe_gpu_norm_backward_cuda<Dtype> << <CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS >> >
            (num, dim, data, diff, norm, min_norm, 2 * alpha);
        CUDA_POST_KERNEL_CHECK;
    }

    template void caffe_gpu_norm_backward<float>(const int num, const int dim, const float* data, float* diff, float* norm, const double min_norm, const double alpha);
    template void caffe_gpu_norm_backward<double>(const int num, const int dim, const double* data, double* diff, double* norm, const double min_norm, const double alpha);
}