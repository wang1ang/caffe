#ifndef MATH_FUNCS_EXT_HPP_
#define MATH_FUNCS_EXT_HPP_

#include <algorithm>
#include <cmath>
#include <limits>
#include <caffe/util/math_functions.hpp>

namespace caffe{

  template <typename Dtype>
  Dtype caffe_cpu_norm(const int n, const Dtype* x);

  template <typename Dtype>
  Dtype caffe_cpu_norm(const int num, const int dim, const Dtype* data, Dtype sub_norm, Dtype* norm);

  template <typename Dtype>
  Dtype caffe_cpu_l2dist(const int n, const Dtype* x, const Dtype* y);

  template <typename Dtype>
  Dtype caffe_cpu_euclidean(const int n, const Dtype* x, const Dtype* y);

  template <typename Dtype>
  Dtype caffe_cpu_cosine(const int n, const Dtype* x, const Dtype* y);  

  template <typename Dtype>
  Dtype caffe_gpu_mean_norm(const Dtype* data, int num, int dim, Dtype sub)
  {
	  Dtype sum = 0;
	  for (int i = 0; i < num; i++){
		  Dtype n(0);
		  caffe_gpu_dot(dim, data + i * dim, data + i * dim, &n);
		  n = sqrt(n) - sub;
		  if (n > sub){
			  sum += n * n;
		  }
	  }
	  return sum / Dtype(num);
  }
  /*template<> float caffe_gpu_mean_norm(const float* data, int num, int dim, float sub);
  template<> double caffe_gpu_mean_norm(const double* data, int num, int dim, double sub);*/

  template <typename Dtype>
  void caffe_gpu_batch_euclidean(const int num_pairs, const int dim, const Dtype* data, Dtype* dist);
  template <typename Dtype>
  void caffe_cpu_pairwise_euclidean(const int row, const int col, const Dtype* data, Dtype* dist);
  template <typename Dtype>
  void caffe_gpu_pairwise_euclidean(const int row, const int col, const Dtype* data, Dtype* dist);
  template <typename Dtype>
  void caffe_gpu_pairwise_euclidean(const int row_a, const int row_b, const int col, const Dtype* data_a, const Dtype* data_b, Dtype* dist);
  template <typename Dtype>
  void caffe_cpu_pairwise_l2dist(const int row, const int col, const Dtype* data, Dtype* dist);
  template <typename Dtype>
  void caffe_gpu_pairwise_l2dist(const int row, const int col, const Dtype* data, Dtype* dist);

  template <typename Dtype>
  void caffe_gpu_norm(const int row, const int col, const Dtype* data, Dtype* norm);
  template <typename Dtype>
  void caffe_gpu_norm_backward(const int row, const int col, const Dtype* data, Dtype* diff, Dtype* norm, const double min_norm, const double alpha);

  template <typename Dtype>
  Dtype ForwardPull(int num, const Dtype* label, const Dtype* dist) {
	  Dtype sum = Dtype(0);
	  for (int k = 0; k < num; k++){
		  for (int i = k + 1; i < num; i++){
			  if (label[i] == label[k]){
				  sum += dist[k * num + i];
			  }
		  }
	  }
	  return sum / Dtype(num * (num - 1));
  }

  template <typename Dtype>
  void BackwardPull(int num, const Dtype* label, const Dtype* dist, Dtype* dist_diff, Dtype alpha) {
	  Dtype sum = Dtype(0);
	  int count = 0;
	  for (int k = 0; k < num; k++){
		  for (int i = k + 1; i < num; i++){
			  if (label[i] == label[k]){
				  sum += dist[k * num + i];
				  count++;
			  }
		  }
	  }
	  if (count == 0){
		  return;
	  }
	  alpha *= Dtype(2.0) / Dtype(num * (num - 1));

	  for (int k = 0; k < num; k++){
		  for (int i = k + 1; i < num; i++){
			  if (label[i] == label[k]){
				  Dtype d = dist[k * num + i];
				  if (d > 0){
					  dist_diff[k * num + i] += alpha;
					  dist_diff[i * num + k] += alpha;
				  }
			  }
		  }
	  }
  }

  template <typename Dtype>
  void caffe_cpu_norm_backward(const int num, const int dim, const Dtype* data, Dtype* diff, Dtype* norm, const double min_norm, const double alpha){

    double alpha_m2 = 2.0 * alpha;

    for (int i = 0; i < num; ++i) {
      Dtype decay_part = norm[i] - min_norm;
      if (decay_part > 0){
        decay_part = Dtype(decay_part * alpha_m2) / (Dtype(num) * norm[i]);
        const Dtype* row0 = data + i * dim;
        Dtype* diff0 = diff + i * dim;
        caffe_axpy(dim, decay_part, row0, diff0);
      }
    }
  }

  //template <typename Dtype>
  //void caffe_gpu_norm_backward(const int num, const int dim, const Dtype* data, Dtype* diff, Dtype* norm, const double min_norm, const double alpha);



  template <typename Dtype>
  void ShowAllDist(const Dtype* dist, int num, int row, int col, Dtype val){

    std::cout << "dist[" << row << "]" << std::endl;
    std::cout << "loss(" << row << "," << col << ")=" << val << "too large" << std::endl;

    for (int k = 0; k < num; ++k){
      std::cout << k << ":" << dist[row * num + k] << " ";
    }
    std::cout << std::endl;
  }

  template <typename Dtype>
  Dtype min_dist(const Dtype* dist, int num, int except) {
    Dtype min = Dtype(30.0);
    for (int k = 0; k < num; k++){
      if (k != except && dist[k] < min){
        min = dist[k];
      }
    }
    return min;
  }


  template <typename Dtype> Dtype neg_log_sigmoid(Dtype x);

  template <typename Dtype> Dtype neg_log_sigmoid_diff(Dtype x);

}


#endif  // MATH_FUNCS_EXT_HPP_