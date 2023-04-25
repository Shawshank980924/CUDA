#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <x86intrin.h>
#include <xmmintrin.h>  
using namespace std;
using namespace chrono;


#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>

#include <iostream>
#include <chrono>
#include <vector>

// ...将您的原始 CUDA 代码放在这里...
__global__ void matrix_mul_kernel(float* A, float* B, float* C, int m, int n, int p) {
     int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < p) {
        float sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += A[row*n+k] * B[k*p+col];
        }
        C[row*p+col] = sum;
    }
   
}

int main() {
    int m,n,p;
    printf("请输入m n p");
    scanf("%d %d %d", &m, &n, &p);

    // 在设备上分配内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * n * sizeof(float));
    cudaMalloc(&d_B, n * p * sizeof(float));
    cudaMalloc(&d_C, m * p * sizeof(float));

    // 在主机上分配内存，并初始化矩阵数据
    float *h_A = new float[n * m];
    float *h_B = new float[n * p];
    float *h_C = new float[m * p];
    // ...初始化矩阵数据...
    for (int i = 0; i < m*n; i++) {
        h_A[i] = i;
    }
    for (int i = 0; i < n*p; i++) {
        h_B[i] = i;
    }
    auto start_time = high_resolution_clock::now();
    
    // 将矩阵数据从主机复制到设备上
    cudaMemcpy(d_A, h_A, n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, m * p * sizeof(float), cudaMemcpyHostToDevice);

    // 进行矩阵乘法计算
    // ...
    matrix_mul_kernel<<< 1,32>>>(d_A, d_B, d_C, m, n, p);
    
    // 将结果从设备复制回主机
    cudaMemcpy(h_C, d_C, n * p * sizeof(float), cudaMemcpyDeviceToHost);
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time);
    cout << "矩阵乘积计算时间：" << duration.count() << "ms" << endl;
    // 释放内存
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
template <typename scalar_t>
__global__ void matrix_mul_kernel(const scalar_t* A, const scalar_t* B, scalar_t* C, int m, int n, int p) {
    // ...将原始 CUDA 代码放在这里，但需使用模板标量类型 scalar_t 替换 float 类型...
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < p) {
        float sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += A[row*n+k] * B[k*p+col];
        }
        C[row*p+col] = sum;
    }
}

// 包装函数供 Python 端调用
void matrix_mul_forward_cuda(const at::Tensor& input_A, const at::Tensor& input_B, at::Tensor& output_C) {
  const auto batch_size = input_A.size(0);
  const auto input_size = input_A.size(1);
  const auto output_size = input_B.size(1);

  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  const auto blocks_x = (output_size + 31) / 32;
  const auto blocks_y = (batch_size + 31) / 32;
  const dim3 threads(32, 32);
  const dim3 blocks(blocks_x, blocks_y);

  AT_DISPATCH_FLOATING_TYPES(input_A.scalar_type(), "matrix_mul_forward_cuda", ([&] {
    matrix_mul_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
      input_A.data_ptr<scalar_t>(),
      input_B.data_ptr<scalar_t>(),
      output_C.data_ptr<scalar_t>(),
      batch_size,
      input_size,
      output_size);
  }));
}
