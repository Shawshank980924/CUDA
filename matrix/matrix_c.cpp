#include <iostream>
#include <chrono>
#include <omp.h>
#include <x86intrin.h>
#include <xmmintrin.h>  
using namespace std;
using namespace chrono;

void matrix_mult(float* A, float* B, float* C, int m, int n, int p) {
    // 使用OpenMP并行计算矩阵乘积
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            // 计算C[i][j]的值
            __m128 sum = _mm_setzero_ps();
            for (int k = 0; k < n; k += 4) {
                __m128 a = _mm_load_ps(&A[i*n+k]);
                __m128 b = _mm_load_ps(&B[j*p+k]);
                sum = _mm_add_ps(sum, _mm_mul_ps(a, b));
            }
            float res[4];
            _mm_store_ps(res, sum);
            C[i*p+j] = res[0] + res[1] + res[2] + res[3];
        }
    }
}

int main() {
    // 设置矩阵的维数
    int m = 1000, n = 1000, p = 1000;
    // 初始化矩阵A和B
    float *A = static_cast<float*>(_mm_malloc(n * m * sizeof(float), 16));
    float *B = static_cast<float*>(_mm_malloc(m * p * sizeof(float), 16));
    float *C = static_cast<float*>(_mm_malloc(n * p * sizeof(float), 16));
    for (int i = 0; i < m*n; i++) {
        A[i] = i;
    }
    for (int i = 0; i < n*p; i++) {
        B[i] = i;
    }
    // 初始化矩阵C
    // 计时开始
    auto start_time = high_resolution_clock::now();
    // 计算矩阵乘积
    matrix_mult(A, B, C, m, n, p);
    // 计时结束
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time);
    cout << "矩阵乘积计算时间：" << duration.count() << "ms" << endl;
    // 释放内存
    _mm_free(A);
    _mm_free(B);
    _mm_free(C);

    return 0;
}
