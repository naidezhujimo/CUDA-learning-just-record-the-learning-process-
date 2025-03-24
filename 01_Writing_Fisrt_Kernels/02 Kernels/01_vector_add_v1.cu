#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <windows.h>

#define N 10000000 // 定义向量的大小为 1000 万
#define BLOCK_SIZE 256 // 定义每个线程块的大小为 256 个线程

// CPU向量加法函数
void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// CUDA 核函数
__global__ void vector_add_gpu(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // 计算当前线程的全局索引
    if (i < n){ // 确保当前线程的索引在向量范围内
        c[i] = a[i] + b[i]; // 计算当前线程负责的向量元素的和
    }
}

// 初始化向量函数
void init_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

// 时间函数
double get_time() {
    LARGE_INTEGER frequency;
    LARGE_INTEGER time;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&time);
    return (double)time.QuadPart / frequency.QuadPart;
}

int main() {
    // 定义了host和device的向量指针
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);
    // 分配host内存
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu = (float*)malloc(size);
    // 初始化向量
    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);
    // 分配device内存
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    // 复制数据到device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    // 定义grid和block的数量
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 预热
    printf("正在预热...\n");
    for (int i = 0; i < 3; i++) {
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize(); // 确保GPU上操作完成
    }

    // 对CPU进行基准测试，计算平均时间
    printf("对CPU进行基准测试...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    // 对GPU进行基准测试，计算平均时间
    printf("对GPU进行基准测试...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    // 打印结果
    printf("CPU 平均用时: %f ms\n", cpu_avg_time*1000);
    printf("GPU 平均用时: %f ms\n", gpu_avg_time*1000);
    printf("两者加速比: %fx\n", cpu_avg_time / gpu_avg_time);

    // 验证结果
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5) { // 如果两个结果的差值小于 1e-5，则认为结果正确
            correct = false;
            break;
        }
    }
    printf("结果是 %s\n", correct ? "正确" : "有误");

    // 释放内存
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;

}





