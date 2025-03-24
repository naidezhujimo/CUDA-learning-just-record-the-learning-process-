by “atomic” we are referring to the indivisibility concept in physics where a thing cannot be broken down further.
“原子”在物理学中指的是一个事物不能被进一步分解的概念。

An atomic operation ensures that a particular operation on a memory location is completed entirely by one thread before another thread can access or modify the same memory location. This prevents race conditions.
原子操作确保在另一个线程可以访问或修改同一内存位置之前，特定线程对内存位置的某个操作完全完成。这防止了竞态条件。

Since we limit the amount of work done on a single piece of memory per unit time throughout an atomic operation, we lose slightly to speed. It is hardware guaranteed to be memory safe at a cost of speed.
由于我们在原子操作中限制了单位时间内对单个内存位置所做的工作量，我们稍微牺牲了一些速度。这是以速度为代价，硬件保证的内存安全。

Integer Atomic Operations
整数原子操作
atomicAdd(int* address, int val): Atomically adds val to the value at address and returns the old value.
atomicAdd(int* address, int val) : 原子性地将 val 添加到 address 的值中，并返回旧值。
atomicSub(int* address, int val): Atomically subtracts val from the value at address and returns the old value.
atomicSub(int* address, int val) : 原子性地从 address 的值中减去 val ，并返回旧值。
atomicExch(int* address, int val): Atomically exchanges the value at address with val and returns the old value.
atomicExch(int* address, int val) : 原子性地交换 address 的值与 val ，并返回旧值。
atomicMax(int* address, int val): Atomically sets the value at address to the maximum of the current value and val.
atomicMax(int* address, int val) : 原子性地将 address 的值设置为当前值和 val 中的最大值。
atomicMin(int* address, int val): Atomically sets the value at address to the minimum of the current value and val.
atomicMin(int* address, int val) : 原子地将 address 处的值设置为当前值和 val 中的最小值。
atomicAnd(int* address, int val): Atomically performs a bitwise AND of the value at address and val.
atomicAnd(int* address, int val) : 原子地对 address 处的值与 val 进行按位与操作。
atomicOr(int* address, int val): Atomically performs a bitwise OR of the value at address and val.
atomicOr(int* address, int val) : 原子地对 address 处的值与 val 进行按位或操作。
atomicXor(int* address, int val): Atomically performs a bitwise XOR of the value at address and val.
atomicXor(int* address, int val) : 原子地对 address 处的值与 val 进行按位异或操作。
atomicCAS(int* address, int compare, int val): Atomically compares the value at address with compare, and if they are equal, replaces it with val. The original value is returned.
atomicCAS(int* address, int compare, int val) : 原子比较 address 处的值与 compare ，如果它们相等，则替换为 val 。返回原始值。
Floating-Point Atomic Operations
浮点原子操作
atomicAdd(float* address, float val): Atomically adds val to the value at address and returns the old value. Available from CUDA 2.0.
atomicAdd(float* address, float val) : 原子将 val 加到 address 处的值上，并返回旧值。从 CUDA 2.0 版本开始可用。
Note: Floating-point atomic operations on double precision variables are supported starting from CUDA Compute Capability 6.0 using atomicAdd(double* address, double val).
注意：从 CUDA 计算能力 6.0 开始，支持使用 atomicAdd(double* address, double val) 对双精度浮点变量执行浮点原子操作。
From Scratch  从零开始
Modern GPUs have special hardware instructions to perform these operations efficiently. They use techniques like Compare-and-Swap (CAS) at the hardware level.
现代 GPU 具有特殊的硬件指令来高效执行这些操作。它们使用诸如比较和交换（CAS）等硬件级技术。

You can think of atomics as a very fast, hardware-level mutex operation. It's as if each atomic operation does this:
您可以将原子操作视为一种非常快速、硬件级的互斥锁操作。就好像每个原子操作都执行以下操作：

lock(memory_location)  锁定(内存位置)
old_value = *memory_location
*memory_location = old_value + increment
unlock(memory_location)
return old_value
__device__ int softwareAtomicAdd(int* address, int increment) {
    __shared__ int lock;
    int old;
    
    if (threadIdx.x == 0) lock = 0;
    __syncthreads();
    
    while (atomicCAS(&lock, 0, 1) != 0);  // Acquire lock
    
    old = *address;
    *address = old + increment;
    
    __threadfence();  // Ensure the write is visible to other threads
    
    atomicExch(&lock, 0);  // Release lock
    
    return old;
}
Mutual Exclusion ⇒ https://www.youtube.com/watch?v=MqnpIwN7dz0&t
相互排斥 ⇒ https://www.youtube.com/watch?v=MqnpIwN7dz0&t
"Mutual":   "相互":
Implies a reciprocal or shared relationship between entities (in this case, threads or processes).
指实体（在这种情况下，线程或进程）之间的相互或共享关系。
Suggests that the exclusion applies equally to all parties involved.
意味着排斥适用于所有相关方。
"Exclusion":   "排除"
Refers to the act of keeping something out or preventing access.
指的是将某物排除在外或阻止其访问的行为。
In this context, it means preventing simultaneous access to a resource.
在这个语境中，意味着防止对资源的并发访问。
#include <cuda_runtime.h>
#include <stdio.h>

// Our mutex structure
struct Mutex {
    int *lock;
};

// Initialize the mutex
__host__ void initMutex(Mutex *m) {
    cudaMalloc((void**)&m->lock, sizeof(int));
    int initial = 0;
    cudaMemcpy(m->lock, &initial, sizeof(int), cudaMemcpyHostToDevice);
}

// Acquire the mutex
__device__ void lock(Mutex *m) {
    while (atomicCAS(m->lock, 0, 1) != 0) {
        // Spin-wait
    }
}

// Release the mutex
__device__ void unlock(Mutex *m) {
    atomicExch(m->lock, 0);
}

// Kernel function to demonstrate mutex usage
__global__ void mutexKernel(int *counter, Mutex *m) {
    lock(m);
    // Critical section
    int old = *counter;
    *counter = old + 1;
    unlock(m);
}

int main() {
    Mutex m;
    initMutex(&m);
    
    int *d_counter;
    cudaMalloc((void**)&d_counter, sizeof(int));
    int initial = 0;
    cudaMemcpy(d_counter, &initial, sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel with multiple threads
    mutexKernel<<<1, 1000>>>(d_counter, &m);
    
    int result;
    cudaMemcpy(&result, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Counter value: %d\n", result);
    
    cudaFree(m.lock);
    cudaFree(d_counter);
    
    return 0;
}