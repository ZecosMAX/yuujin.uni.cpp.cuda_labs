#include "common.cuh"
#include "labs_tasks.cuh"

#include <chrono>
#include <math.h>

using namespace std::chrono;

__global__ void kernel_lab3_t1_minreduction(float* in_data, float* out_data, int data_size)
{
    // max shared memory = 48 KB (let's say 32KB (32768B) so we have something left)
    // float = 4B
    // 32768B / 4B = 8192

    __shared__ float data[8192]; // allocate 32kb of storage data
                                 // most likely block size (thread count) will never exceed that
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid] = in_data[i];

    __syncthreads();

    // assuming blockDim.x is a power of 2 e.g: [2, 4, 8, 16, ...]
    for (int step = blockDim.x / 2; step > 0; step >>= 1)
    {
        if (tid < step)
            data[tid] = min(data[tid], data[tid + step]);
        __syncthreads();
    }

    // first thread of the block
    // now we work like blocks are threads
    if (tid == 0)
    {
        out_data[blockIdx.x] = data[0];
    }
}

float result_cpu_minreduction(float* in_data, int data_size)
{
    float result = INFINITY;

    for (int i = 0; i < data_size; i++)
    {
        result = min(result, in_data[i]);
    }

    return result;
}

void Lab3_Task1()
{
    printf("===============LAB 3 TASK 1===============\n");

    int N = 1 * 1024 * 1024; // vertical size aka col_size

    float memAllocTime = 0.0f;
    float memCopyTime = 0.0f;
    float gpuExecTime1 = 0.0f;
    float gpuExecTime2 = 0.0f;
    float gpuReadTime = 0.0f;
    float cpuExecTime = 0.0f;
    float overallTime = 0.0f;

    float* hostVectorInData;
    float* hostVectorOutData;
    float cpuResult;

    float* gpuVectorInData;
    float* gpuVectorOutData;
    float gpuResult;

    int _blockSize = 512;

    memAllocTime += (float)TimeEventCPU([&]()
    {
        hostVectorInData = (float*)malloc(N * sizeof(float));
        hostVectorOutData = (float*)malloc(N / _blockSize * sizeof(float));

        fillRandomRealVector(hostVectorInData, N);
    });

    memAllocTime += (float)TimeEventGPU([&]()
    {
        cudaMalloc((void**)&gpuVectorInData, N * sizeof(float));
        cudaMalloc((void**)&gpuVectorOutData, N * sizeof(float));

        cudaMemcpy(gpuVectorInData, hostVectorInData, N * sizeof(float), cudaMemcpyHostToDevice);
    });

    dim3 blockSize = dim3(_blockSize, 1, 1);
    dim3 gridSize = dim3(N / _blockSize, 1, 1);
    gpuExecTime1 += (float)TimeEventGPU([&]()
    {
        kernel_lab3_t1_minreduction<<<gridSize, blockSize>>>(gpuVectorInData, gpuVectorOutData, N);
    });

    gpuReadTime += (float)TimeEventGPU([&]()
    {
        cudaMemcpy(hostVectorOutData, gpuVectorOutData, N / _blockSize * sizeof(float), cudaMemcpyDeviceToHost);
    });
    
    // not counting the cpu-after-cleanup
    gpuResult = result_cpu_minreduction(hostVectorOutData, N / _blockSize);

    cpuExecTime += (float)TimeEventCPU([&]()
    {
        cpuResult = result_cpu_minreduction(hostVectorInData, N);
    });

    overallTime = memAllocTime + memCopyTime + gpuExecTime1 + gpuExecTime2 + gpuReadTime + cpuExecTime;

    printf("-----------------------------------------\n");
    printf("Basic sum of 2 vectors, %d elements: \n", N);
    printf("Time taken to allocate memory       : %f ms\n", memAllocTime);
    printf("Time taken to copy memory to GPU    : %f ms\n", memCopyTime);
    printf("Time taken to execute kernel        : %f ms\n", gpuExecTime1);
    printf("Time taken to copy memory from GPU  : %f ms\n", gpuReadTime);
    printf("CPU execution time                  : %f ms\n", cpuExecTime);
    printf("Overall time taken                  : %f ms\n", overallTime);
    printf("-----------------------------------------\n");

    printf("...Verifying\n");

    if (abs(cpuResult - gpuResult) < 1e-6)
    {
        printf("Code completed successfully!\n");
    }
    else
    {
        printf("Verify failed: cpuResult = %f; \tgpuResult = %f; abs = %f\n", cpuResult, gpuResult, abs(cpuResult - gpuResult));
    }

    free(hostVectorInData);
    free(hostVectorOutData);

    cudaFree(gpuVectorInData);
    cudaFree(gpuVectorOutData);

    printf("=============LAB 3 TASK 1 END=============\n");
}