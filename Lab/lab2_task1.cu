#include "common.cuh"
#include "labs_tasks.cuh"

#include <chrono>
#include <math.h>

using namespace std::chrono;

__device__ int reverse[32] = { 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };
__device__ int forward[32] = { 0, 1, 2, 3, 4, 5, 6, 8, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 };

__global__ void kernel_no_coalece(float* a, float* b, float* result, int N)
{
    //assume block size 32
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 5)
    {
        idx = 6;
    }
    else if (idx == 6)
    {
        idx = 5;
    }

    result[idx] = a[idx] * b[idx];
    //result[(idx + 1) % N] = a[(idx + 1) % N] * b[(idx + 1) % N];
}

__global__ void kernel_coalece(float* a, float* b, float* result, int N)
{
    //assume block size 32
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    result[idx] = a[idx] * b[idx];
}

void result_cpu(float* a, float* b, float* result, int size)
{
    for (int i = 0; i < size; i++)
    {
        result[i] = a[i] * b[i];
    }
}

void Lab2_Task1()
{
    int N = 1024;

    float memAllocTime = 0.0f;
    float memFillTime = 0.0f;
    float memCopyTime = 0.0f;
    float gpuExecTime1 = 0.0f;
    float gpuExecTime2 = 0.0f;
    float gpuReadTime = 0.0f;
    float overallTime = 0.0f;

    printf("===============LAB 2 TASK 1===============\n");

    printf("...Allocating memory\n");
    
    float* hostVecA;
    float* hostVecB;
    float* hostVecResult;
    float* cpuVecResult;

    memAllocTime += (float)TimeEventCPU([&]() 
        {
            hostVecA = (float*)malloc(N * sizeof(float));
            hostVecB = (float*)malloc(N * sizeof(float));
            hostVecResult = (float*)malloc(N * sizeof(float));
            cpuVecResult = (float*)malloc(N * sizeof(float));
        });

    float* gpuVecA1 = nullptr;
    float* gpuVecB1 = nullptr;
    float* gpuVecResult1 = nullptr;

    float* gpuVecA2 = nullptr;
    float* gpuVecB2 = nullptr;
    float* gpuVecResult2 = nullptr;

    memAllocTime += (float)TimeEventGPU([&]() 
        {
            cudaMalloc((void**)&gpuVecA1, N * sizeof(float));
            cudaMalloc((void**)&gpuVecB1, N * sizeof(float));
            cudaMalloc((void**)&gpuVecResult1, N * sizeof(float));

            cudaMalloc((void**)&gpuVecA2, N * sizeof(float));
            cudaMalloc((void**)&gpuVecB2, N * sizeof(float));
            cudaMalloc((void**)&gpuVecResult2, N * sizeof(float));
        });

    printf("...Filling host memory\n");
    memFillTime += (float)TimeEventCPU([&]()
        {
            // заполняем массивы случайными числами на CPU
            fillRandomNaturalVector(hostVecA, N);
            fillRandomNaturalVector(hostVecB, N);
        });

    printf("...Copying host memory to device\n");
    memCopyTime += (float)TimeEventGPU([&]()
        {
            // Копируем данные на GPU
            cudaMemcpy(gpuVecA1, hostVecA, N * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(gpuVecB1, hostVecB, N * sizeof(float), cudaMemcpyHostToDevice);

            cudaMemcpy(gpuVecA2, hostVecA, N * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(gpuVecB2, hostVecB, N * sizeof(float), cudaMemcpyHostToDevice);
        });

    printf("...Running GPU kernel\n");

    int blockSize = 128;

    // warmup kernel, see: https://stackoverflow.com/questions/57709333/cuda-kernel-runs-faster-the-second-time-it-is-run-why
    TimeEventGPU([&]()
    {
        kernel_coalece<<<dim3((N / blockSize)), dim3(blockSize)>>>(gpuVecA2, gpuVecB2, gpuVecResult2, N);
    });

    gpuExecTime1 += (float)TimeEventGPU([&]()
        {
            // Запускаем и ждём выполнения кернела
            kernel_no_coalece <<<dim3(N / blockSize), dim3(blockSize)>>> (gpuVecA1, gpuVecB1, gpuVecResult1, N);
        });

    gpuExecTime2 += (float)TimeEventGPU([&]()
        {
            // Запускаем и ждём выполнения кернела
            kernel_coalece << <dim3((N / blockSize)), dim3(blockSize) >> > (gpuVecA2, gpuVecB2, gpuVecResult2, N);
        });

    

    printf("...Copying GPU result to Host\n");
    gpuReadTime += (float)TimeEventGPU([&]()
        {
            // скопировать результаты в память CPU
            cudaMemcpy(hostVecResult, gpuVecResult1, N * sizeof(float), cudaMemcpyDeviceToHost);
        });

    overallTime = memAllocTime + memFillTime + memCopyTime + gpuExecTime1 + gpuExecTime2 + gpuReadTime;

    // запрашиваем и выводим время между событиями

    printf("-----------------------------------------\n");
    printf("Basic sum of 2 vectors, %d elements: \n", N);
    printf("Time taken to allocate memory       : %f ms\n", memAllocTime);
    printf("Time taken to fill memory           : %f ms\n", memFillTime);
    printf("Time taken to copy memory to GPU    : %f ms\n", memCopyTime);
    printf("Time taken to execute kernel (c)    : %f ms\n", gpuExecTime2);
    printf("Time taken to execute kernel (no c) : %f ms\n", gpuExecTime1);
    printf("Time taken to copy memory from GPU  : %f ms\n", gpuReadTime);
    printf("Overall time taken                  : %f ms\n", overallTime);
    printf("-----------------------------------------\n");

    printf("...Verifying\n");
    result_cpu(hostVecA, hostVecB, cpuVecResult, N);

    int fail_index = 0;
    if (verify_vectors(hostVecResult, cpuVecResult, N, &fail_index))
    {
        printf("Code completed successfully!\n");
    }
    else
    {
        printf("Verify failed on index %d\n", fail_index);
        print_vector(hostVecResult, 20, fail_index - 10);
        print_vector(cpuVecResult, 20, fail_index - 10);
    }

    // освободить выделенную память
    cudaFree(gpuVecA1);
    cudaFree(gpuVecB1);
    cudaFree(gpuVecA2);
    cudaFree(gpuVecB2);
    cudaFree(gpuVecResult1);
    cudaFree(gpuVecResult2);

    free(hostVecA);
    free(hostVecB);
    free(hostVecResult);
    free(cpuVecResult);

    printf("=============LAB 2 TASK 1 END=============\n");

}