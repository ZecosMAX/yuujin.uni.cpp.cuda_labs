#include "common.cuh"
#include "labs_tasks.cuh"

#include <chrono>
#include <math.h>

using namespace std::chrono;

__global__ void kernel_lab2_t2(float* a, float* b, float* result, int N)
{
    //assume block size 32
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    result[idx] = a[idx] * b[idx];
}

void result_cpu_lab2_t2(float* a, float* b, float* result, int size)
{
    for (int i = 0; i < size; i++)
    {
        result[i] = a[i] * b[i];
    }
}

std::pair<float, float> measure_stream_time(int streamCount)
{
    int N = 1 * 2 * 3 * 4 * 5 * 6 * 7 * 8;

    float cpuTime = 0.0f;
    float gpuTime = 0.0f;
    float gpuExecTime = 0.0f;

    float* hostVecA;
    float* hostVecB;
    float* hostVecResult;
    float* cpuVecResult;

    float* inputDevPtrA;
    float* inputDevPtrB;
    float* outputDevPtr;

    cpuTime += (float)TimeEventCPU([&]()
        {
            cudaMallocHost((void**)&hostVecA,       N * sizeof(float));
            cudaMallocHost((void**)&hostVecB,       N * sizeof(float));
            cudaMallocHost((void**)&hostVecResult,  N * sizeof(float));
            cpuVecResult = (float*)malloc(N * sizeof(float));

            // заполняем массивы случайными числами на CPU
            fillRandomNaturalVector(hostVecA, N);
            fillRandomNaturalVector(hostVecB, N);
        });

    int streamMemorySize = N * sizeof(float) / streamCount;
    int streamItemSize = N / streamCount;

    gpuTime += (float)TimeEventGPU([&]()
        {
            // создание CUDA-потоков
            cudaStream_t *streams = new cudaStream_t[streamCount];
            for (int i = 0; i < streamCount; ++i)
                cudaStreamCreate(&streams[i]);

            // резервирование на device места для входных массивов
            cudaMalloc((void**)&inputDevPtrA, N * sizeof(float));
            cudaMalloc((void**)&inputDevPtrB, N * sizeof(float));

            // резервирование на device места для выходных массивов
            cudaMalloc((void**)&outputDevPtr, N * sizeof(float));


            // асинхронное копирование массива на device
            for (int i = 0; i < streamCount; ++i)
            {
                cudaMemcpyAsync(inputDevPtrA + i * streamItemSize, hostVecA + i * streamItemSize, streamMemorySize, cudaMemcpyHostToDevice, streams[i]);
                cudaMemcpyAsync(inputDevPtrB + i * streamItemSize, hostVecB + i * streamItemSize, streamMemorySize, cudaMemcpyHostToDevice, streams[i]);
            }

            int blockSize = 32;
            int gridSize = N / (blockSize * streamCount);
            // обработка массива 

            gpuExecTime = TimeEventGPU([&]() {
                for (int i = 0; i < streamCount; ++i)
                {
                    kernel_lab2_t2<<<dim3(gridSize), dim3(blockSize), 0, streams[i]>>>(inputDevPtrA + i * streamItemSize, inputDevPtrB + i * streamItemSize, outputDevPtr + i * streamItemSize, N);
                }
            });

            // асинхронное копирование c device на  host
            for (int i = 0; i < streamCount; ++i)
            {
                cudaMemcpyAsync(
                    hostVecResult + i * streamItemSize,
                    outputDevPtr + i * streamItemSize, streamMemorySize,
                    cudaMemcpyDeviceToHost, streams[i]);
            }

            // синхронизация CUDA-потоков
            cudaDeviceSynchronize();

            // уничтожение CUDA-потоков
            for (int i = 0; i < streamCount; ++i)
                cudaStreamDestroy(streams[i]);
        });

    printf("...Verifying\n");
    result_cpu_lab2_t2(hostVecA, hostVecB, cpuVecResult, N);

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

    cudaFreeHost(hostVecA);
    cudaFreeHost(hostVecB);
    cudaFreeHost(hostVecResult);

    cudaFree(inputDevPtrA);
    cudaFree(inputDevPtrB);
    cudaFree(outputDevPtr);

    free(cpuVecResult);

    return { gpuTime, gpuExecTime };
}

void Lab2_Task2()
{
    printf("===============LAB 2 TASK 2===============\n");

    int streamCount = 16;

    float* gpuTimes = new float[streamCount];
    float* execTimes = new float[streamCount];

    for (size_t i = 1; i < streamCount; i++)
    {
        auto times = measure_stream_time(i);

        gpuTimes[i] = times.first;
        execTimes[i] = times.second;

        printf("Streams:        %i\n", i);
        printf("GPU Time:       %f ms\n", times.first);
        printf("GPU Exec Time:  %f ms\n", times.second);
        printf("\n");
    }

    graph_vector(gpuTimes, streamCount, 8);

    printf("=============LAB 2 TASK 2 END=============\n");
}
