#include "common.cuh"
#include "labs_tasks.cuh"

#include <chrono>
#include <math.h>

using namespace std::chrono;

__global__ void kernel_lab2_t3(float* matrix, int row_size, int col_size, float* result)
{
    //assume block size 32
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx_x >= row_size)
        return;

    if (idx_y >= col_size)
        return;

    result[idx_x * col_size + idx_y] = matrix[idx_y * row_size + idx_x];
}

__global__ void kernel_lab2_t3_shared(float* matrix, int row_size, int col_size, float* result)
{
    //assume block size 32
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx_x >= row_size)
        return;

    if (idx_y >= col_size)
        return;

    result[idx_x * col_size + idx_y] = matrix[idx_y * row_size + idx_x];
}

void result_cpu_lab2_t3(float* matrix, int row_size, int col_size, float* result)
{
    for (int y = 0; y < col_size; y++)
    {
        for (int x = 0; x < row_size; x++)
        {
            result[x * col_size + y] = matrix[y * row_size + x];
        }
    }
}

std::pair<float, float> execute_task3(int block_size_index, int block_size, int sizeMul)
{
    int N = sizeMul * 1024; // vertical size aka col_size
    int M = sizeMul * 1024; // horizonal size aka row_size

    float memAllocTime = 0.0f;
    float memCopyTime = 0.0f;
    float gpuExecTime1 = 0.0f;
    float gpuExecTime2 = 0.0f;
    float gpuReadTime = 0.0f;
    float overallTime = 0.0f;

    float* hostMatrixA;
    float* hostMatrixResult;
    float* cpuMatrixResult;

    float* gpuMatrixA;
    float* gpuMatrixResult;

    memAllocTime += (float)TimeEventCPU([&]()
    {
        hostMatrixA = (float*)malloc(N * M * sizeof(float));
        hostMatrixResult = (float*)malloc(N * M * sizeof(float));
        cpuMatrixResult = (float*)malloc(N * M * sizeof(float));

        // заполняем массивы случайными числами на CPU
        fillRandomNaturalVector(hostMatrixA, N * M);
    });

    memAllocTime += (float)TimeEventGPU([&]()
    {
        cudaMalloc((void**)&gpuMatrixA, N * M * sizeof(float));
        cudaMalloc((void**)&gpuMatrixResult, N * M * sizeof(float));

        // Копируем данные на GPU
        cudaMemcpy(gpuMatrixA, hostMatrixA, N * M * sizeof(float), cudaMemcpyHostToDevice);
    });

    int maxBlockSize = 1024;
    if (block_size > maxBlockSize)
        block_size = maxBlockSize;

    // x * y = 1024
    // y = 1024 / x
    dim3 blockSize = dim3(block_size_index, maxBlockSize / block_size_index, 1);
    dim3 gridSize = dim3(M / blockSize.x, N / blockSize.y, 1);

    gpuExecTime1 += (float)TimeEventGPU([&]()
        {
            // Запускаем и ждём выполнения кернела
            kernel_lab2_t3<<<gridSize, blockSize>>>(gpuMatrixA, M, N, gpuMatrixResult);
        });

    gpuReadTime += (float)TimeEventGPU([&]()
        {
            // скопировать результаты в память CPU
            cudaMemcpy(hostMatrixResult, gpuMatrixResult, N * M * sizeof(float), cudaMemcpyDeviceToHost);
        });

    overallTime = memAllocTime + memCopyTime + gpuExecTime1 + gpuExecTime2 + gpuReadTime;

    printf("...Verifying\n");
    result_cpu_lab2_t3(hostMatrixA, M, N, cpuMatrixResult);

    int fail_index = 0;
    if (verify_vectors(hostMatrixResult, cpuMatrixResult, N * M, &fail_index))
    {
        printf("Code completed successfully!\n");
    }
    else
    {
        printf("Verify failed on index %d\n", fail_index);
        print_vector(hostMatrixResult, 20, fail_index - 10);
        print_vector(cpuMatrixResult, 20, fail_index - 10);
    }

    free(hostMatrixA);
    free(hostMatrixResult);
    free(cpuMatrixResult);

    cudaFree(gpuMatrixA);
    cudaFree(gpuMatrixResult);

    return { overallTime, gpuExecTime1 };
}

void Lab2_Task3() 
{
    int block_sizes[] = { 32, 64, 128, 256, 512, 1024 };
    int dividers[] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 };

    float* overallTimes = new float[11];
    float* execTimes = new float[11];

    float* sizeOTimes = new float[10];
    float* sizeETimes = new float[10];

    printf("===============LAB 2 TASK 3===============\n");

    for (size_t i = 0; i < 11; i++)
    {
        int divider = dividers[i];

        auto time = execute_task3(divider, 1024, 1);
        overallTimes[i] = time.first;
        execTimes[i] = time.second;

        printf("Dim = {%i, %i}\n", divider, 1024 / divider);
        printf("GPU Time:       %f ms\n", time.first);
        printf("GPU Exec Time:  %f ms\n", time.second);
    }
    std::cout << "block size variations (1x1024), (2x512)..." << std::endl;
    std::cout << "overall time:" << std::endl;
    graph_vector(overallTimes, 11, 8);

    std::cout << "execution time:" << std::endl;
    graph_vector(overallTimes, 11, 8);

    for (size_t i = 0; i < 10; i++)
    {
        int divider = dividers[i];

        auto time = execute_task3(divider, 1024, i + 1);
        printf("size = {%i, %i}\n", 1024 * (i + 1), 1024 * (i + 1));
        printf("GPU Time:       %f ms\n", time.first);
        printf("GPU Exec Time:  %f ms\n", time.second);

        sizeOTimes[i] = time.first;
        sizeETimes[i] = time.first;
    }
    std::cout << "data size variations (1x1024), (2x1024)..." << std::endl;

    std::cout << "overall time:" << std::endl;
    graph_vector(sizeOTimes, 10, 8);

    std::cout << "execution time:" << std::endl;
    graph_vector(sizeETimes, 10, 8);

    printf("=============LAB 2 TASK 3 END=============\n");

}
