#include "common.cuh"
#include "labs_tasks.cuh"

#include <chrono>
#include <math.h>
#include <random>

using namespace std::chrono;

__global__ void kernel_lab3_t2_hist(float* in_data, float* out_data, int data_size, int bins, float range_min, float range_max, bool density)
{
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // skip all data outside out bounds
    if (i >= data_size)
        return;
    

    // max shared memory = 48 KB (let's say 32KB (32768B) so we have something left)
    // float = 4B
    // 32768B / 4B = 8192
    // allocate 32kb of storage data
    // most likely block size (thread count) will never exceed that
    __shared__ float data[8192]; 
    
    data[tid] = in_data[i];

    __syncthreads();

    // 12 KB (12288B) of shared data is left 
    // 12288B / 4B = 3072
    // so we can have up to 3072 float bins
    // let's use 2048 because i feel like it
    // also
    // why use floats for bins? 
    // 1. higher dynamic range than UInt32 (10^309 vs 10^9 [~2.47B])
    // 2. usually histogramms are computed as 'density' when count is divided by data size, and int can't support 0..1 range
    __shared__ float bins_data[2048]; 

    for (int i = threadIdx.x; i < 2048; i += blockDim.x)
        bins_data[i] = 0.0f;

    __syncthreads();

    float range_size = range_max - range_min;
    float range_step = range_size / bins;

    // if range is [0; 100]
    // bins = 10
    // range_step = (100 - 0) / 10 = 10
    // then value [0; 9.99...] -> 0; [10; 19.99...] -> 1; [20; 20.99...] -> 2; ...
    // or
    // [0 * range_step; 1 * range_step), [1 * range_step; 2 * range_step); ...
    // dividing it by range_step will give vector like: [0, 1); [1, 2); [2, 3)...
    // so floor(value / range_step) will give a bin index.
    int bin_index = (int)(data[tid] / range_step);

    // handle if data is outside the bounds
    if (bin_index < 0)
        bin_index = 0;

    if (bin_index >= bins)
        bin_index = bins - 1;

    if(density)
        atomicAdd(&((float*)bins_data)[bin_index], (1.0f / data_size));
    else
        atomicAdd(&((float*)bins_data)[bin_index], 1.0f);

    __syncthreads();

    // at this point [bins] holds the distribution of this block's data portion.
    // we have to add this to output.
    // because [bins] != [blockDim.x] we could:
    // 1. copy all the data in one thread (bad)
    // 2. distribute copying load to all threads inside the block (bad, but better)

    float binsPerThread = (float)bins / blockDim.x;
    if (binsPerThread <= 1.0f)
    {
        if (tid >= bins)
            return;

        atomicAdd(&out_data[tid], bins_data[tid]);
    }
    else
    {
        // for example
        // bins = 144
        // blockDim.x = 32
        // binsPerThread = 4.5
        // floor(binsPerThread) = 4
        // floor(binsPerThread) * blockDim.x = 128
        // remainder = 144 - 128 = 16
        // so... threads [0..rem-1] will copy 5 (4+1) items, total of 16 (rem) * 5 (4 + 1) = 80 items (heavy threads)
        // threads [rem..31] will copy 4 items, total of 16 (blockDim.x - rem) * 4 = 64 items (light threads)

        int baseline_item_count = (int)binsPerThread;
        int remainder = baseline_item_count * blockDim.x;

        int light_thread_offset = remainder * (baseline_item_count + 1);

        if (tid < remainder)
        {
            for (int i = 0; i < (baseline_item_count + 1); i++)
            {
                atomicAdd(
                    &out_data[tid * (baseline_item_count + 1) + i],
                    bins_data[tid * (baseline_item_count + 1) + i]
                );
            }
        }
        else
        {
            for (int i = 0; i < baseline_item_count; i++)
            {
                atomicAdd(
                    &out_data[light_thread_offset + (tid - remainder) * baseline_item_count + i],
                    bins_data[light_thread_offset + (tid - remainder) * baseline_item_count + i]
                );
            }
        }
    }
}

void result_cpu_hist(float* in_data, float* out_data, int data_size, int bins, float range_min, float range_max, bool density)
{
    for (int i = 0; i < data_size; i++)
    {
        float range_size = range_max - range_min;
        float range_step = range_size / bins;

        //printf("range size: %f\n", range_size);
        //printf("range step: %f\n", range_step);

        // if range is [0; 100]
        // bins = 10
        // range_step = (100 - 0) / 10 = 10
        // then value [0; 9.99...] -> 0; [10; 19.99...] -> 1; [20; 20.99...] -> 2; ...
        // or
        // [0 * range_step; 1 * range_step), [1 * range_step; 2 * range_step); ...
        // dividing it by range_step will give vector like: [0, 1); [1, 2); [2, 3)...
        // so floor(value / range_step) will give a bin index.
        int bin_index = (int)(in_data[i] / range_step);

        //printf("value:     %f\n", in_data[i]);
        //printf("bin_index: %i\n", bin_index);

        // handle if data is outside the bounds
        if (bin_index < 0)
            bin_index = 0;

        if (bin_index >= bins)
            bin_index = bins - 1;

        if (density)
            out_data[bin_index] += (1.0f / data_size);
        else
            out_data[bin_index] += 1.0f;

        //printf("===========================\n", bin_index);
    }
}

void Lab3_Task2()
{
    printf("===============LAB 3 TASK 2===============\n");

    int N = 1 * 1024 * 1024; // data size
    int bins = 96; 

    float mean = 23.5;
    float stddev = 2.5;

    float min_range = -10.5;
    float max_range = +31.5;

    // random device class instance, source of 'true' randomness for initializing random seed
    std::random_device rd{};
    // Mersenne twister PRNG, initialized with seed from previous random device instance
    std::mt19937 gen{ rd() };
    // instance of class std::normal_distribution with specific mean and stddev
    std::normal_distribution<float> d{ mean, stddev };

    float memAllocTime = 0.0f;
    float memCopyTime = 0.0f;
    float gpuExecTime1 = 0.0f;
    float gpuExecTime2 = 0.0f;
    float gpuReadTime = 0.0f;
    float cpuExecTime = 0.0f;
    float overallTime = 0.0f;

    float* hostVectorData;
    float* hostVectorBins;
    float* cpuVectorBins;

    float* gpuVectorData;
    float* gpuVectorBins;

    int _blockSize = 1024;

    memAllocTime += (float)TimeEventCPU([&]()
    {
        hostVectorData = (float*)malloc(N * sizeof(float));
        hostVectorBins = (float*)malloc(bins * sizeof(float));
        cpuVectorBins = (float*)malloc(bins * sizeof(float));

        for (int i = 0; i < N; ++i)
        {
            // get random number with normal distribution using gen as random source
            hostVectorData[i] = d(gen);
        }

        for (int i = 0; i < bins; i++)
        {
            hostVectorBins[i] = 0.0f;
            cpuVectorBins[i] = 0.0f;
        }

    });

    //printf("InputData: \n");
    //print_vector(hostVectorData, N, 0);

    memAllocTime += (float)TimeEventGPU([&]()
    {
        cudaMalloc((void**)&gpuVectorData, N * sizeof(float));
        cudaMalloc((void**)&gpuVectorBins, bins * sizeof(float));

        cudaMemcpy(gpuVectorData, hostVectorData, N * sizeof(float), cudaMemcpyHostToDevice);
    });

    int blocks = N / _blockSize;
    if (blocks == 0)
        blocks = 1;

    dim3 blockSize = dim3(_blockSize, 1, 1);
    dim3 gridSize = dim3(blocks, 1, 1);
    gpuExecTime1 += (float)TimeEventGPU([&]()
    {
        kernel_lab3_t2_hist<<<gridSize, blockSize>>>(gpuVectorData, gpuVectorBins, N, bins, min_range, max_range, true);
    });

    gpuReadTime += (float)TimeEventGPU([&]()
    {
        cudaMemcpy(hostVectorBins, gpuVectorBins, bins * sizeof(float), cudaMemcpyDeviceToHost);
    });

    cpuExecTime += (float)TimeEventCPU([&]()
    {
        result_cpu_hist(hostVectorData, cpuVectorBins, N, bins, min_range, max_range, true);
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
    int fail_index = 0;
    if (verify_vectors(hostVectorBins, cpuVectorBins, bins, &fail_index, 1e-1))
    {
        printf("Code completed successfully!\n");
    }
    else
    {
        printf("Verify failed on index %d\n", fail_index);
        print_vector(hostVectorBins, 20, fail_index - 10);
        print_vector(cpuVectorBins, 20, fail_index - 10);
    }

    graph_vector(hostVectorBins, bins, 16);

    free(hostVectorData);
    free(hostVectorBins);
    free(cpuVectorBins);

    cudaFree(gpuVectorData);
    cudaFree(gpuVectorBins);

    printf("=============LAB 3 TASK 2 END=============\n");
}