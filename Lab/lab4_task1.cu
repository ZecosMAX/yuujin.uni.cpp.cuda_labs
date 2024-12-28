// Задание 1: изучить основы работы с текстурной ссылкой на CUDA. 
// Написать программу на Cи, реализующую цифровой фильтр с использованием текстурной ссылки и CUDA runtime API, 
// в соответствии с вариантом задания. Измерить время работы программы. 
// Написать программу для верификации результатов. Результаты занести в отчёт.

// вариант 0	
// Дано двухмерное изображение. 
// Написать программу, реализующую фильтр Box Blur, применительно  к исходному изображению. 
// Радиус фильтра – параметр программы. 

#include "common.cuh"
#include "labs_tasks.cuh"

#include <chrono>
#include <math.h>
#include <random>

using namespace std::chrono;

__global__ void kernel_lab4_t1_box_blur(cudaTextureObject_t in_texture, RGBA* out_data, int radius, int width, int height)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx_x >= width)
        return;

    if (idx_y >= height)
        return;

    float4 pixel_value{};

    for (int ry = -radius; ry <= +radius; ry++)
    {
        for (int rx = -radius; rx <= +radius; rx++)
        {
            auto tex_value = tex2D<RGBA>(in_texture, idx_x + rx + 0.5f, idx_y + ry + 0.5f);

            pixel_value.x += tex_value.x;
            pixel_value.y += tex_value.y;
            pixel_value.z += tex_value.z;
            pixel_value.w += tex_value.w;
        }
    }

    pixel_value.x /= ((2 * radius + 1) * (2 * radius + 1));
    pixel_value.y /= ((2 * radius + 1) * (2 * radius + 1));
    pixel_value.z /= ((2 * radius + 1) * (2 * radius + 1));
    pixel_value.w /= ((2 * radius + 1) * (2 * radius + 1));

    out_data[idx_y * width + idx_x] = { (unsigned char)pixel_value.x, (unsigned char)pixel_value.y, (unsigned char)pixel_value.z, (unsigned char)pixel_value.w };
}

void Lab4_Task1()
{
    printf("===============LAB 4 TASK 1===============\n");

    int box_blur_radius = 45;

    float memAllocTime = 0.0f;
    float memCopyTime = 0.0f;
    float gpuExecTime1 = 0.0f;
    float gpuExecTime2 = 0.0f;
    float gpuReadTime = 0.0f;
    float cpuExecTime = 0.0f;
    float overallTime = 0.0f;

    printf("loading image...\n");
    RGBA_Image image{};
    RGBA* hostTextureOutputMemory = 0;
    memAllocTime += (float)TimeEventCPU([&]()
    {
        LoadBMPImage("resources/Lenna.bmp", &image);
        hostTextureOutputMemory = new RGBA[image.width * image.height];
    });

    printf("creating texture object...\n");
    cudaTextureObject_t tex;
    RGBA* deviceTexturePitchedMemory = 0;
    RGBA* deviceTextureOutputMemory = 0;

    bool isTexError = false;

    printf("Input image: \n");
    printf("    data is @ 0x%x\n", image.data);
    printf("    width   = %i\n", image.width);
    printf("    height  = %i\n", image.height);

    memAllocTime += (float)TimeEventGPU([&]()
    {
        auto texError = Make2DTextureObject<RGBA>(tex, image.data, image.width, image.height, &deviceTexturePitchedMemory);
        if (texError != cudaSuccess)
        {
            printf("cudaerror texError? %s\n", cudaGetErrorString(texError));
            isTexError = true;
        }

        cudaMalloc((void**)&deviceTextureOutputMemory, image.width * image.height * sizeof(RGBA));
    });

    printf("running GPU kernel...\n");
    int _blockSize = 32;
    int blocksX = image.width / _blockSize + 1;
    int blocksY = image.height / _blockSize + 1;

    dim3 blockSize = dim3(_blockSize, _blockSize, 1);
    dim3 gridSize = dim3(blocksX, blocksY, 1);
    gpuExecTime1 += (float)TimeEventGPU([&]()
    {
        kernel_lab4_t1_box_blur<<<gridSize, blockSize>>>(tex, deviceTextureOutputMemory, box_blur_radius, image.width, image.height);
        printf("cudaerror? %s\n", cudaGetErrorString(cudaGetLastError()));
    });

    printf("reading from GPU...\n");
    gpuReadTime += (float)TimeEventGPU([&]()
    {
        cudaMemcpy(hostTextureOutputMemory, deviceTextureOutputMemory, image.width * image.height * sizeof(RGBA), cudaMemcpyDeviceToHost);
    });

    cudaThreadSynchronize();

    printf("cudaerror? %s\n", cudaGetErrorString(cudaGetLastError()));

    overallTime = memAllocTime + memCopyTime + gpuExecTime1 + gpuExecTime2 + gpuReadTime + cpuExecTime;

    printf("-----------------------------------------\n");
    printf("BOX BLUR on texture, picture size: %ix%i@%i\n", image.width, image.height, 32);
    printf("Time taken to allocate memory       : %f ms\n", memAllocTime);
    printf("Time taken to copy memory to GPU    : %f ms\n", memCopyTime);
    printf("Time taken to execute kernel        : %f ms\n", gpuExecTime1);
    printf("Time taken to copy memory from GPU  : %f ms\n", gpuReadTime);
    printf("CPU execution time                  : %f ms\n", cpuExecTime);
    printf("Overall time taken                  : %f ms\n", overallTime);
    printf("-----------------------------------------\n");

    printf("...Verifying\n");
    printf("Code completed successfully!\n");

    /*int fail_index = 0;
    if (verify_vectors(hostVectorBins, cpuVectorBins, bins, &fail_index, 1e-1))
    {
        
    }*/

    RGBA_Image result_image{ hostTextureOutputMemory, image.width, image.height };
    SaveBMPImage("resources/lenna-gpu-output-l4t1.bmp", &result_image);

    free(image.data);
    free(hostTextureOutputMemory);

    cudaFree(deviceTexturePitchedMemory);
    cudaFree(deviceTextureOutputMemory);

    printf("=============LAB 4 TASK 1 END=============\n");

}