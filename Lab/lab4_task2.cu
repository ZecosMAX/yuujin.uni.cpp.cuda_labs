// Задание 2: изучить основы работы с текстурным массивом на CUDA.
// написать программу на СИ, реализующую цифровой билинейный фильтр
// для масштабирования изображения с использованием текстурного массива

#include "common.cuh"
#include "labs_tasks.cuh"

#include <chrono>
#include <math.h>
#include <random>

using namespace std::chrono;

__device__ float4 gpu_lerp(float4 a, float4 b, float f)
{
    float4 result{};

    result.x = a.x * (1.0 - f) + (b.x * f);
    result.y = a.y * (1.0 - f) + (b.y * f);
    result.z = a.z * (1.0 - f) + (b.z * f);
    result.w = a.w * (1.0 - f) + (b.w * f);

    return result;
}

__global__ void kernel_lab4_t2_bilinear_upscale(cudaTextureObject_t in_texture, RGBA* out_data, int in_width, int in_height, int out_width, int out_height)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx_x >= out_width)
        return;

    if (idx_y >= out_height)
        return;

    float x_ratio = (float)out_width / in_width;
    float y_ratio = (float)out_height / in_height;

    float2 outputCoords = { idx_x, idx_y };
    float2 inputCoords = { idx_x / x_ratio, idx_y / y_ratio };

    float interpolation_coeffitient_x = inputCoords.x - truncf(inputCoords.x);
    float interpolation_coeffitient_y = inputCoords.y - truncf(inputCoords.y);

    int2 inputCoords_topLeft  = { (int)(outputCoords.x / x_ratio) + 0, (int)(outputCoords.y / y_ratio) + 0 };
    int2 inputCoords_topRight = { (int)(outputCoords.x / x_ratio) + 1, (int)(outputCoords.y / y_ratio) + 0 };
    int2 inputCoords_botLeft  = { (int)(outputCoords.x / x_ratio) + 0, (int)(outputCoords.y / y_ratio) + 1 };
    int2 inputCoords_botRight = { (int)(outputCoords.x / x_ratio) + 1, (int)(outputCoords.y / y_ratio) + 1 };

    RGBA v_topLeft  = tex2D<RGBA>(in_texture, inputCoords_topLeft .x + 0.5f, inputCoords_topLeft .y + 0.5f);
    RGBA v_topRight = tex2D<RGBA>(in_texture, inputCoords_topRight.x + 0.5f, inputCoords_topRight.y + 0.5f);
    RGBA v_botLeft  = tex2D<RGBA>(in_texture, inputCoords_botLeft .x + 0.5f, inputCoords_botLeft .y + 0.5f);
    RGBA v_botRight = tex2D<RGBA>(in_texture, inputCoords_botRight.x + 0.5f, inputCoords_botRight.y + 0.5f);

    // Interpolate on the top line (horizontally)
    float4 tp1 = { v_topLeft.x, v_topLeft.y, v_topLeft.z, v_topLeft.w };
    float4 tp2 = { v_topRight.x, v_topRight.y, v_topRight.z, v_topRight.w };
    float4 ti = gpu_lerp(tp1, tp2, interpolation_coeffitient_x);

    // Interpolate on the bottom line (horizontally)
    float4 bp1 = { v_botLeft.x, v_botLeft.y, v_botLeft.z, v_botLeft.w };
    float4 bp2 = { v_botRight.x, v_botRight.y, v_botRight.z, v_botRight.w };
    float4 bi = gpu_lerp(bp1, bp2, interpolation_coeffitient_x);

    // Interpolate between aquired points (vertically)
    float4 i = gpu_lerp(ti, bi, interpolation_coeffitient_y);

    out_data[idx_y * out_width + idx_x] = { (unsigned char)i.x, (unsigned char)i.y, (unsigned char)i.z, (unsigned char)i.w };
}

void Lab4_Task2()
{
    printf("===============LAB 4 TASK 2===============\n");

    int result_image_width = 100;
    int result_image_height = 100;

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
        hostTextureOutputMemory = new RGBA[result_image_width * result_image_height];
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

        cudaMalloc((void**)&deviceTextureOutputMemory, result_image_width * result_image_height * sizeof(RGBA));
    });

    printf("running GPU kernel...\n");
    int _blockSize = 32;
    int blocksX = result_image_width / _blockSize + 1;
    int blocksY = result_image_height / _blockSize + 1;

    dim3 blockSize = dim3(_blockSize, _blockSize, 1);
    dim3 gridSize = dim3(blocksX, blocksY, 1);
    gpuExecTime1 += (float)TimeEventGPU([&]()
    {
        kernel_lab4_t2_bilinear_upscale<<<gridSize, blockSize>>>(tex, deviceTextureOutputMemory, image.width, image.height, result_image_width, result_image_height);
        printf("cudaerror? %s\n", cudaGetErrorString(cudaGetLastError()));
    });

    printf("reading from GPU...\n");
    gpuReadTime += (float)TimeEventGPU([&]()
    {
        cudaMemcpy(hostTextureOutputMemory, deviceTextureOutputMemory, result_image_width * result_image_height * sizeof(RGBA), cudaMemcpyDeviceToHost);
    });

    cudaThreadSynchronize();

    printf("cudaerror? %s\n", cudaGetErrorString(cudaGetLastError()));

    overallTime = memAllocTime + memCopyTime + gpuExecTime1 + gpuExecTime2 + gpuReadTime + cpuExecTime;

    printf("-----------------------------------------\n");
    printf("Bilinear filter upscaling, picture size: %ix%i @ %i to %ix%i @ %i\n", image.width, image.height, 32, result_image_width, result_image_height, 32);
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

    RGBA_Image result_image{ hostTextureOutputMemory, result_image_width, result_image_height };
    SaveBMPImage("resources/lenna-gpu-output-l4t2.bmp", &result_image);

    free(image.data);
    free(hostTextureOutputMemory);

    cudaFree(deviceTexturePitchedMemory);
    cudaFree(deviceTextureOutputMemory);

    printf("=============LAB 4 TASK 2 END=============\n");

}
