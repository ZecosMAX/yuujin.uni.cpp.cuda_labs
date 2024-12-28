#pragma once

#ifndef __CUDACC__  
	#define __CUDACC__
#endif

#define SAVE_BMP_IMPLEMENT

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "types.hpp"
#include "vector_utils.hpp"
#include "windows_utils.hpp"
#include "zluda_utils.hpp"
#include "bitmap.hpp"
#include "save_bitmap.hpp"


#include <iostream>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <functional>
#include <chrono>
#include <vector>

bool Initialize(bool isZludaRuntime, int argc, char* argv[]);

bool CheckArgument(int argc, char* argv[], const char* arg);

double TimeEventCPU(std::function<void()> wrapper);

double TimeEventGPU(std::function<void()> wrapper);

bool LoadPGMImage(const char* path, PGM_Image* out_image); 
bool LoadBMPImage(const char* path, RGBA_Image* out_image); 
bool LoadPNGImage(const char* path, RGBA_Image* out_image); 
bool LoadJPGImage(const char* path, RGBA_Image* out_image);

bool SaveBMPImage(const char* path, PGM_Image* image, bool rewrite = true);
bool SaveBMPImage(const char* path, RGBA_Image* image, bool rewrite = true);


// helper function to create 2D texture object
// ref: https://stackoverflow.com/questions/54098747/cuda-how-to-create-2d-texture-object
template<class T>
cudaError_t Make2DTextureObject(cudaTextureObject_t& texObject, T* host_data, int num_cols, int num_rows, T** out_device_ptr)
{
    cudaDeviceProp prop;
    auto propError = cudaGetDeviceProperties(&prop, 0);

    if (propError != cudaSuccess)
    {
        printf("Make2DTextureObject propError! %s\n", cudaGetErrorString(propError));
        return propError;
    }

    //num_cols = prop.texturePitchAlignment * num_cols;

    const int ts = num_cols * num_rows;
    const int ds = ts * sizeof(T);

    T* dataDev = 0;
    size_t pitch;

    auto mallocError = cudaMallocPitch(
        (void**)&dataDev, 
        &pitch, 
        num_cols * sizeof(T), 
        num_rows);

    if (mallocError != cudaSuccess)
    {
        printf("Make2DTextureObject mallocError! %s\n", cudaGetErrorString(mallocError));
        return mallocError;
    }

    printf("cudaMemcpy2D:\n");
    printf("    dst (device) = 0x%x\n", dataDev);
    printf("    sizeof(T)    = %i\n", sizeof(T));
    printf("    dpitch       = %i\n", pitch);
    printf("    src (host)   = 0x%x\n", host_data);
    printf("    spitch       = %i\n", num_cols * sizeof(T));
    printf("    width        = %i\n", num_cols);
    printf("    height       = %i\n", num_rows);

    auto memcpyError = cudaMemcpy2D(dataDev, pitch, host_data, num_cols * sizeof(T), num_cols * sizeof(T), num_rows, cudaMemcpyHostToDevice);
    //auto memcpyError = cudaMemcpy(dataDev, host_data, num_rows * num_cols * sizeof(T), cudaMemcpyHostToDevice);

    if (memcpyError != cudaSuccess)
    {
        printf("Make2DTextureObject memcpyError! %s\n", cudaGetErrorString(memcpyError));
        return memcpyError;
    }

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = dataDev;
    resDesc.res.pitch2D.width = num_cols;
    resDesc.res.pitch2D.height = num_rows;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<T>();
    resDesc.res.pitch2D.pitchInBytes = pitch;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));

    auto createTextureError = cudaCreateTextureObject(&texObject, &resDesc, &texDesc, NULL);

    if (createTextureError != cudaSuccess)
    {
        printf("Make2DTextureObject createTextureError! %s\n", cudaGetErrorString(createTextureError));
        return createTextureError;
    }

    *out_device_ptr = dataDev;

    return cudaSuccess;
}