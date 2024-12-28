#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

typedef unsigned char PGM;
typedef uchar4 RGBA;

struct PGM_Image_s
{
	PGM* data;
	int width;
	int height;
};
typedef PGM_Image_s PGM_Image;


struct RGBA_Image_s
{
	RGBA* data;
	int width;
	int height;
};
typedef RGBA_Image_s RGBA_Image;