#include "common.cuh"

using namespace std::chrono;

bool Initialize(bool isZludaRuntime, int argc, char* argv[])
{
    srand(time(NULL));

    int deviceCount;
    cudaDeviceProp devProp;

    cudaGetDeviceCount(&deviceCount);
    printf("Found %d devices\n", deviceCount);

    if (deviceCount > 4 || deviceCount < 0)
    {
        if (isZludaRuntime)
        {
            std::wcout << L"Device count inadequate. Are you sure you're running with GPU present?\n\n";
        }
        else
        {
            std::wcout << L"Device count inadequate. Trying to run ZLUDA\n\n";
            RunWithZluda(argc, argv);
        }
        
        return false;
    }
    else
    {
        printf("=========================================\n");
        for (int device = 0; device < deviceCount; device++)
        {
            cudaGetDeviceProperties(&devProp, device);
            printf("Device %d\n", device);
            printf("Compute capability      : %d.%d\n", devProp.major, devProp.minor);
            printf("Name                    : %s\n", devProp.name);
            printf("Total Global Memory     : %i (MiB)\n", (int)(devProp.totalGlobalMem / (1024 * 1024)));
            printf("Shared memory per block : %i (kiB)\n", (int)(devProp.sharedMemPerBlock / (1024)));
            printf("Registers per block     : %d\n", devProp.regsPerBlock);
            printf("Warp size               : %d\n", devProp.warpSize);
            printf("Max threads per block   : %d\n", devProp.maxThreadsPerBlock);
            printf("Total constant memory   : %i (kiB)\n", (int)(devProp.totalConstMem / 1024));
            printf("=========================================\n\n");
        }
    }
    return true;
}

bool CheckArgument(int argc, char* argv[], const char* arg)
{
    std::vector<std::string> args(argv, argv + argc);

    for (size_t i = 0; i < args.size(); ++i) 
    {
        if (args[i] == std::string(arg))
        {
            return true;
        }
    }

    return false;
}

double TimeEventCPU(std::function<void()> wrapper)
{
    auto t1 = high_resolution_clock::now();

    wrapper();

    auto t2 = high_resolution_clock::now();
    auto ms_double = duration_cast<microseconds>(t2 - t1);

    double time = ms_double.count() / 1000.0;

    return time;
}

double TimeEventGPU(std::function<void()> wrapper)
{
    cudaEvent_t start, stop;
    float      Time = 0.0f;

    // создаем события начала и окончания выполнения ядра
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    wrapper();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&Time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (double)Time;
}

bool LoadPGMImage(const char* path, PGM_Image* out_image) 
{
    return false;
}

bool LoadBMPImage(const char* path, RGBA_Image* out_image)
{
    // load the file. The constructor now does most of the work
    printf("LoadBMPImage :: creating bmp object...\n");
    BitMap example_bmp(path);

    printf("LoadBMPImage :: converting to raw image...\n");
    RGBA_Image image = example_bmp.ConvertToRawImage();

    printf("LoadBMPImage :: setting output pointer...\n");
    *out_image = image;

    //printf("LoadBMPImage :: cleaning up containter...\n");
    //example_bmp.~BitMap();

    return true;
}

bool SaveBMPImage(const char* path, RGBA_Image* image, bool rewrite)
{
    std::vector<uint8_t> image_vector(image->width * image->height * 3);
    auto channel{ image_vector.data() };

    for (int y = 0; y < image->height; y++)
    {
        for (int x = 0; x < image->width; x++)
        {
            auto r = image->data[y * image->width + x].x;
            auto g = image->data[y * image->width + x].y;
            auto b = image->data[y * image->width + x].z;

            *channel++ = static_cast<uint8_t>(r);
            *channel++ = static_cast<uint8_t>(g);
            *channel++ = static_cast<uint8_t>(b);

            //printf("COLOR: (%i, %i, %i) @ x = %i; y = %i;\n", r, g, b, x, y);
        }
    }

    save_bmp(path, image->width, image->height, image_vector.data());
    return true;
}