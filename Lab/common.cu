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
            printf("Total Global Memory     : %u (MiB)\n", devProp.totalGlobalMem / (1024 * 1024));
            printf("Shared memory per block : %d (kiB)\n", devProp.sharedMemPerBlock / (1024));
            printf("Registers per block     : %d\n", devProp.regsPerBlock);
            printf("Warp size               : %d\n", devProp.warpSize);
            printf("Max threads per block   : %d\n", devProp.maxThreadsPerBlock);
            printf("Total constant memory   : %d\n", devProp.totalConstMem);
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