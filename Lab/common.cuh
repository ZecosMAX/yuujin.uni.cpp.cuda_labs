#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "vector_utils.hpp"
#include "windows_utils.hpp"
#include "zluda_utils.hpp"

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