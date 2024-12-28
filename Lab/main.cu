#define SAVE_BMP_IMPLEMENT
#include "common.cuh"
#include "labs_tasks.cuh"


int main(int argc, char* argv[])
{
    bool isZludaRuntime = CheckArgument(argc, argv, "zluda");
    if (!Initialize(isZludaRuntime, argc, argv))
        return -1;
    
    RunAllTasks();

    return 0;
}

void RunAllTasks()
{
    //Lab2_Task1();
    //Lab2_Task2();
    //Lab2_Task3();

    Lab3_Task1();
    Lab3_Task2();

    Lab4_Task1();
}
