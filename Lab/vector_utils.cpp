#include "vector_utils.hpp"

#include <iostream>
#include <vector>

void fillRandomNaturalVector(float* vector, int size)
{
    for (int i = 0; i < size; i++)
    {
        vector[i] = (float)(rand() % 99 + 1);
    }
}

void fillRandomRealVector(float* vector, int size)
{
    for (int i = 0; i < size; i++)
    {
        vector[i] = (float)(rand() / RAND_MAX);
    }
}

bool verify_vectors(float* a, float* b, int size, int* fail_index)
{
    if (a == nullptr || b == nullptr)
    {
        *fail_index = -1;
        return false;
    }

    printf("Verifying start, size: %d\n", size);
    printf("a: 0x%X, b: 0x%X\n", a, b);

    for (int i = 0; i < size; i++)
    {
        if (a[i] != b[i])
        {
            *fail_index = i;
            return false;
        }
    }

    printf("Verifying end\n");
    return true;
}

void print_vector(float* vec, int size, int offset)
{
    if (offset < 0)
        offset = 0;

    printf("{ ");
    for (int i = offset; i < offset + size; i++)
    {
        printf("%f", vec[i]);
        if (i != size - 1)
            printf(", ");
    }
    printf(" }\n");
}

void graph_vector(float* ys, int size, int cliHeight)
{
    if (cliHeight < 4)
        cliHeight = 4;

    float minY = +INFINITY;
    float maxY = -INFINITY;

    for (size_t i = 0; i < size; i++)
    {
        float item = ys[i];

        if (item < minY)
            minY = item;

        if (item > maxY)
            maxY = item;
    }

    float step = (maxY - minY) / (cliHeight - 2);

    for (int y = (cliHeight - 1); y >= 0; y--)
    {
        if (y == 0)
        {
            std::cout << '+';
            for (int x = 0; x < size; x++)
            {
                std::cout << '-';
            }
            std::cout << std::endl;
        }
        else
        {
            std::cout << '|';

            for (int x = 0; x < size; x++)
            {
                float item = ys[x];

                if (item > ((float)(y - 1)) * step)
                {
                    std::wcout << L'X';
                }
                else
                {
                    std::wcout << L' ';
                }
            }

            std::wcout << std::endl;

        }
    }
}