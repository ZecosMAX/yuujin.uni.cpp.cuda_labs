#pragma once

void fillRandomNaturalVector(float* vector, int size);
void fillRandomRealVector(float* vector, int size);
bool verify_vectors(float* a, float* b, int size, int* fail_index, float eps = 1e-6);
void print_vector(float* vec, int size, int offset);
void graph_vector(float* ys, int size, int cliHeight);