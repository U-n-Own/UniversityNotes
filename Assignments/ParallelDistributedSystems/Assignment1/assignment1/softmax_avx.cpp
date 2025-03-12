#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <limits>      
#include <hpc_helpers.hpp>
#include <avx_mathfun.h>

void softmax_avx(const float *input, float *output, size_t K) {

	// Find the maximum to stabilize the computation of the exponential
	float max_val = -std::numeric_limits<float>::infinity();

	__m256 max_val_v = _mm256_set1_ps(max_val);
	__m256 input_v;

	for (size_t i = 0; i < K; i+=8) {
		input_v = _mm256_loadu_ps(input+i);
		max_val_v = _mm256_max_ps(max_val_v, input_v);
	}

	float max_val_arr[8];
	_mm256_storeu_ps(max_val_arr, max_val_v);
	for (size_t i = 0; i < 8; ++i) {
		max_val = std::max(max_val, max_val_arr[i]);
	}

	// Compute exponential with the function in avx_mathfun.h: v8sf exp256_ps(v8sf x)
	__m256 max_val_v2 = _mm256_set1_ps(max_val);
	__m256 exp_v;

	for (size_t i = 0; i < K; i+=8) {
		input_v = _mm256_loadu_ps(input+i);
		exp_v = exp256_ps(_mm256_sub_ps(input_v, max_val_v2));
		_mm256_storeu_ps(output+i, exp_v);
	}

	// Sum all exponentials
	float sum = 0.0f;
	__m256 sum_v = _mm256_setzero_ps();
	__m256 output_v;

	for (size_t i = 0; i < K; i+=8) {
		output_v = _mm256_loadu_ps(output+i);
		sum_v = _mm256_add_ps(sum_v, output_v);
	}

	float sum_arr[8];
	_mm256_storeu_ps(sum_arr, sum_v);
	for (size_t i = 0; i < 8; ++i) {
		sum += sum_arr[i];
	}

	// Normalize to get probabilities
	__m256 sum_v2 = _mm256_set1_ps(sum);
	for (size_t i = 0; i < K; i+=8) {
		output_v = _mm256_loadu_ps(output+i);
		output_v = _mm256_div_ps(output_v, sum_v2);
		_mm256_storeu_ps(output+i, output_v);
	}
}

// Version that takes into account alignment, scalar remainder handling, and horizontal vector reductions
void softmax_avx_optimized(const float* __restrict input, float* __restrict output, size_t K){
    if (K == 0) return;

    // Align input/output to 32-byte boundaries (caller's responsibility)
    constexpr size_t VEC_SIZE = 8; // AVX processes 8 floats per vector

    // Step 1: Find maximum value (vectorized + remainder handling)
    float max_val = -std::numeric_limits<float>::infinity();
    __m256 max_v = _mm256_set1_ps(max_val);
    size_t i = 0;

    // Vectorized max (aligned)
    for (; i + VEC_SIZE <= K; i += VEC_SIZE) {
        __m256 data = _mm256_loadu_ps(input + i);
        max_v = _mm256_max_ps(max_v, data);
    }

    // Horizontal max reduction
    alignas(32) float max_buffer[VEC_SIZE];
    _mm256_storeu_ps(max_buffer, max_v);
    for (size_t j = 0; j < VEC_SIZE; ++j)
        max_val = std::max(max_val, max_buffer[j]);

    // Scalar remainder
    for (; i < K; ++i)
        max_val = std::max(max_val, input[i]);

    // Step 2: Compute exponentials (vectorized + remainder)
    __m256 max_vec = _mm256_set1_ps(max_val);
    i = 0;
    for (; i + VEC_SIZE <= K; i += VEC_SIZE) {
        __m256 data = _mm256_loadu_ps(input + i);
        __m256 exp_data = exp256_ps(_mm256_sub_ps(data, max_vec));
        _mm256_storeu_ps(output + i, exp_data);
    }

    // Scalar remainder
    for (; i < K; ++i)
        output[i] = expf(input[i] - max_val);

    // Step 3: Sum exponentials (vectorized + remainder)
    float sum = 0.0f;
    __m256 sum_v = _mm256_setzero_ps();
    i = 0;
    for (; i + VEC_SIZE <= K; i += VEC_SIZE) {
        __m256 data = _mm256_loadu_ps(output + i);
        sum_v = _mm256_add_ps(sum_v, data);
    }

    // Horizontal sum reduction
    alignas(32) float sum_buffer[VEC_SIZE];
    _mm256_storeu_ps(sum_buffer, sum_v);
    for (size_t j = 0; j < VEC_SIZE; ++j)
        sum += sum_buffer[j];

    // Scalar remainder
    for (; i < K; ++i)
        sum += output[i];

    // Step 4: Normalize (vectorized + remainder)
    __m256 inv_sum = _mm256_set1_ps(1.0f / sum);
    i = 0;
    for (; i + VEC_SIZE <= K; i += VEC_SIZE) {
        __m256 data = _mm256_loadu_ps(output + i);
        data = _mm256_mul_ps(data, inv_sum);
        _mm256_storeu_ps(output + i, data);
    }

    // Scalar remainder
    for (; i < K; ++i)
        output[i] /= sum;
}

void printResult(std::vector<float> &v, size_t K) {
	for(size_t i=0; i<K; ++i) {
		std::fprintf(stderr, "%f\n",v[i]);
	}
}

std::vector<float> generate_random_input(size_t K, float min = -1.0f, float max = 1.0f) {
    std::vector<float> input(K);
    //std::random_device rd;
    //std::mt19937 gen(rd());
	std::mt19937 gen(5489); // fixed seed for reproducible results
    std::uniform_real_distribution<float> dis(min, max);
    for (size_t i = 0; i < K; ++i) {
        input[i] = dis(gen);
    }
    return input;
}

int main(int argc, char *argv[]) {
	if (argc == 1) {
		std::printf("use: %s K [print] [optimized]\n", argv[0]);
		return 0;		
	}
	size_t K = std::stol(argv[1]);
	bool print = (argc >= 3) ? std::string(argv[2]) == "true" : false;
	bool optimized = (argc >= 4) ? std::string(argv[3]) == "true" : false;

	std::vector<float> input = generate_random_input(K);
	std::vector<float> output(K);

	/*
	float* input_aligned = static_cast<float*>(aligned_alloc(32, K * sizeof(float)));
	float* output_aligned = static_cast<float*>(aligned_alloc(32, K * sizeof(float)));
	*/

	float* input_aligned = (float*)_mm_malloc(K * sizeof(float), 32);
	float* output_aligned = (float*)_mm_malloc(K * sizeof(float), 32);
	
	TIMERSTART(softime_avx);
	if (optimized) {
		printf("Optimized version avx\n");
		softmax_avx_optimized(input_aligned, output_aligned, K);
	} else {
		printf("Standard avx\n");
		softmax_avx(input.data(), output.data(), K);
	}
	TIMERSTOP(softime_avx);
	

	_mm_free(input_aligned);
	_mm_free(output_aligned);
	// why this free works as well? Valgrind don't detect leaks
	//free(input_aligned);
	//free(output_aligned);
	
	// print the results on the standard output
	if (print) {
		printResult(output, K);
	}
}

