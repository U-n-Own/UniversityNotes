#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <limits>      
#include <hpc_helpers.hpp>
#include <avx_mathfun.h>

void softmax_avx(const float *input, float *output, size_t K) {


    float max_val = -std::numeric_limits<float>::infinity();
    __m256 max_val_v = _mm256_set1_ps(max_val);
	__m256 input_v;
    
    // Process by blocks of 8 using vectorization
    size_t VEC_SIZE = K / 8 * 8; 
	

	// Find the maximum to stabilize the computation of the exponential
	for (size_t i = 0; i < VEC_SIZE; i+=8) {
		input_v = _mm256_loadu_ps(input+i);
		max_val_v = _mm256_max_ps(max_val_v, input_v);
	}

	float max_val_arr[8];
	_mm256_storeu_ps(max_val_arr, max_val_v);

    // get current max value
    float current_max = *std::max_element(max_val_arr, max_val_arr + 8);
	
    for (size_t i = VEC_SIZE; i < K; ++i) {
		max_val = std::max(current_max, input[i]);
	}

	// Compute exponential with the function in avx_mathfun.h: v8sf exp256_ps(v8sf x)
	__m256 max_val_v2 = _mm256_set1_ps(max_val);
	__m256 exp_v;

	for (size_t i = 0; i < VEC_SIZE; i+=8) {
		input_v = _mm256_loadu_ps(input+i);
		exp_v = exp256_ps(_mm256_sub_ps(input_v, max_val_v2));
		_mm256_storeu_ps(output+i, exp_v);
	}

    // Process remaining elements sequentially
    for (size_t i = VEC_SIZE; i < K; i++) {
        output[i] = std::exp(input[i] - max_val);
    }

	// Sum all exponentials
	float sum = 0.0f;
	__m256 sum_v = _mm256_setzero_ps();
	__m256 output_v;

	for (size_t i = 0; i < VEC_SIZE; i+=8) {
		output_v = _mm256_loadu_ps(output+i);
		sum_v = _mm256_add_ps(sum_v, output_v);
	}

    // Sum to scalar
	float sum_arr[8];
	_mm256_storeu_ps(sum_arr, sum_v);
	for (size_t i = 0; i < 8; ++i) {
		sum += sum_arr[i];
	}

    // Sum remaining elements sequentially
    for (size_t i = VEC_SIZE; i < K; i++) {
        sum += output[i];
    }

	// Normalize to get probabilities
	__m256 sum_v2 = _mm256_set1_ps(sum);
	for (size_t i = 0; i < VEC_SIZE; i+=8) {
		output_v = _mm256_loadu_ps(output+i);
		output_v = _mm256_div_ps(output_v, sum_v2);
		_mm256_storeu_ps(output+i, output_v);
	}

    // Normalize remaining elements sequentially
    for (size_t i = VEC_SIZE; i < K; i++) {
        output[i] /= sum;
    }

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
		std::printf("use: %s K [1]\n", argv[0]);
		return 0;		
	}
	size_t K=0;
	if (argc >= 2) {
		K = std::stol(argv[1]);
	}
	bool print=false;
	if (argc == 3) {
		print=true;
	}	
	std::vector<float> input=generate_random_input(K);
	std::vector<float> output(K);


	TIMERSTART(softime_avx);
		softmax_avx(input.data(), output.data(), K);
	TIMERSTOP(softime_avx);
	
	// print the results on the standard output
	if (print) {
		printResult(output, K);
	}
}

