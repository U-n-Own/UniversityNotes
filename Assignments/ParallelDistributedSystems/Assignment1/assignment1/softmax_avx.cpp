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


void softmax_avx_op2(const float *input, float *output, size_t K) {

    size_t vec_size = K / 8 * 8;  // Process only multiples of 8 using vectorization
    __m256 max_val_vec = _mm256_set1_ps(-INFINITY);
    
    // Find the maximum value in blocks of 8
    for (size_t i = 0; i < vec_size; i += 8) {
        __m256 in_vec = _mm256_loadu_ps(&input[i]);
        max_val_vec = _mm256_max_ps(max_val_vec, in_vec);
    }

    // Reduce the maximum value to a single scalar
    float max_val[8];
    _mm256_storeu_ps(max_val, max_val_vec);
    float final_max = *std::max_element(max_val, max_val + 8);

    // Process remaining elements sequentially
    for (size_t i = vec_size; i < K; i++) {
        final_max = std::max(final_max, input[i]);
    }

    __m256 final_max_vec = _mm256_set1_ps(final_max);
    __m256 sum_vec = _mm256_setzero_ps();

    // Compute exp(x - max) in blocks of 8
    for (size_t i = 0; i < vec_size; i += 8) {
        __m256 in_vec = _mm256_loadu_ps(&input[i]);
        __m256 exp_vec = exp256_ps(_mm256_sub_ps(in_vec, final_max_vec));
        _mm256_storeu_ps(&output[i], exp_vec);
        sum_vec = _mm256_add_ps(sum_vec, exp_vec);
    }

    // Reduce the sum to a single scalar
    float sum[8];
    _mm256_storeu_ps(sum, sum_vec);
    float final_sum = 0.0f;
    for (int i = 0; i < 8; ++i) {
        final_sum += sum[i];
    }

    // Compute exp(x - max) sequentially for remaining elements
    for (size_t i = vec_size; i < K; i++) {
        output[i] = std::exp(input[i] - final_max);
        final_sum += output[i];
    }

    __m256 final_sum_vec = _mm256_set1_ps(final_sum);

    // Normalize the values in blocks of 8
    for (size_t i = 0; i < vec_size; i += 8) {
        __m256 out_vec = _mm256_loadu_ps(&output[i]);
        out_vec = _mm256_div_ps(out_vec, final_sum_vec);
        _mm256_storeu_ps(&output[i], out_vec);
    }

    // Normalize remaining elements sequentially
    for (size_t i = vec_size; i < K; i++) {
        output[i] /= final_sum;
    }
}

// softmax implementation this support alignment and scalar handling of remaining elements
void softmax_avx_optimized(float* input, float* output, size_t K) {
    // Find the maximum to stabilize the computation of the exponential

    const size_t VEC_SIZE = 8;

    float max_val = -std::numeric_limits<float>::infinity();

    __m256 max_val_v = _mm256_set1_ps(max_val);
    __m256 input_v;

    size_t i = 0;
    for (; i < K - VEC_SIZE + 1; i += VEC_SIZE) {
        input_v = _mm256_loadu_ps(input + i);
        max_val_v = _mm256_max_ps(max_val_v, input_v);
    }

    float max_val_arr[VEC_SIZE];
    _mm256_storeu_ps(max_val_arr, max_val_v);

    for (size_t j = 0; j < VEC_SIZE; ++j) {
        max_val = std::max(max_val, max_val_arr[j]);
    }

    // computes all exponentials with the shift of max_val and the total sum 
    float sum = 0.0f;

    __m256 sum_v = _mm256_setzero_ps();
    __m256 exp_v;

    for (i = 0; i < K - VEC_SIZE + 1; i += VEC_SIZE) {
        input_v = _mm256_loadu_ps(input + i);
        exp_v = exp256_ps(_mm256_sub_ps(input_v, _mm256_set1_ps(max_val)));
        _mm256_storeu_ps(output + i, exp_v);
        sum_v = _mm256_add_ps(sum_v, exp_v);
    }

    float sum_arr[VEC_SIZE];
    _mm256_storeu_ps(sum_arr, sum_v);
    
    for (size_t j = 0; j < VEC_SIZE; ++j) {
        sum += sum_arr[j];
    }

    // scalar handling of remaining elements
    for (; i < K; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }

    // normalize by dividing for the total sum
    __m256 sum_v2 = _mm256_set1_ps(sum);
    __m256 output_v;

    for (i = 0; i < K - VEC_SIZE + 1; i += VEC_SIZE) {
        output_v = _mm256_loadu_ps(output + i);
        output_v = _mm256_div_ps(output_v, sum_v2);
        _mm256_storeu_ps(output + i, output_v);
    }

    // scalar handling of remaining elements
    for (; i < K; ++i) {
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
		std::printf("use: %s K [print] [optimized]\n", argv[0]);
		return 0;		
	}
	size_t K = std::stol(argv[1]);
	bool print = (argc >= 3) ? std::string(argv[2]) == "true" : false;
	bool optimized = (argc >= 4) ? std::string(argv[3]) == "true" : false;

	std::vector<float> input = generate_random_input(K);
	std::vector<float> output(K);


	TIMERSTART(softime_avx);
	if (optimized) {
		printf("Optimized version avx\n");
        softmax_avx_op2(input.data(), output.data(), K);
	} else {
		printf("Standard avx\n");
		softmax_avx(input.data(), output.data(), K);
	}
	TIMERSTOP(softime_avx);
	
	// print the results on the standard output
	if (print) {
		printResult(output, K);
	}
}

