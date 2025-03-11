#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <limits>      
#include <hpc_helpers.hpp>

// Write optimized version using auto-vectorization
void softmax_auto_p(const float *input, float *output, size_t K) {

	if (K == 0) {
		return;
	}

	float max_val = input[0];

    #pragma omp simd reduction(max:max_val)
    for (size_t i = 1; i < K; ++i) {
        if (input[i] > max_val) max_val = input[i];
    }

    // Compute exponentials and store in output
    #pragma omp simd
    for (size_t i = 0; i < K; ++i) {
        output[i] = expf(input[i] - max_val);
    }

    // Sum all exponentials
    float sum = 0.0f;
    #pragma omp simd reduction(+:sum)
    for (size_t i = 0; i < K; ++i) {
        sum += output[i];
    }

    // Normalize to get probabilities
    #pragma omp simd
    for (size_t i = 0; i < K; ++i) {
        output[i] /= sum;
    }
}

void softmax_auto(const float *input, float *output, size_t K) {
    // Find the maximum to stabilize the computation of the exponential
    float max_val = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < K; ++i) {
		max_val = std::max(max_val, input[i]);
    }

    // computes all exponentials with the shift of max_val and the total sum
    float sum = 0.0f;
    for (size_t i = 0; i < K; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }

    // normalize by dividing for the total sum
    for (size_t i = 0; i < K; ++i) {
        output[i] /= sum;
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

void printResult(std::vector<float> &v, size_t K) {
	for(size_t i=0; i<K; ++i) {
		std::fprintf(stderr, "%f\n",v[i]);
	}
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

	TIMERSTART(softime_auto);
	softmax_auto_p(input.data(), output.data(), K);
	TIMERSTOP(softime_auto);
	
	// print the results on the standard output
	if (print) {
		printResult(output, K);
	}
}
