#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cstdlib>

using namespace std;

int f(int x) { return x * x; }

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <num_threads> <vector_size>" << endl;
        return 1;
    }

    int num_threads = atoi(argv[1]);
    int n = atoi(argv[2]);

    // Initialize vector
    vector<int> v(n);
    for (int i = 0; i < n; ++i) v[i] = i + 1;

    vector<int> w_parallel(n), w_sequential(n);

    // Warm-up runs (parallel)
    for (int warm = 0; warm < 3; ++warm) {
        vector<thread> threads;
        int chunk = (n + num_threads - 1) / num_threads;
        for (int i = 0; i < num_threads; ++i) {
            int start = i * chunk;
            int end = min(start + chunk, n);
            threads.emplace_back([start, end, &v, &w_parallel]() {
                for (int j = start; j < end; ++j) w_parallel[j] = f(v[j]);
            });
        }
        for (auto& t : threads) t.join();
    }

    // Time parallel (average of 50 trials)
    const int trials = 50;

    long total_parallel = 0;
    for (int trial = 0; trial < trials; ++trial) {
        auto start = chrono::steady_clock::now();
        vector<thread> threads;
        int chunk = (n + num_threads - 1) / num_threads;
        for (int i = 0; i < num_threads; ++i) {
            int start = i * chunk;
            int end = min(start + chunk, n);
            threads.emplace_back([start, end, &v, &w_parallel]() {
                for (int j = start; j < end; ++j) w_parallel[j] = f(v[j]);
            });
        }
        for (auto& t : threads) t.join();
        auto end = chrono::steady_clock::now();
        total_parallel += chrono::duration_cast<chrono::microseconds>(end - start).count();
    }

    // Time sequential (average of 50 trials)
    long total_sequential = 0;
    for (int trial = 0; trial < trials; ++trial) {
        auto start = chrono::steady_clock::now();
        for (int i = 0; i < n; ++i) w_sequential[i] = f(v[i]);
        auto end = chrono::steady_clock::now();
        total_sequential += chrono::duration_cast<chrono::microseconds>(end - start).count();
    }

    // Validate & output
    if (w_parallel != w_sequential) cerr << "Error: Mismatch!\n";
    cout << "Avg Parallel: " << total_parallel / trials << " μs\n";
    cout << "Avg Sequential: " << total_sequential / trials << " μs\n";
    cout << "Avg Speedup: " << (static_cast<double>(total_sequential) / total_parallel) << endl;

    return 0;
}