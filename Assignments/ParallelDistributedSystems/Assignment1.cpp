#include <iostream>
#include <omp.h>
#include <vector>

// Function to simulate some work
int work(int x) {
    return x * x;
}

int main() {
    const int num_tasks = 100;
    std::vector<int> tasks(num_tasks);
    std::vector<int> results(num_tasks);

    // Initialize tasks
    for (int i = 0; i < num_tasks; ++i) {
        tasks[i] = i;
    }

    // Parallelize using OpenMP farm pattern
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 0; i < num_tasks; ++i) {
                #pragma omp task firstprivate(i)
                {
                    // Each task performs the work function
                    results[i] = work(tasks[i]);
                }
            }
        }
    }

    // Print results
    for (int i = 0; i < num_tasks; ++i) {
        std::cout << "Task " << i << " result: " << results[i] << std::endl;
    }

    return 0;
}