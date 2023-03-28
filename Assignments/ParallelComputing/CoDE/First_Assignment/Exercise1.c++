// Parallel Programming exercise 1

#include<iostream>
#include<algorithm>
#include<chrono>
#include<thread>
#include<vector>
#include<random>


// Macros 
#define START_TIMER std::chrono::steady_clock::time_point _start(std::chrono::steady_clock::now());

#define ENDS_TIMER std::chrono::steady_clock::time_point _end(std::chrono::steady_clock::now());

#define PRINT_TIME std::cout << "\n\nTime in seconds:\t" << std::chrono::duration_cast<std::chrono::duration<double>>(_end - _start).count(); // in seconds, read more about std::chrono

//#define SEQUENTIAL 1

#define PARALLEL 1

//#define TEST 1

/* Assignment: write a program that transforms a vector<int> into another vector<int> 
where position i is the result of the computation of function f (given) 
on the corresponding position of the first vector (look at the lesson recording for the details). 
*/


int fibonacci(int n){
    
    if(n == 0){
        return 0;
    }
    else if(n == 1){
        return 1;
    }
    else{
        return fibonacci(n-1) + fibonacci(n-2);
    }
}

int main(int argc, char* argv[]){

    // Seed the random number generator
    srand(time(0));

    // Getting number of threads
    int nthread = (argc == 1 ? 4 : atoi(argv[1]));

    // Initialize a vector of threads, do not running them yet
    std::vector<std::thread> t(nthread);
    
    // Initialize vector size
    int size = (argc == 2 ? 300 : atoi(argv[2]));

    // Vector modified in parallel way
    std::vector<int> v1;
    // Vector modified in sequential way
    std::vector<int> v2;

    // Insert random numbers in the vector
    for (int i = 0; i < size; i++) {
        v1.push_back(rand() % 10);
    }

    v2 = v1;


    //A lambda that takes v1 and apply a function f to the elements from i to j
    auto modify_array = [] (std::vector<int>& v1, int i, int j) {

        // Function f
        auto f = [] (int x) {
            return x * 2;
        };

        // Apply another f: calculating the fibonacci sequence of the element
     
        for(int k = i; k < j; k++){
            v1[k] = f(v1[k]);
        }

        return v1;
    };

    auto modify_array_fibo = [] (std::vector<int>& v1, int i, int j) {

        // Apply another f: calculating the fibonacci sequence of the element
     
        for(int k = i; k < j; k++){
            v1[k] = fibonacci(v1[k]);
        }

        return v1;
    };




// Sequential computation

// Bash code to run the sequential computation 30 times
/*

for i in {1..30}                                                                                                    CoDE
do
  ./sequential 4 1000000 | grep 'Time in seconds:*'
done

*/

#ifdef SEQUENTIAL

    START_TIMER

    for(int i = 0; i < size; i++){
        v2[i] = fibonacci(v2[i]);
    }

    ENDS_TIMER

    std::cout << "\n\n Sequential computation has terminated" << std::endl;

    std::cout << "\n\nVector v2 after the computation:\n";

    std::cout << "\n\n#########################################" << std::endl;

    PRINT_TIME

#endif


#ifdef PARALLEL


    // code you want to time here

    START_TIMER

    for(int i = 0; i < nthread; i++){
    
    // Make sure that if you have an array of lenght n and k threads, each threads work on n/k elements
    // so first thread works on elements 0 to n/k, second thread works on elements n/k + 1 to 2n/k, and so on
    // For each thread slice the vector i to n/i elements
    
        // Worke on the v1 vector from i to n/i elements

        //std::cout << "Threads running" << i << std::endl;
        // Passing v1 by ref

        if(nthread < size){
            //std::cout << "Working on elements from " << i*(size/nthread) << " to " << (i+1)*(size/nthread) << std::endl;
            //t[i] = std::thread(modify_array, std::ref(v1), i*(size/nthread), (i+1)*(size/nthread));
            t[i] = std::thread(modify_array_fibo, std::ref(v1), i*(size/nthread), (i+1)*(size/nthread));
        }
        else{
            // This case if the number of threads is greater than the size of the vector some threads
            // will not work on any element
            t[i] = std::thread(modify_array, std::ref(v1), i, i+1);
        }
    }

    // Join all threads
    for(int i = 0; i < nthread; i++) {
        t[i].join();
    }


    ENDS_TIMER

    PRINT_TIME

    std::cout << "\n\nParallel computation has terminated" << std::endl;



//Parallel cumputation
#endif
    
    return 0;

}
