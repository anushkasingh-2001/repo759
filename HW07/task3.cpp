#include <omp.h>

#include <iostream>
#include <vector>

long long factorial(int n) {
    long long ans = 1;
    for (int i = 2; i <= n; ++i) {
        ans *= i;
    }
    return ans;
}

int main() {
    omp_set_num_threads(4);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        #pragma omp single
        {
            std::cout << "Number of threads: " << omp_get_num_threads() << "\n";
        }

        #pragma omp critical
        {
            std::cout << "I am thread No. " << tid << "\n";
        }

        #pragma omp for
        for (int i = 1; i <= 8; ++i) {
            long long f = factorial(i);
            #pragma omp critical
            {
                std::cout << i << "!=" << f << "\n";
            }
        }
    }

    return 0;
}