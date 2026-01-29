#include <cstdio>
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " N\n";
        return 1;
    }

    char* end = nullptr;
    long n_long = std::strtol(argv[1], &end, 10);
    if (end == argv[1] || *end != '\0' || n_long < 0) {
        std::cerr << "N must be a non-negative integer.\n";
        return 1;
    }

    int N = static_cast<int>(n_long);

    // Ascending with printf
    for (int i = 0; i <= N; i++) {
        std::printf("%d%s", i, (i == N) ? "\n" : " ");
    }

    // Descending with cout
    for (int i = N; i >= 0; i--) {
        std::cout << i << (i == 0 ? '\n' : ' ');
    }

    return 0;
}
