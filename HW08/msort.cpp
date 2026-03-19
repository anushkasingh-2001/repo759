#include "msort.h"

#include <algorithm>
#include <vector>

static void merge_ranges(int* arr, std::size_t left,
                         std::size_t mid, std::size_t right) {
    std::vector<int> temp;
    temp.reserve(right - left);

    std::size_t i = left;
    std::size_t j = mid;

    while (i < mid && j < right) {
        if (arr[i] <= arr[j]) temp.push_back(arr[i++]);
        else temp.push_back(arr[j++]);
    }

    while (i < mid) temp.push_back(arr[i++]);
    while (j < right) temp.push_back(arr[j++]);

    for (std::size_t k = 0; k < temp.size(); ++k) {
        arr[left + k] = temp[k];
    }
}

static void msort_rec(int* arr, std::size_t left,
                      std::size_t right, std::size_t threshold) {
    const std::size_t len = right - left;
    if (len <= 1) return;

    if (len <= threshold) {
        std::sort(arr + left, arr + right);
        return;
    }

    const std::size_t mid = left + len / 2;

    #pragma omp task shared(arr)
    msort_rec(arr, left, mid, threshold);

    #pragma omp task shared(arr)
    msort_rec(arr, mid, right, threshold);

    #pragma omp taskwait
    merge_ranges(arr, left, mid, right);
}

void msort(int* arr, const std::size_t n, const std::size_t threshold) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            msort_rec(arr, 0, n, threshold);
        }
    }
}