#include "count.cuh"

#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>

void count(const thrust::device_vector<int>& d_in,
           thrust::device_vector<int>& values,
           thrust::device_vector<int>& counts) {
    if (d_in.empty()) {
        values.clear();
        counts.clear();
        return;
    }

    thrust::device_vector<int> sorted = d_in;
    thrust::sort(sorted.begin(), sorted.end());

    values.resize(sorted.size());
    counts.resize(sorted.size());

    auto new_end = thrust::reduce_by_key(
        sorted.begin(),
        sorted.end(),
        thrust::make_constant_iterator(1),
        values.begin(),
        counts.begin()
    );

    size_t num_unique = new_end.first - values.begin();
    values.resize(num_unique);
    counts.resize(num_unique);
}