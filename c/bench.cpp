#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <bitset>
#include <tuple>

#include <bops.h>

uint64_t rand64() {
    uint64_t x = rand() | ((uint64_t)rand())<<32;

    if (rand()/(float)RAND_MAX > 0.5) {
        x = x | (((uint64_t)1)<<63);
    }

    return x;
}

void showMemory(size_t* dims,
                size_t num_dims,
                size_t B) {

    auto N = dims[0];

    auto input_size_1 = 1*N/64*sizeof(uint64_t);
    auto input_size = B*N/64*sizeof(uint64_t);
    auto weights_size = mlp_weights_size(dims, num_dims);
    auto scratch_size_1 = mlp_scratchs_size(dims, num_dims, 1);
    auto scratch_size = mlp_scratchs_size(dims, num_dims, B);

    std::cout << "MLP with parameters: [";
    for (size_t i=0; i<num_dims; ++i) {
        std::cout << dims[i];

        if (i<num_dims-1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

    std::cout << "Size on disk: "
              << weights_size/(1024.*1024) << " MB" << std::endl;
    std::cout << "Inference total memory (B=1): "
              << (input_size_1 + weights_size + scratch_size_1)/(1024.*1024) << " MB" << std::endl;
    std::cout << "Train memory "
              << (input_size + weights_size + scratch_size)/(1024.*1024) << " MB" << std::endl;
}

/**
 * @brief take the given duration in nano seconds and returned the duration in a human readable unit.
 * @returns a tuple with a human readable duration and its unit as a string
 */
std::tuple<float, std::string> humanReadableDuration(float duration_ns) {
    std::string unit = " ns";
    float duration = duration_ns;
    if (duration > 1e9) {
        duration /= 1e9;
        unit = " s";
    } else if (duration > 1e6) {
        duration /= 1e6;
        unit = " ms";
    } else if (duration > 1e3) {
        duration /= 1e3;
        unit = " us";
    }

    return std::make_tuple(duration, unit);
}

bool checkResults(size_t* dims, size_t num_dims, size_t B, uint64_t* r) {
    bool success = true;
    for (size_t i=0; i<B*dims[num_dims-1]/64; ++i) {
        if(r[i] != 0xffffffffffffffff) {
            std::cout << "failure in i: " << i << std::endl;
            std::cout << std::bitset<64>(r[i]) << std::endl;
            success = false;
            break;
        }
    }

    return success;
}

void showTimings(float duration_ns, size_t B) {
    float duration_s = duration_ns / 1e9;

    auto [duration, unit] = humanReadableDuration(duration_ns);
    auto [duration_per_sample, unit_per_sample] = humanReadableDuration(duration_ns/B);

    std::cout << "Execution time: "
              << duration
              << unit << std::endl;
    std::cout << B/duration_s
              << " examples/s"
              << std::endl;
    std::cout << duration_per_sample
              << unit_per_sample
              << "/sample"
              << std::endl;
}

int main() {

    constexpr size_t B = 100;

    constexpr size_t num_dims = 3;

    size_t dims[num_dims] = {
        28*28*32, // N
        128*32, // M
        10*32, // M2
    };

    uint64_t* x = (uint64_t*)malloc(B*dims[0]/64*sizeof(uint64_t));
    uint64_t* r = (uint64_t*)malloc(B*dims[num_dims-1]/64*sizeof(uint64_t));

    uint64_t* weights = mlp_alloc_weights(dims, num_dims);
    uint64_t* scratchs = mlp_alloc_scratchs(dims, num_dims, B);

    showMemory(dims, num_dims, B);

    for (size_t i=0; i<B*dims[0]/64; ++i) {
        x[i] = 0xffffffffffffffff;
        //x[i] = rand64();
    }

    for (size_t i=0; i<B*dims[num_dims-1]/64; ++i) {
        r[i] = 0;
    }

    for (size_t i=0; i<mlp_weights_size(dims, num_dims)/sizeof(uint64_t); ++i) {
        weights[i] = 0xffffffffffffffff;
        //w[i] = rand64();
    }

    for (size_t i=0; i<mlp_scratchs_size(dims, num_dims, B)/sizeof(uint64_t); ++i) {
        scratchs[i] = 0;
    }

    auto start = std::chrono::high_resolution_clock::now();
    mlp(r, x, weights, scratchs, B, dims, num_dims);
    auto end = std::chrono::high_resolution_clock::now();

    float duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    showTimings(duration_ns, B);

    bool success = checkResults(dims, num_dims, B, r);

    if (success) {
        std::cout << "SUCCESS" << std::endl;
    } else {
        std::cout << "FAILURE" << std::endl;
    }

    free(x);
    x = nullptr;

    free(r);
    r = nullptr;

    free(weights);
    weights = nullptr;

    free(scratchs);
    scratchs = nullptr;

    return 0;
}
