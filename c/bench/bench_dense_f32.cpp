#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <bitset>
#include <tuple>

#include <cblas.h>
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

void showTimings(float duration_ns, size_t N) {
    float duration_s = duration_ns / 1e9;

    auto [duration, unit] = humanReadableDuration(duration_ns);
    auto [duration_per_sample, unit_per_sample] = humanReadableDuration(duration_ns/N);

    std::cout << "Execution time: "
              << duration
              << unit << std::endl;
    std::cout << N/duration_s
              << " examples/s"
              << std::endl;
    std::cout << duration_per_sample
              << unit_per_sample
              << "/sample"
              << std::endl;
}

int main() {
    constexpr size_t num_threads = 1;
    constexpr size_t repeat = 100;
    constexpr size_t N = 60000;
    constexpr size_t C = 1;
    constexpr size_t H = 32;
    constexpr size_t W = H;
    constexpr size_t M = 128;

    openblas_set_num_threads(num_threads);

    float* x = (float*)malloc(N*C*H*W*sizeof(float));
    float* r = (float*)malloc(N*M*sizeof(float));

    float* weights = (float*)malloc(M*C*H*W*sizeof(float));
    float* b = (float*)malloc(M*sizeof(float));

    for (size_t i=0; i<N*C*H*W; ++i) {
        x[i] = (float)i/N*C*H*W;
    }

    for (size_t i=0; i<N*M; ++i) {
        r[i] = 0;
    }

    for (size_t i=0; i<M*C*H*W; ++i) {
        weights[i] = (float)i/M*C*H*W;
    }

    for (size_t i=0; i<M; ++i) {
        b[i] = (float)i/M;
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i=0; i<repeat; ++i) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, M, C*H*W, 1.0f, x, C*H*W, weights, M, 1.0f, r, M);
    }
    auto end = std::chrono::high_resolution_clock::now();

    float duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()/repeat;

    showTimings(duration_ns, N);

    free(x);
    x = nullptr;

    free(r);
    r = nullptr;

    free(weights);
    weights = nullptr;

    free(b);
    b = nullptr;

    return 0;
}
