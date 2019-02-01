#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <bitset>
#include <tuple>

#define INLINE inline

uint64_t rand64() {
    uint64_t x = rand() | ((uint64_t)rand())<<32;

    if (rand()/(float)RAND_MAX > 0.5) {
        x = x | (((uint64_t)1)<<63);
    }

    return x;
}

INLINE bool dense1_elem(uint64_t* __restrict__ xi,
                        uint64_t* __restrict__ wj,
                        size_t N) {
    int32_t total = 0;
    for (size_t k=0; k<N/64; ++k) {
        total += __builtin_popcountll(xi[k] ^ wj[k]);
    }
    total = N - 2*total;

    return (total>=0);
}

INLINE void dense1(uint64_t* __restrict__ ri,
                   uint64_t* __restrict__ xi,
                   uint64_t* __restrict__ w,
                   size_t N,
                   size_t M) {

    for (size_t j1=0; j1<M/64; ++j1) {
        uint64_t res = 0;

        for (size_t j2=0; j2<64; ++j2) {
            uint64_t* wj = w + (j1*64 + j2)*N/64;
            uint64_t b = dense1_elem(xi, wj, N) ? 1 : 0;
            res = res | (b<<j2);
        }

        ri[j1] = res;
    }
}

INLINE void dense(uint64_t* __restrict__ r,
                  uint64_t* __restrict__ x,
                  uint64_t* __restrict__ w,
                  size_t N,
                  size_t M,
                  size_t B) {

    for (size_t i=0; i<B; ++i) {
        uint64_t* xi = x + i*N/64;
        uint64_t* ri = r + i*M/64;
        dense1(ri, xi, w, N, M);
    }
}

void showMemory(size_t B,
                size_t N,
                size_t M,
                size_t M2) {

    auto sx = B*N/64*sizeof(uint64_t);
    auto sx_1 = 1*N/64*sizeof(uint64_t);
    auto sw = M*N/64*sizeof(uint64_t);
    auto sr = B*M/64*sizeof(uint64_t);
    auto sr_1 = 1*M/64*sizeof(uint64_t);
    auto sw2 = M*M2/64*sizeof(uint64_t);
    auto sr2 = B*M2/64*sizeof(uint64_t);
    auto sr2_1 = 1*M2/64*sizeof(uint64_t);

    auto total_1 = sx_1 + sw + sr_1 + sw2 + sr2_1;
    auto total = sx + sw + sr + sw2 + sr2;

    std::cout << "B = " << B << std::endl;
    std::cout << "N = " << N << std::endl;
    std::cout << "M = " << M << std::endl;
    std::cout << "M2= " << M2 << std::endl;

    std::cout << "Size on disk: " << (sw + sw2)/(1024.*1024) << " MB" << std::endl;
    std::cout << "Inference total memory (B=1): "
              << total_1/(1024.*1024) << " MB" << std::endl;
    std::cout << "Train memory " << total/(1024.*1024) << " MB" << std::endl;

}

/**
 * @brief take the given duration in nano sconds and returnd the duration in a human readable unit.
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

int main(int argc, char *argv[]) {

    // constexpr size_t B = 60000;
    // constexpr size_t N = 28*28*32;
    // constexpr size_t M = 128*32;
    // constexpr size_t M2 = 10*32;

    constexpr size_t B = 100;
    constexpr size_t N = 28*28*32;
    constexpr size_t M = 128*32;
    constexpr size_t M2 = 10*32;

    // constexpr size_t B = 10;
    // constexpr size_t N = 256;
    // constexpr size_t M = 128;
    // constexpr size_t M2 = 10*32;

    uint64_t* x = (uint64_t*)malloc(B*N/64*sizeof(uint64_t));
    uint64_t* w = (uint64_t*)malloc(M*N/64*sizeof(uint64_t));
    uint64_t* r = (uint64_t*)malloc(B*M/64*sizeof(uint64_t));

    uint64_t* w2 = (uint64_t*)malloc(M*M2/64*sizeof(uint64_t));
    uint64_t* r2 = (uint64_t*)malloc(B*M2/64*sizeof(uint64_t));

    showMemory(B, N, M, M2);

    for (size_t i=0; i<B*N/64; ++i) {
        x[i] = 0xffffffffffffffff;
        //x[i] = rand64();
    }

    for (size_t i=0; i<M*N/64; ++i) {
        w[i] = 0xffffffffffffffff;
        //w[i] = rand64();
    }

    for (size_t i=0; i<B*M/64; ++i) {
        r[i] = 0;
    }

    for (size_t i=0; i<M*M2/64; ++i) {
        w2[i] = 0xffffffffffffffff;
        //w2[i] = rand64();
    }

    for (size_t i=0; i<B*M2/64; ++i) {
        r2[i] = 0;
    }

    auto start = std::chrono::high_resolution_clock::now();
    dense(r, x, w, N, M, B);
    dense(r2, r, w2, M, M2, B);
    auto end = std::chrono::high_resolution_clock::now();

    float duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

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

    bool success = true;
    for (size_t i=0; i<B*M/64; ++i) {
        if(r[i] != 0xffffffffffffffff) {
            std::cout << "failure in i: " << i << std::endl;
            success = false;
            break;
        }
    }

    bool success2 = true;
    for (size_t i=0; i<B*M2/64; ++i) {
        if(r2[i] != 0xffffffffffffffff) {
            std::cout << "failure r2 in i: " << i << std::endl;
            std::cout << std::bitset<64>(r2[i]) << std::endl;
            success2 = false;
            break;
        }
    }

    if (success && success2) {
        std::cout << "SUCCESS" << std::endl;
    } else {
        std::cout << "FAILURE" << std::endl;
    }

    free(x);
    x = nullptr;

    free(w);
    w = nullptr;

    free(r);
    r = nullptr;

    free(w2);
    w2 = nullptr;

    free(r2);
    r2 = nullptr;

    return 0;
}
