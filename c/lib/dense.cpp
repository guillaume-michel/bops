#include <bops.h>

namespace {

inline bool dense1_elem(uint64_t* __restrict__ xi,
                        uint64_t* __restrict__ wj,
                        size_t N) {
    int32_t total = 0;
    for (size_t k=0; k<N/64; ++k) {
        total += __builtin_popcountll(xi[k] ^ wj[k]);
    }
    total = N - 2*total;

    return (total>=0);
}

inline void dense1(uint64_t* __restrict__ ri,
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

}

void dense(uint64_t* __restrict__ r,
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
