#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief compute dense binary operation between x and w
 * results is stored in r
 * x shape is: (B, N/64)
 * w shape is: (M, N/64)
 * r shape is: (B, M/64)
 */
void dense(uint64_t* __restrict__ r,
           uint64_t* __restrict__ x,
           uint64_t* __restrict__ w,
           size_t N,
           size_t M,
           size_t B);

#ifdef __cplusplus
}
#endif
