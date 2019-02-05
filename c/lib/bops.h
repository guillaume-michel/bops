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

/**
 * @brief compute memory size in bytes to hold mlp weights
 * @details dims is an array of sizes with
 *          the first number beeing the input size
 *          the second number beeing the output size of the first layer
 *          and so on
 */
size_t mlp_weights_size(size_t* dims,
                        size_t num_dims);

/**
 * @brief allocate memory to hold mlp weights
 * @details dims is an array of sizes with
 *          the first number beeing the input size
 *          the second number beeing the output size of the first layer
 *          and so on
 * the returned pointer should be free with the free function
 */
uint64_t* mlp_alloc_weights(size_t* dims,
                            size_t num_dims);

/**
 * @brief compute scratch memory size in bytes to hold mlp intermediate results
 * @details dims is an array of sizes with
 *          the first number beeing the input size
 *          the second number beeing the output size of the first layer
 *          and so on
 */
size_t mlp_scratchs_size(size_t* dims,
                         size_t num_dims,
                         size_t B);

/**
 * @brief allocate memory to hold mlp intermediate results
 * @details dims is an array of sizes with
 *          the first number beeing the input size
 *          the second number beeing the output size of the first layer
 *          and so on
 * the returned pointer should be free with the free function
 */
uint64_t* mlp_alloc_scratchs(size_t* dims,
                             size_t num_dims,
                             size_t B);

/**
 * @brief Compute multi layer perceptron
 * result is stored in r
 * x shape is: (B, dims[0]/64)
 * w shape is: [(dim[1], dims[0]/64), (dim[2], dims[1]/64), ..., (dim[num_dims-1], dims[num_dims-2]/64)]
 * B is the batch size
 */
void mlp(uint64_t* __restrict__ r,
         uint64_t* __restrict__ x,
         uint64_t* __restrict__ weights,
         uint64_t* __restrict__ scratchs,
         size_t B,
         size_t* dims,
         size_t num_dims);

#ifdef __cplusplus
}
#endif
