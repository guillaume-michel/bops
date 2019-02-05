#include <bops.h>

#include <cstring>
#include <algorithm>

namespace {

/// compute size in bytes for weights in given layer
size_t mlp_weight_layer_size(size_t* dims,
                             size_t layer) {

    auto N = dims[layer];
    auto M = dims[layer+1];
    return M*N/64*sizeof(uint64_t);
}

/// returns the weights for a given layer
uint64_t* mlp_weight_layer(uint64_t* weights,
                           size_t* dims,
                           size_t layer) {

    if (layer == 0) {
        return weights;
    }

    size_t offset = 0;
    for (size_t i=0; i<layer; ++i) {
        offset += mlp_weight_layer_size(dims, layer);
    }

    return weights + offset/sizeof(uint64_t);
}

/// compute size in bytes for scratch in given layer
size_t mlp_scratch_layer_size(size_t* dims,
                              size_t layer,
                              size_t B) {

    auto M = dims[layer+1];
    return B*M/64*sizeof(uint64_t);
}

/// returns the scratch area for a given layer
uint64_t* mlp_scratch_layer(uint64_t* scratchs,
                            size_t* dims,
                            size_t layer,
                            size_t B) {

    if (layer%2 == 0) {
        return scratchs;
    } else {

        size_t offset = 0;
        for (size_t i=0; i<layer; i+=2) {
            size_t r_size = mlp_scratch_layer_size(dims, i, B);
            offset = std::max(offset, r_size);
        }

        return scratchs + offset/sizeof(uint64_t);
    }
}

}

size_t mlp_weights_size(size_t* dims,
                        size_t num_dims) {
    if (num_dims < 2) {
        return 0;
    }

    size_t total_size = 0;
    for (size_t layer=0; layer<num_dims-1; ++layer) {
        total_size += mlp_weight_layer_size(dims, layer);
    }

    return total_size;
}

uint64_t* mlp_alloc_weights(size_t* dims,
                            size_t num_dims) {

    auto total_size = mlp_weights_size(dims, num_dims);

    if (total_size == 0) {
        return nullptr;
    }

    return (uint64_t*)malloc(total_size);
}

size_t mlp_scratchs_size(size_t* dims,
                         size_t num_dims,
                         size_t B) {
    if (num_dims < 2) {
        return 0;
    }

    size_t r0_size = 0;
    size_t r1_size = 0;
    for (size_t layer=0; layer<num_dims-1; ++layer) {
        size_t r_size = mlp_scratch_layer_size(dims, layer, B);

        if (layer%2 == 0) {
            r0_size = std::max(r0_size, r_size);
        } else {
            r1_size = std::max(r1_size, r_size);
        }
    }

    return r0_size + r1_size;
}

uint64_t* mlp_alloc_scratchs(size_t* dims,
                             size_t num_dims,
                             size_t B) {

    auto total_size = mlp_scratchs_size(dims, num_dims, B);

    if (total_size == 0) {
        return nullptr;
    }

    return (uint64_t*)malloc(total_size);
}

void mlp(uint64_t* __restrict__ r,
         uint64_t* __restrict__ x,
         uint64_t* __restrict__ weights,
         uint64_t* __restrict__ scratchs,
         size_t B,
         size_t* dims,
         size_t num_dims) {

    if (num_dims < 2) {
        return;
    }

    uint64_t* ri = nullptr;
    uint64_t* input = x;
    size_t layer = 0;
    for (layer=0; layer<num_dims-1; ++layer) {
        ri = mlp_scratch_layer(scratchs,
                               dims,
                               layer,
                               B);
        uint64_t* wi = mlp_weight_layer(weights,
                                        dims,
                                        layer);
        size_t N = dims[layer];
        size_t M = dims[layer+1];

        dense(ri, input, wi, N, M, B);

        input = ri;
    }

    // copy last result stored in scratch to final result
    memcpy(r, ri, mlp_scratch_layer_size(dims, layer-1, B));
}
