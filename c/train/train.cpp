#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <bitset>
#include <tuple>
#include <vector>
#include <algorithm>
#include <ctime>
#include <numeric>
#include <exception>

#include <bops.h>
#include <mnist.hpp>

const std::string mnist_dirPath = std::string(getenv("HOME")) + "/data/mnist";

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

uint8_t prediction(const std::vector<uint64_t>& y,
                   size_t index) {

    uint32_t* y32 = (uint32_t*)y.data();

    auto begin = &(y32[index*mnist::num_classes]);
    auto end = &(y32[index*mnist::num_classes + mnist::num_classes - 1]);

    return (uint8_t)std::distance(begin,
                                  std::max_element(begin, end));
}

float accuracy(const std::vector<uint64_t>& y,
               const std::vector<uint8_t>& Y,
               size_t num_samples) {

    // compute accuracy
    float acc = 0;

    // compute preds
    for (size_t i=0; i<num_samples; ++i) {
        acc += float(prediction(y, i) == Y[i]);
    }

    return acc / num_samples;
}


void probabilities(std::vector<float>& probabilities,
                   const std::vector<uint64_t>& y,
                   size_t num_samples) {
    for (size_t i=0; i<num_samples; ++i) {
        uint32_t* y32 = (uint32_t*)y.data();

        auto begin = &(y32[i*mnist::num_classes]);
        auto end = &(y32[i*mnist::num_classes + mnist::num_classes - 1]);

        float sum = std::accumulate(begin, end, 0.0f);

        for (size_t k=0; k<mnist::num_classes; ++k) {
            if (sum == 0.0f) {
                probabilities[i*mnist::num_classes + k] = 1.0f / mnist::num_classes;
            } else {
                probabilities[i*mnist::num_classes + k] = float(y[i*mnist::num_classes + k]) / sum;
            }
        }
    }
}

auto load_mnist_dataset(const std::string& mnist_dirPath) {
    std::cout << "Reading MNIST dataset...";

    auto mnist_datas = mnist::read_mnist(mnist_dirPath);
    if (!mnist_datas) {
        throw std::runtime_error("Could not load MNIST dataset");
    }

    std::cout << "DONE" << std::endl;

    return *mnist_datas;
}

int main() {

    srand(time(NULL)); // randomize seed

    constexpr size_t B = 6000;
    static_assert(mnist::num_train_samples % B == 0, "batch size should be a multiple of num_train_samples");

    constexpr size_t num_dims = 3;

    size_t dims[num_dims] = {
        mnist::width * mnist::height * mnist::channels * mnist::bits_per_channels, // N
        128*32, // M
        mnist::num_classes*32, // M2
    };

    //--------------------------------------------------------------------------
    // MNIST dataset
    auto [Xtrain, Ytrain, Xtest, Ytest] = load_mnist_dataset(mnist_dirPath);
    //--------------------------------------------------------------------------

    std::vector<uint64_t> ytrain(B*dims[num_dims-1]/64, 0);
    std::vector<uint64_t> ytest(mnist::num_test_samples*dims[num_dims-1]/64, 0);

    uint64_t* weights = mlp_alloc_weights(dims, num_dims);
    uint64_t* train_scratchs = mlp_alloc_scratchs(dims, num_dims, B);
    uint64_t* test_scratchs = mlp_alloc_scratchs(dims, num_dims, mnist::num_test_samples);

    showMemory(dims, num_dims, B);

    // random initialization of weights
    for (size_t i=0; i<mlp_weights_size(dims, num_dims)/sizeof(uint64_t); ++i) {
        weights[i] = rand64();
    }

    auto start = std::chrono::high_resolution_clock::now();
    mlp(ytest.data(),
        Xtest.data(),
        weights,
        test_scratchs,
        mnist::num_test_samples,
        dims,
        num_dims);
    auto end = std::chrono::high_resolution_clock::now();

    float duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    showTimings(duration_ns, mnist::num_test_samples);

    //--------------------------------------------------------
    float test_acc = accuracy(ytest, Ytest, mnist::num_test_samples);

    std::cout << "Test accuracy: " << test_acc << std::endl;

    mlp(ytrain.data(),
        Xtrain.data(),
        weights,
        train_scratchs,
        B,
        dims,
        num_dims);

    float train_acc = accuracy(ytrain, Ytrain, B);

    std::cout << "Train accuracy: " << train_acc << std::endl;

    //--------------------------------------------------------
    free(weights);
    weights = nullptr;

    free(train_scratchs);
    train_scratchs = nullptr;

    free(test_scratchs);
    test_scratchs = nullptr;

    return 0;
}
