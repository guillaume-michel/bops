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

float randf() {
    return rand() / (float)RAND_MAX;
}

uint64_t rand64() {
    uint64_t x = rand() | ((uint64_t)rand())<<32;

    if (rand()/(float)RAND_MAX > 0.5) {
        x = x | (((uint64_t)1)<<63);
    }

    return x;
}

void randomize(std::vector<uint64_t>& w) {
    for (size_t i=0; i<w.size(); ++i) {
        w[i] = rand64();
    }
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
              << unit
              << " - "
              << B/duration_s
              << " examples/s"
              << " - "
              << duration_per_sample
              << unit_per_sample
              << "/sample"
              << std::endl;
}

size_t argmin(const std::vector<float>& values) {
    auto it = std::min_element(std::begin(values), std::end(values));

    return std::distance(std::begin(values), it);
}

uint8_t prediction(const std::vector<uint32_t>& y,
                   size_t index) {

    auto begin = &(y[index*mnist::num_classes]);
    auto end = &(y[index*mnist::num_classes + mnist::num_classes - 1]);

    return (uint8_t)std::distance(begin,
                                  std::max_element(begin, end));
}

float accuracy(const std::vector<uint32_t>& y,
               uint8_t* Y,
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

template<typename InputIt>
float probability(InputIt begin,
                  InputIt end,
                  size_t index) {

    float sum = std::accumulate(begin, end, 0.0f);

    if (sum == 0.0f) {
        return 1.0f / mnist::num_classes;
    } else {
        auto it = begin;
        std::advance(it, index);
        return float(*it) / sum;
    }
}

float loss(const std::vector<uint32_t>& y,
           uint8_t* Y,
           size_t num_samples) {

    float loss = 0.0f;

    for (size_t i=0; i<num_samples; ++i) {
        auto begin = &(y[i*mnist::num_classes]);
        auto end = &(y[i*mnist::num_classes + mnist::num_classes - 1]);
        auto ground_truth_label = Y[i];
        loss += (1 - probability(begin, end, ground_truth_label));
    }

    return loss / num_samples;
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

uint64_t uniformMask(float prob) {
    uint64_t mask = 0;

    for (size_t j=0; j<64; ++j) {
        if (randf() <= prob) {
            mask |= (uint64_t(1)<<j);
        }
    }

    return mask;
}

void uniformMutation(std::vector<uint64_t>& w,
                     float pmut) {

    for (size_t i=0; i<w.size(); ++i) {
        uint64_t mask = uniformMask(pmut);

        w[i] ^= mask;
    }
}

void uniformCrossover(std::vector<uint64_t>& w1,
                      std::vector<uint64_t>& w2,
                      float pcross) {

    for (size_t i=0; i<w1.size(); ++i) {
        uint64_t mask = uniformMask(pcross);

        uint64_t bitdiff = (w1[i]^w2[i]) & mask;

        w1[i] ^= bitdiff;
        w2[i] ^= bitdiff;
    }
}

std::vector<std::vector<uint64_t>> createPopulation(size_t* dims,
                                                    size_t num_dims,
                                                    size_t population_size) {
    std::vector<std::vector<uint64_t>> population(population_size);

    size_t weights_size = mlp_weights_size(dims, num_dims)/sizeof(uint64_t);

    for (size_t i=0; i<population_size; ++i) {
        population[i].resize(weights_size);
        randomize(population[i]);
    }

    return population;
}

std::tuple<std::vector<float>,
           std::vector<float>>
evaluateFitness(std::vector<std::vector<uint64_t>>& population,
                size_t* dims,
                size_t num_dims,
                size_t num_samples,
                uint64_t* Xtrain_batch,
                uint8_t* Ytrain_batch) {

    std::vector<float> fitnesses(population.size());
    std::vector<float> accuracies(population.size());

    for(size_t i=0; i<population.size(); ++i) {
        std::vector<uint32_t> ytrain(num_samples*dims[num_dims-1]/64*2, 0);
        std::vector<uint64_t> train_scratchs(mlp_scratchs_size(dims, num_dims, num_samples));

        auto start = std::chrono::high_resolution_clock::now();
        mlp((uint64_t*)ytrain.data(),
            Xtrain_batch,
            population[i].data(),
            train_scratchs.data(),
            num_samples,
            dims,
            num_dims);
        auto end = std::chrono::high_resolution_clock::now();

        float duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        (void)duration_ns;
        //showTimings(duration_ns, num_samples);

        float train_acc = accuracy(ytrain, Ytrain_batch, num_samples);

        //std::cout << "Train accuracy: " << train_acc << std::endl;

        float train_loss = loss(ytrain, Ytrain_batch, num_samples);

        //std::cout << "Train loss: " << train_loss << std::endl;

        fitnesses[i] = train_loss;
        accuracies[i] = train_acc;
    }

    return std::make_tuple(fitnesses, accuracies);
}

std::tuple<std::vector<uint64_t>, // best individual weights
           float,                 // best individual train loss
           float,                 // best individual train accuracy
           float,                 // best individual test loss
           float>                 // best individual test accuracy
run(std::vector<std::vector<uint64_t>>& population,
    float pcross,
    float pmut,
    size_t* dims,
    size_t num_dims,
    size_t train_batch_size,
    size_t eval_batch_size,
    size_t NG,
    std::vector<uint64_t>& Xtrain,
    std::vector<uint8_t>& Ytrain,
    std::vector<uint64_t>& Xtest,
    std::vector<uint8_t>& Ytest) {

    (void)pcross;
    (void)pmut;
    const size_t population_size = population.size();

    size_t Xbatch_offset = train_batch_size * mnist::width * mnist::height * mnist::channels * mnist::bits_per_channels / 64;

    uint64_t* Xtrain_batch = Xtrain.data();
    uint8_t* Ytrain_batch = Ytrain.data();

    auto [fitnesses, train_accuracies] = evaluateFitness(population,
                                                         dims,
                                                         num_dims,
                                                         train_batch_size,
                                                         Xtrain_batch,
                                                         Ytrain_batch);
    auto bestIndex = argmin(fitnesses);
    auto bestIndividual = population[bestIndex];

    float best_train_loss = fitnesses[bestIndex];
    float best_train_accuracy = train_accuracies[bestIndex];
    float best_test_loss = 1;
    float best_test_accuracy = 0;

    std::vector<uint32_t> ytest(eval_batch_size*dims[num_dims-1]/64*2, 0);
    std::vector<uint64_t> test_scratchs(mlp_scratchs_size(dims, num_dims, eval_batch_size));

    // evaluate TEST metrics for best individual
    mlp((uint64_t*)ytest.data(),
        Xtest.data(),
        bestIndividual.data(),
        test_scratchs.data(),
        eval_batch_size,
        dims,
        num_dims);

    best_test_accuracy = accuracy(ytest, Ytest.data(), eval_batch_size);
    best_test_loss = loss(ytest, Ytest.data(), eval_batch_size);

    std::cout << "best individual: TRAIN accuracy: " << best_train_accuracy
              << " - TRAIN loss: " << best_train_loss
              << " - TEST accuracy: " << best_test_accuracy
              << " - TEST loss: " << best_test_loss << std::endl;

    for (size_t g=0; g<NG; ++g) {
        std::cout << "-----------------------------------------------------------" << std::endl;
        std::cout << "Generation: " << g << std::endl;

        Xtrain_batch += Xbatch_offset;
        Ytrain_batch += train_batch_size;

        // candidate individuals
        auto candidates = createPopulation(dims, num_dims, population_size);

        auto [candidates_fitnesses, candidates_train_accuracies] = evaluateFitness(candidates,
                                                                                   dims,
                                                                                   num_dims,
                                                                                   train_batch_size,
                                                                                   Xtrain_batch,
                                                                                   Ytrain_batch);

        auto all = fitnesses;
        all.insert(std::end(all), std::begin(candidates_fitnesses), std::end(candidates_fitnesses));

        std::vector<size_t> index(all.size(), 0);
        for (size_t i=0; i<index.size(); ++i) {
            index[i] = i;
        }

        std::sort(std::begin(index),
                  std::end(index),
                  [&](size_t a, size_t b) {
                      return (all[a] < all[b]);
                  });

        // update population and fitness with the best elements
        auto new_population = population;
        auto new_fitnesses = fitnesses;
        auto new_train_accuracies = train_accuracies;
        for (size_t i=0; i<population_size; ++i) {
            if (index[i] < population_size) {
                // parent is better
                new_population[i] = population[index[i]];
                new_fitnesses[i] = fitnesses[index[i]];
                new_train_accuracies[i] = train_accuracies[index[i]];
            } else {
                // candidate is better
                new_population[i] = candidates[index[i]-population_size];
                new_fitnesses[i] = candidates_fitnesses[index[i]-population_size];
                new_train_accuracies[i] = candidates_train_accuracies[index[i]];
            }
        }

        population = new_population;
        fitnesses = new_fitnesses;
        train_accuracies = new_train_accuracies;
        bestIndex = 0;
        bestIndividual = population[0];

        best_train_loss = fitnesses[bestIndex];
        best_train_accuracy = train_accuracies[bestIndex];

        // evaluate TEST metrics for best individual
        mlp((uint64_t*)ytest.data(),
            Xtest.data(),
            bestIndividual.data(),
            test_scratchs.data(),
            eval_batch_size,
            dims,
            num_dims);

        best_test_accuracy = accuracy(ytest, Ytest.data(), eval_batch_size);
        best_test_loss = loss(ytest, Ytest.data(), eval_batch_size);

        std::cout << "best individual: TRAIN accuracy: " << best_train_accuracy
                  << " - TRAIN loss: " << best_train_loss
                  << " - TEST accuracy: " << best_test_accuracy
                  << " - TEST loss: " << best_test_loss << std::endl;
    }

    return make_tuple(bestIndividual, best_train_loss, best_train_accuracy, best_test_loss, best_test_accuracy);
}

int main() {

    srand(time(NULL)); // randomize seed

    constexpr size_t B = 600;
    static_assert(mnist::num_train_samples % B == 0, "batch size should be a multiple of num_train_samples");

    constexpr size_t num_dims = 3;

    size_t dims[num_dims] = {
        mnist::width * mnist::height * mnist::channels * mnist::bits_per_channels, // N
        128*32, // M
        mnist::num_classes*32, // M2
    };

    constexpr size_t NG = 1000;
    constexpr size_t population_size = 50;
    constexpr float pcross = 1e-4;
    constexpr float pmut = 1e-5;

    showMemory(dims, num_dims, B);

    //--------------------------------------------------------------------------
    // MNIST dataset
    auto [Xtrain, Ytrain, Xtest, Ytest] = load_mnist_dataset(mnist_dirPath);
    //--------------------------------------------------------------------------

    // population for the current generation
    auto population = createPopulation(dims, num_dims, population_size);

    auto [bestIndividual,
          best_train_loss,
          best_train_accuracy,
          best_test_loss,
          best_test_accuracy] = run(population,
                                    pcross,
                                    pmut,
                                    dims,
                                    num_dims,
                                    B,
                                    mnist::num_test_samples,
                                    NG,
                                    Xtrain,
                                    Ytrain,
                                    Xtest,
                                    Ytest);

    std::cout << "===========================================================" << std::endl;

    std::cout << "weights:" << std::endl;
    for (size_t i=0; i<bestIndividual.size(); ++i) {
        std::cout << std::hex << bestIndividual[i] << std::endl;
    }

    std::cout << "best individual: TRAIN accuracy: " << best_train_accuracy
              << " - TRAIN loss: " << best_train_loss
              << " - TEST accuracy: " << best_test_accuracy
              << " - TEST loss: " << best_test_loss << std::endl;



    // std::vector<uint32_t> ytrain(B*dims[num_dims-1]/64*2, 0);
    // std::vector<uint32_t> ytest(mnist::num_test_samples*dims[num_dims-1]/64*2, 0);

    // std::vector<uint64_t> weights(mlp_weights_size(dims, num_dims)/sizeof(uint64_t));
    // std::vector<uint64_t> train_scratchs(mlp_scratchs_size(dims, num_dims, B));
    // std::vector<uint64_t> test_scratchs(mlp_scratchs_size(dims, num_dims, mnist::num_test_samples));

    // // random initialization of weights
    // randomize(weights);

    // auto start = std::chrono::high_resolution_clock::now();
    // mlp((uint64_t*)ytest.data(),
    //     Xtest.data(),
    //     weights.data(),
    //     test_scratchs.data(),
    //     mnist::num_test_samples,
    //     dims,
    //     num_dims);
    // auto end = std::chrono::high_resolution_clock::now();

    // float duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // showTimings(duration_ns, mnist::num_test_samples);

    // //--------------------------------------------------------
    // float test_acc = accuracy(ytest, Ytest, mnist::num_test_samples);

    // std::cout << "Test accuracy: " << test_acc << std::endl;

    // float test_loss = loss(ytest, Ytest, mnist::num_test_samples);

    // std::cout << "Test loss: " << test_loss << std::endl;

    // mlp((uint64_t*)ytrain.data(),
    //     Xtrain.data(),
    //     weights.data(),
    //     train_scratchs.data(),
    //     B,
    //     dims,
    //     num_dims);

    // float train_acc = accuracy(ytrain, Ytrain, B);

    // std::cout << "Train accuracy: " << train_acc << std::endl;

    // float train_loss = loss(ytrain, Ytrain, B);

    // std::cout << "Train loss: " << train_loss << std::endl;

    return 0;
}
