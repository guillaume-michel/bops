#include <mnist.hpp>

#include <cstdio>
#include <optional>

namespace mnist {

namespace {

// train files
const std::string x_train_filename = "train-images-idx3-ubyte";
const std::string y_train_filename = "train-labels-idx1-ubyte";

// test files
const std::string x_test_filename = "t10k-images-idx3-ubyte";
const std::string y_test_filename = "t10k-labels-idx1-ubyte";

constexpr uint32_t reverseBits(uint32_t n) {
    uint8_t ch1 = n & 255;
    uint8_t ch2 = (n >> 8) & 255;
    uint8_t ch3 = (n >> 16) & 255;
    uint8_t ch4 = (n >> 24) & 255;

    return((uint32_t)ch1 << 24) + ((uint32_t)ch2 << 16) + ((uint32_t)ch3 << 8) + ch4;
}

std::optional<std::vector<uint64_t>> read_images(const std::string& filename) {

    FILE* file = fopen(filename.c_str(), "rb");

    uint8_t magic[4];
    if (fread(&magic, sizeof(uint8_t), 4, file) != 4*sizeof(uint8_t) ||
        magic[0] != 0 ||
        magic[1] != 0 ||
        magic[2] != 0x08 ||
        magic[3] != 0x03) {
        fclose(file);
        return std::nullopt;
    }

    uint32_t count;
    if (fread(&count, sizeof(count), 1, file) != 1) {
        fclose(file);
        return std::nullopt;
    }
    count = reverseBits(count);

    uint32_t rows;
    if (fread(&rows, sizeof(rows), 1, file) != 1) {
        fclose(file);
        return std::nullopt;
    }
    rows = reverseBits(rows);

    uint32_t cols;
    if (fread(&cols, sizeof(cols), 1, file) != 1) {
        fclose(file);
        return std::nullopt;
    }
    cols = reverseBits(cols);

    auto dim = count*rows*cols*bits_per_channels*channels/64;
    std::vector<uint64_t> images(dim, 0);
    if (fread(images.data(), sizeof(uint64_t), dim, file) != dim) {
        fclose(file);
        return std::nullopt;
    }

    fclose(file);

    return images;
}

std::optional<std::vector<uint8_t>> read_labels(const std::string& filename) {

    FILE* file = fopen(filename.c_str(), "rb");

    uint8_t magic[4];
    if (fread(&magic, sizeof(uint8_t), 4, file) != 4*sizeof(uint8_t) ||
        magic[0] != 0 ||
        magic[1] != 0 ||
        magic[2] != 0x08 ||
        magic[3] != 0x01) {
        fclose(file);
        return std::nullopt;
    }

    uint32_t count;
    if (fread(&count, sizeof(count), 1, file) != 1) {
        fclose(file);
        return std::nullopt;
    }
    count = reverseBits(count);

    std::vector<uint8_t> labels(count, 0);
    if (fread(labels.data(), sizeof(uint8_t), count, file) != count) {
        fclose(file);
        return std::nullopt;
    }

    fclose(file);

    return labels;
}

}

std::optional<
std::tuple<std::vector<uint64_t>, // Xtrain
           std::vector<uint8_t>,  // Ytrain
           std::vector<uint64_t>, // Xtest
           std::vector<uint8_t>>> // Ytest
read_mnist(const std::string& dirPath) {

    auto Xtrain = read_images(dirPath + "/" + x_train_filename);
    auto Ytrain = read_labels(dirPath + "/" + y_train_filename);

    auto Xtest = read_images(dirPath + "/" + x_test_filename);
    auto Ytest = read_labels(dirPath + "/" + y_test_filename);

    if (!Xtrain ||
        !Ytrain ||
        !Xtest ||
        !Ytest) {
        return std::nullopt;
    }

    if (Xtrain->size() != num_train_samples*width*height*channels*bits_per_channels/64) {
        return std::nullopt;
    }

    if (Ytrain->size() != num_train_samples) {
        return std::nullopt;
    }

    if (Xtest->size() != num_test_samples*width*height*channels*bits_per_channels/64) {
        return std::nullopt;
    }

    if (Ytest->size() != num_test_samples) {
        return std::nullopt;
    }

    return std::make_tuple(*Xtrain, *Ytrain, *Xtest, *Ytest);
}

}
