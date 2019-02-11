#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <cstdint>
#include <optional>

namespace mnist {

inline constexpr size_t num_train_samples = 60000;
inline constexpr size_t num_test_samples = 10000;

inline constexpr size_t width = 28;
inline constexpr size_t height = width;
inline constexpr size_t channels = 1;
inline constexpr size_t bits_per_channels = 8;

inline constexpr size_t num_classes = 10;

std::optional<
std::tuple<std::vector<uint64_t>, // Xtrain
           std::vector<uint8_t>,  // Ytrain
           std::vector<uint64_t>, // Xtest
           std::vector<uint8_t>>> // Ytest
read_mnist(const std::string& dirPath);

}
