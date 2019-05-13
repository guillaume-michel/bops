#include <vector>
#include <chrono>
#include <iostream>
#include <cstdint>
#include <cstddef>

constexpr size_t N = 1024*1024*1;
constexpr size_t M = 1000;

void vec_xor(std::vector<uint64_t>& r, const std::vector<uint64_t>& x, const std::vector<uint64_t>& y) {
    for (size_t i=0; i<x.size(); ++i) {
        r[i] = x[i] ^ y[i];
    }
}

int main() {

    const std::vector<uint64_t> x(N, 1);
    const std::vector<uint64_t> y(N, 0);
    std::vector<uint64_t> r(N, 0);

    auto start = std::chrono::system_clock::now();
    for (size_t i=0; i<M; ++i) {
        vec_xor(r, x, y);
    }
    auto end = std::chrono::system_clock::now();

    std::cout << "execution time: "
              << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()/1e9
              << " s" << std::endl;

    return 0;
}
