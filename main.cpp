#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <cassert>

std::vector<float> rope(const std::vector<float>& vec, int pos)
{
    size_t D = vec.size();
    assert(D % 2 == 0 && "Dimension must be even");
    std::vector<float> result(D);
    for (size_t i = 0; i < D; i += 2) {
        float exponent = i / float(D);
        float freq = 1.0f / std::pow(10000.0f, exponent);

        float angle = pos * freq;
        float cos_angle = std::cos(angle);
        float sin_angle = std::sin(angle);

        float x = vec[i];
        float y = vec[i + 1];

        result[i] = x * cos_angle - y * sin_angle;
        result[i + 1] = x * sin_angle + y * cos_angle;
    }

    return result;
}

int main()
{
    constexpr size_t D = 128;
    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> vec(D);
    for (auto& v : vec) {
        v = dist(rng);
    }

    auto res = rope(vec, 0);
    for (size_t i = 0; i < res.size(); ++i) {
        std::cout << res[i] << (i < res.size() - 1 ? ", " : "\n");
    }
}