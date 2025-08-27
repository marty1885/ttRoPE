#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <cassert>

std::vector<float> rope(const std::vector<float>& vec, int pos, int active_dim = -1)
{
    size_t rotate_dim = active_dim == -1 ? vec.size() : active_dim;
    size_t D = vec.size();
    assert(D % 2 == 0 && "Dimension must be even");
    std::vector<float> result(D);
    for (size_t i = 0; i < rotate_dim/2; ++i) {
        float exponent = 2.0f * i / rotate_dim;
        float freq = 1.0f / std::pow(10000.0f, exponent);

        float angle = pos * freq;
        float cos_angle = std::cos(angle);
        float sin_angle = std::sin(angle);

        float x = vec[i];
        float y = vec[i + rotate_dim/2];

        result[i] = x * cos_angle - y * sin_angle;
        result[i + rotate_dim/2] = x * sin_angle + y * cos_angle;
    }
    for(size_t i = rotate_dim; i < D; ++i) {
        result[i] = vec[i];
    }

    return result;
}

int main()
{
    constexpr size_t D = 32;
    std::vector<float> vec(D);
    for(size_t i = 0; i < D; ++i) {
        vec[i] = i;
    }

    auto res = rope(vec, 1, D/2);
    for (size_t i = 0; i < res.size(); ++i) {
        std::cout << res[i] << (i < res.size() - 1 ? ", " : "\n");
    }
}
