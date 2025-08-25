#include <cstdint>
#include <random>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>

using namespace tt::tt_metal;

constexpr uint32_t TILE_WIDTH = 32;
constexpr uint32_t TILE_HEIGHT = 32;

std::shared_ptr<Buffer> MakeBuffer(IDevice* device, uint32_t size, uint32_t page_size, bool sram = false) {
    InterleavedBufferConfig config{
        .device = device,
        .size = size,
        .page_size = page_size,
        .buffer_type = (sram ? BufferType::L1 : BufferType::DRAM)};
    return CreateBuffer(config);
}


using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;
std::shared_ptr<Buffer> MakeBuffer(IDevice* device, uint32_t n_tiles, size_t element_size, bool sram = false) {
    const uint32_t tile_size = element_size * TILE_WIDTH * TILE_HEIGHT;
    const uint32_t page_tiles = sram ? n_tiles : 1;
    return MakeBuffer(device, tile_size * n_tiles, page_tiles * tile_size, sram);
}

CBHandle MakeCircularBuffer(
    Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t size, uint32_t page_size, tt::DataFormat format) {
    CircularBufferConfig cb_src0_config = CircularBufferConfig(size, {{cb, format}}).set_page_size(cb, page_size);
    return CreateCircularBuffer(program, core, cb_src0_config);
}

CBHandle MakeCircularBufferFP32(Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t n_tiles) {
    constexpr uint32_t tile_size = sizeof(float) * TILE_WIDTH * TILE_HEIGHT;
    return MakeCircularBuffer(program, core, cb, n_tiles * tile_size, tile_size, tt::DataFormat::Float32);
}

CBHandle MakeCircularBufferBFP16(Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t n_tiles) {
    constexpr uint32_t tile_size = sizeof(bfloat16) * TILE_WIDTH * TILE_HEIGHT;
    return MakeCircularBuffer(program, core, cb, n_tiles * tile_size, tile_size, tt::DataFormat::Float16_b);
}

std::vector<float> cpu_rope(const std::vector<float>& vec, int pos, size_t D, size_t N)
{
    assert(D % 2 == 0 && "Dimension must be even");
    std::vector<float> result(vec.size());
    for(size_t n = 0; n < N; ++n) {
        size_t offset = n * D;
        for (size_t i = 0; i < D; i += 2) {
            float exponent = i / float(D);
            float e =  std::log(std::exp(exponent) * 10000.f);
            float freq = 1.0f / std::pow(10000.0f, exponent);

            float angle = pos * freq;
            float cos_angle = std::cos(angle);
            float sin_angle = std::sin(angle);

            float x = vec[offset + i];
            float y = vec[offset + i + 1];

            result[offset + i] = x * cos_angle - y * sin_angle;
            result[offset + i + 1] = x * sin_angle + y * cos_angle;
            // result[offset + i] = angle;
            // result[offset + i + 1] = angle;
        }
    }

    return result;
}

inline float check_vector_pcc(const std::vector<float>& vec_a, const std::vector<float>& vec_b) {
    // Calculate the mean of x and y values
    float x_mean = 0.0f;
    float y_mean = 0.0f;

    for (size_t i = 0; i < vec_a.size(); i++) {
        x_mean += vec_a[i];
        y_mean += vec_b[i];
    }

    x_mean /= vec_a.size();
    y_mean /= vec_b.size();

    // Calculate the covariance and standard deviation of x and y values
    float covariance = 0.0f;
    float x_stddev = 0.0f;
    float y_stddev = 0.0f;

    for (size_t i = 0; i < vec_a.size(); i++) {
        float x_diff = vec_a[i] - x_mean;
        float y_diff = vec_b[i] - y_mean;

        covariance += x_diff * y_diff;
        x_stddev += x_diff * x_diff;
        y_stddev += y_diff * y_diff;
    }

    covariance /= vec_a.size();
    x_stddev /= vec_a.size();
    y_stddev /= vec_b.size();

    // Calculate the correlation coefficient
    float correlation_coefficient_ = covariance / (std::sqrt(x_stddev) * std::sqrt(y_stddev));
    return correlation_coefficient_;
}

int main()
{
    IDevice* device = CreateDevice(0);
    device->enable_program_cache();

    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    CommandQueue& cq = device->command_queue();

    constexpr size_t D = 32;
    constexpr size_t N = 32;
    static_assert(D % 32 == 0 && N % 32 == 0);
    constexpr uint32_t Dt = D/32;
    constexpr uint32_t Nt = N/32;
    auto src = MakeBuffer(device, Dt * Nt, sizeof(float));
    auto dst = MakeBuffer(device, Dt * Nt, sizeof(float));

    std::vector<float> src_vec(N * D);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for(auto& v : src_vec) {
        v = dist(rng);
    }
    std::cout << "\n";

    std::vector<float> tilized_src = convert_layout(tt::stl::Span<const float>(src_vec), {N, D}, TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_NFACES);
    EnqueueWriteBuffer(cq, src, tilized_src, false);

    MakeCircularBufferFP32(program, core, tt::CBIndex::c_0, 4);
    MakeCircularBufferFP32(program, core, tt::CBIndex::c_16, 4);

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src).append_to(reader_compile_time_args);
    KernelHandle reader = CreateKernel(program, "../ttrope/kernels/reader.cpp", core, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = reader_compile_time_args
    });

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst).append_to(writer_compile_time_args);
    KernelHandle writer = CreateKernel(program, "../ttrope/kernels/writer.cpp", core, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
        .compile_args = writer_compile_time_args
    });

    KernelHandle compute = CreateKernel(program, "../ttrope/kernels/compute.cpp", core, ComputeConfig{
        .fp32_dest_acc_en = true,
    });

    SetRuntimeArgs(program, reader, core, std::vector<uint32_t>{src->address(), Dt, Dt, Nt});
    SetRuntimeArgs(program, compute, core, std::vector<uint32_t>{Dt, Dt, Nt});
    SetRuntimeArgs(program, writer, core, std::vector<uint32_t>{dst->address(), Dt, Dt, Nt});

    EnqueueProgram(cq, program, true);

    std::vector<float> result_vec_tiled;
    EnqueueReadBuffer(cq, dst, result_vec_tiled, true);
    std::vector<float> result_vec = convert_layout(tt::stl::Span<const float>(result_vec_tiled), {N, D}, TensorLayoutType::TILED_NFACES, TensorLayoutType::LIN_ROW_MAJOR);

    // for(auto v : result_vec) {
    //     std::cout << v << " ";
    // }
    // std::cout << "\n";

    std::vector<float> in(src_vec.size());
    for(size_t i = 0; i < src_vec.size(); i++) {
        in[i] = src_vec[i];
    }
    auto reference = cpu_rope(in, 1000, D, N);

    if(reference.size() != result_vec.size()) {
        std::cerr << "Error: Result size mismatch" << std::endl;
        return 0;
    }

    size_t correct_count = 0;
    for(size_t i = 0; i < result_vec.size(); i++) {
        float refv = reference[i];
        float resv = result_vec[i];
        if(std::abs(refv - resv) > 1e-1) {
            std::cerr << "-- Index " << i << ": Reference value " << refv << ", Result value " << resv << std::endl;
        }
        else {
            std::cout << "idx " << i << " ok. value = " << refv << std::endl;
            correct_count++;
        }
    }
    std::cout << "\n";
    std::cout << "OK: " << correct_count << "/" << result_vec.size() << ", precentage: " << (float)correct_count / result_vec.size() * 100 << "%" << std::endl;
    std::cout << "PCC: " << check_vector_pcc(reference, result_vec) << "\n";
}
