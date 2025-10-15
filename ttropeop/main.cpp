#include <tt-metalium/circular_buffer_config.hpp>
#include <ttnn/decorators.hpp>
#include <ttnn/run_operation.hpp>
#include <ttnn/tensor/layout/layout.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/types.hpp>
#include <ttnn/api/ttnn/device.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/copy/typecast/typecast.hpp>
#include <ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp>
#include <cmath>

using namespace tt::tt_metal;
constexpr uint32_t TILE_WIDTH = 32;
constexpr uint32_t TILE_HEIGHT = 32;

// Yanked and adapted from other places in GGML
static float rope_yarn_corr_factor(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * std::log(n_ctx_orig / (n_rot * 2 * M_PI)) / (2 * std::log(base));
}

static std::array<float, 2> rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow
) {
    // start and end correction dims
    return std::array<float, 2>{
        std::max(0.0f,         std::floor(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base))),
        std::min(n_dims - 1.0f, std::ceil(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base)))
    };
}

static std::string to_string_precise(float value)
{
    std::stringstream ss;
    ss << std::hexfloat << value;
    return ss.str();
}


static std::shared_ptr<distributed::MeshBuffer> MakeBuffer(const std::shared_ptr<distributed::MeshDevice>& device, uint32_t size, uint32_t page_size, bool sram = false) {
    distributed::DeviceLocalBufferConfig config{
          .page_size = page_size,
          .buffer_type = (sram ? BufferType::L1 : BufferType::DRAM)
    };
    distributed::ReplicatedBufferConfig buffer_config{.size = size};
    return distributed::MeshBuffer::create(buffer_config, config, device.get());
}


using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;
template <typename T>
static std::shared_ptr<distributed::MeshBuffer> MakeBuffer(const std::shared_ptr<distributed::MeshDevice>& device, uint32_t n_tiles, bool sram = false) {
    const uint32_t tile_size = sizeof(T) * TILE_WIDTH * TILE_HEIGHT;
    return MakeBuffer(device, tile_size * n_tiles, tile_size, sram);
}

static CBHandle MakeCircularBuffer(
    Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t size, uint32_t page_size, tt::DataFormat format) {
    CircularBufferConfig cb_src0_config = CircularBufferConfig(size, {{cb, format}}).set_page_size(cb, page_size);
    return CreateCircularBuffer(program, core, cb_src0_config);
}

static CBHandle MakeCircularBuffer(Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t n_tiles, tt::tt_metal::DataType dtype)
{
    auto dt2dt = [](tt::tt_metal::DataType dt) {
        switch(dt) {
            case tt::tt_metal::DataType::FLOAT32: return tt::DataFormat::Float32;
            case tt::tt_metal::DataType::BFLOAT16: return tt::DataFormat::Float16_b;
            case tt::tt_metal::DataType::INT32: return tt::DataFormat::Int32;
            case tt::tt_metal::DataType::BFLOAT8_B: return tt::DataFormat::Bfp8_b;
            case tt::tt_metal::DataType::BFLOAT4_B: return tt::DataFormat::Bfp4_b;
            case tt::tt_metal::DataType::UINT8: return tt::DataFormat::UInt8;
            case tt::tt_metal::DataType::UINT16: return tt::DataFormat::UInt16;
            case tt::tt_metal::DataType::UINT32: return tt::DataFormat::UInt32;
            default:
                TT_FATAL(false, "Unsupported data type: {}", static_cast<int>(dt));
        }
    };

    auto tile_size = tt::tile_size(dt2dt(dtype));
    return MakeCircularBuffer(program, core, cb, n_tiles*tile_size, tile_size, dt2dt(dtype));
}


namespace ttggml {
using namespace ttnn;
struct RoPEOperation {
    static ttnn::Tensor invoke(const Tensor& src_tensor, const Tensor& index_tensor, uint32_t active_dim_size, uint32_t n_ctx_orig = 512, float freq_base = 10000.0f, float freq_scale = 1.f
        , float ext_factor = 0.f, float attn_factor = 1.f, float beta_fast = 0.f, float beta_slow = 0.f);
};
constexpr auto rope = ttnn::register_operation<"ttggml::rope", ttggml::RoPEOperation>();

struct RoPEDeviceOperation {
    const tt::tt_metal::MemoryConfig output_mem_config;
    const tt::tt_metal::DataType output_dtype{};
    const uint32_t active_dim_size = 0;
    const uint32_t n_ctx_orig = 512;
    const float freq_base = 10000.0f;
    const float freq_scale = 1.f;
    const float ext_factor = 0.f;
    const float attn_factor = 1.f;
    const float beta_fast = 0.f;
    const float beta_slow = 0.f;

    void validate_with_output_tensors(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;

    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};
}

ttnn::Tensor ttggml::RoPEOperation::invoke(const Tensor& src_tensor, const Tensor& index_tensor, uint32_t active_dim_size, uint32_t n_ctx_orig, float freq_base,
    float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    return tt::tt_metal::operation::run(
            RoPEDeviceOperation{
                src_tensor.memory_config(),
                src_tensor.dtype(),
                active_dim_size,
                n_ctx_orig,
                freq_base,
                freq_scale,
                ext_factor,
                attn_factor,
                beta_fast,
                beta_slow
            },
            {src_tensor, index_tensor},
            {},
            {})[0];
}

std::vector<ttnn::TensorSpec> ttggml::RoPEDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const
{
    if (!output_tensors.empty() && output_tensors[0].has_value()) {
        return {output_tensors[0]->tensor_spec()};
    }

    const auto& input_tensor = input_tensors.at(0);
    return {TensorSpec(
        input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout(
            output_dtype,
            tt::tt_metal::PageConfig(input_tensor.layout()),
            output_mem_config)
    )};
}

std::vector<ttnn::Tensor> ttggml::RoPEDeviceOperation::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (!output_tensors.empty() && output_tensors[0].has_value()) {
        return {output_tensors[0].value()};
    }
    const auto& input_tensor = input_tensors.at(0);
    auto spec = compute_output_specs(input_tensors, output_tensors)[0];
    return {create_device_tensor(spec, input_tensor.device())};
}

void ttggml::RoPEDeviceOperation::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    const auto& src_tensor = input_tensors.at(0);
    const auto& index_tensor = input_tensors.at(1);

    // expect src to have shape [batch, n_token, vec_dim]
    // expect index to have shape [batch, n_token]
    const auto& src_shape = src_tensor.logical_shape();
    const auto& index_shape = index_tensor.logical_shape();
    TT_FATAL(src_shape[-3] == index_shape[-1],
        "Shape mismatch: src_shape = {}, index_shape = {}. Expect format [batch, n_token, vec_dim] and [batch]", src_shape, index_shape);

    TT_FATAL(index_tensor.dtype() == tt::tt_metal::DataType::INT32, "Index tensor must be of type INT32");
    TT_FATAL(src_tensor.layout() == tt::tt_metal::Layout::TILE,  "Source tensor must be of layout TILE");
    TT_FATAL(index_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR,  "Index tensor must be of layout ROW_MAJOR");
    TT_FATAL(index_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "Index tensor must be on device");
    TT_FATAL(src_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "Source tensor must be on device");

    if (!output_tensors.empty() && output_tensors.at(0).has_value()) {
        const auto& out_tensor = output_tensors.at(0).value();
        TT_FATAL(out_tensor.logical_shape() == src_shape, "Output tensor shape must match source tensor shape");
        TT_FATAL(out_tensor.padded_shape() == src_tensor.padded_shape(), "Output tensor padded shape must match source tensor padded shape");
    }

    TT_FATAL(active_dim_size % 64 == 0, "active_dim must be a multiple of 64 (2 tiles)");
    TT_FATAL(active_dim_size <= src_tensor.padded_shape()[-1], "active_dim must be less than the last dimension of the source tensor");
    TT_FATAL(freq_base >= 0, "base_freq must be non-negative");
}

tt::tt_metal::operation::ProgramWithCallbacks ttggml::RoPEDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const
{
    tt::tt_metal::Program program{};
    const auto& src_tensor = input_tensors.at(0);
    const auto& index_tensor = input_tensors.at(1);
    const auto& output_tensor = output_tensors.at(0);

    const uint32_t B = src_tensor.logical_shape()[-3];
    const uint32_t D = src_tensor.logical_shape()[-1];
    const uint32_t D_active = active_dim_size;
    const uint32_t N = src_tensor.logical_shape()[-2];

    tt::tt_metal::IDevice* device = src_tensor.device();

    auto src = src_tensor.buffer();
    auto idxs = index_tensor.buffer();
    auto dst = output_tensor.buffer();

    const uint32_t Dt = D/32 + (D % 32 != 0);
    const uint32_t Nt = N/32 + (N % 32 != 0);
    const uint32_t D_activet = D_active/32; // This must divide cleanly

    auto core_grid = device->compute_with_storage_grid_size();

    uint32_t active_tiles = D_activet/2 * Nt * B;
    uint32_t passive_tiles = (Dt - D_activet) * Nt * B;
    auto [num_cores_active,
        all_cores_active,
        core_group_1_active,
        core_group_2_active,
        work_per_core1_active,
        work_per_core2_active] =
        tt::tt_metal::split_work_to_cores(core_grid, active_tiles);
    auto [num_cores_passive,
        all_cores_passive,
        core_group_1_passive,
        core_group_2_passive,
        work_per_core1_passive,
        work_per_core2_passive] =
        tt::tt_metal::split_work_to_cores(core_grid, passive_tiles);

    // Combine the two groups of cores
    auto all_cores = all_cores_active.merge(all_cores_passive);

    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_0, 4, src_tensor.dtype());
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_1, B*sizeof(int32_t), B*sizeof(int32_t), tt::DataFormat::Int32);
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_16, 4, src_tensor.dtype());
    MakeCircularBuffer(program, all_cores, tt::CBIndex::c_17, 4, output_tensor.dtype());

    std::map<std::string, std::string> defines;
    defines["FREQ_BASE"] = std::to_string(freq_base);
    defines["FREQ_BASE_LOG"] = std::to_string(std::log(freq_base));
    if(attn_factor != 1.f) {
        defines["ATTN_FACTOR"] = to_string_precise(attn_factor);
    }
    if(freq_scale != 1.f) {
        defines["FREQ_SCALE"] = to_string_precise(freq_scale);
        defines["LOG_1_FREQ_SCALE"] = to_string_precise(std::log(1.0f / freq_scale));
    }
    if(ext_factor != 0.f) {
        defines["EXT_FACTOR"] = to_string_precise(ext_factor);
        auto corr_dims = rope_yarn_corr_dims(D_active, n_ctx_orig, freq_base, beta_fast, beta_slow);
        if(std::isinf(corr_dims[0]) || std::isnan(corr_dims[0])) {
            corr_dims[0] = 0;
        }
        if(std::isinf(corr_dims[1]) || std::isnan(corr_dims[1])) {
            corr_dims[1] = 0;
        }
        defines["CORR_DIMS0"] = to_string_precise(corr_dims[0]);
        defines["CORR_DIMS1"] = to_string_precise(corr_dims[1]);
    }
    // defines["INV_D_ACTIVE_2"] = float(2.f / D_active); // don't know why this make things slower.


    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src).append_to(reader_compile_time_args);
    TensorAccessorArgs(*idxs).append_to(reader_compile_time_args);
    KernelHandle reader = CreateKernel(program, "../ttrope/kernels/reader.cpp", all_cores, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::RISCV_0_default,
        .compile_args = reader_compile_time_args
    });

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst).append_to(writer_compile_time_args);
    KernelHandle writer = CreateKernel(program, "../ttrope/kernels/writer.cpp", all_cores, DataMovementConfig{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::RISCV_1_default,
        .compile_args = writer_compile_time_args
    });

    KernelHandle compute = CreateKernel(program, "../ttrope/kernels/compute.cpp", all_cores, ComputeConfig{
        .fp32_dest_acc_en = true,
        .defines = defines
    });

    uint32_t active_id = 0;
    uint32_t passive_id = 0;
    for(const auto& range : all_cores.ranges()) {
        for(const auto& core : range) {
            uint32_t active_size = 0;
            uint32_t passive_size = 0;

            if(core_group_1_active.contains(core)) {
                active_size = work_per_core1_active;
            }
            else if(core_group_2_active.contains(core)) {
                active_size = work_per_core2_active;
            }

            if(core_group_1_passive.contains(core)) {
                passive_size = work_per_core1_passive;
            }
            else if(core_group_2_passive.contains(core)) {
                passive_size = work_per_core2_passive;
            }

            SetRuntimeArgs(program, reader, core, std::vector<uint32_t>{(uint32_t)src->address(), D_activet, Dt, Nt, (uint32_t)idxs->address(), B, active_id, active_id+active_size, passive_id, passive_id+passive_size, N});
            SetRuntimeArgs(program, compute, core, std::vector<uint32_t>{D_activet, Dt, Nt, B, active_id, active_id+active_size, N});
            SetRuntimeArgs(program, writer, core, std::vector<uint32_t>{(uint32_t)dst->address(), D_activet, Dt, Nt, B, active_id, active_id+active_size, passive_id, passive_id+passive_size});

            active_id += active_size;
            passive_id += passive_size;
        }
    }

    auto override_runtime_args_callback = [reader, writer, all_cores](
                                                  const void* operation,
                                                  Program& program,
                                                  const std::vector<Tensor>& input_tensors,
                                                  const std::vector<std::optional<const Tensor>>&,
                                                  const std::vector<Tensor>& output_tensors) {
            auto src_buffer = input_tensors.at(0).buffer();
            auto idx_buffer = input_tensors.at(1).buffer();
            auto dst_buffer = output_tensors.at(0).buffer();

            for(const auto& range : all_cores.ranges()) {
                for (const auto& core : range) {
                    {
                        auto& runtime_args = GetRuntimeArgs(program, reader, core);
                        runtime_args[0] = src_buffer->address();
                        runtime_args[4] = idx_buffer->address();
                    }

                    {
                        auto& runtime_args = GetRuntimeArgs(program, writer, core);
                        runtime_args[0] = dst_buffer->address();
                    }
                }
            }
        };

        return {std::move(program), override_runtime_args_callback};

}

int main()
{
    auto device = ttnn::open_mesh_device(0);
    auto src = ttnn::ones(ttnn::Shape({1, 1, 64}), DataType::FLOAT32, Layout::TILE, *device);
    src = ttnn::typecast(src, DataType::BFLOAT16);
    auto idx = ttnn::ones(ttnn::Shape({1}), DataType::INT32, Layout::ROW_MAJOR, *device);
    auto res = ttggml::rope(src, idx, 64);
    std::cout << res.write_to_string() << std::endl;

    device->close();
}
