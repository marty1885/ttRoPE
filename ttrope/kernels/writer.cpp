#include <cstdint>

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles_width_active = get_arg_val<uint32_t>(1);
    uint32_t n_tiles_width = get_arg_val<uint32_t>(2);
    uint32_t n_tiles_height = get_arg_val<uint32_t>(3);
    uint32_t sram_tmp_tile_addr = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    const uint32_t tile_size_bytes = get_tile_size(cb_out0);
    constexpr auto dst_args = TensorAccessorArgs<0>();
    const auto dst = TensorAccessor(dst_args, dst_addr, tile_size_bytes);


    for(uint32_t h = 0; h < n_tiles_height; h++) {
        for(uint32_t w = 0; w < n_tiles_width; w++) {
            uint32_t tile_idx = h * n_tiles_width + w;
            cb_wait_front(cb_out0, 1);
            uint32_t cb_out_addr = get_read_ptr(cb_out0);
            noc_async_write_tile(tile_idx, dst, cb_out_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_out0, 1);
        }
    }
}
