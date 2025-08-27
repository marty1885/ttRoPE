#include <cstdint>

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles_width_active = get_arg_val<uint32_t>(1);
    uint32_t n_tiles_width = get_arg_val<uint32_t>(2);
    uint32_t n_tiles_height = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_bypass = tt::CBIndex::c_17;
    const uint32_t tile_size_bytes = get_tile_size(cb_in0);
    constexpr auto src_args = TensorAccessorArgs<0>();
    const auto src = TensorAccessor(src_args, src_addr, tile_size_bytes);


    for(uint32_t h = 0; h < n_tiles_height; h++) {
        for(uint32_t w = 0; w < n_tiles_width_active/2; w++) {
            cb_reserve_back(cb_in0, 2);
            uint32_t cb_src_addr = get_write_ptr(cb_in0);
            uint32_t tile_idx = h * n_tiles_width + w;
            noc_async_read_tile(tile_idx, src, cb_src_addr);
            uint32_t tile_idx2 = h * n_tiles_width + (w + n_tiles_width_active/2);
            noc_async_read_tile(tile_idx2, src, cb_src_addr + tile_size_bytes);
            noc_async_read_barrier();
            cb_push_back(cb_in0, 2);
        }

        for(uint32_t w = n_tiles_width_active; w < n_tiles_width; w++) {
            cb_reserve_back(cb_bypass, 1);
            uint32_t cb_bypass_addr = get_write_ptr(cb_bypass);
            uint32_t tile_idx = h * n_tiles_width + w;
            noc_async_read_tile(tile_idx, src, cb_bypass_addr);
            noc_async_read_barrier();
            cb_push_back(cb_bypass, 1);
        }
    }
}
