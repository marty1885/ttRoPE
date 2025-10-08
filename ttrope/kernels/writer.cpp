#include <cinttypes>
#include <cstdint>

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles_width_active = get_arg_val<uint32_t>(1);
    uint32_t n_tiles_width = get_arg_val<uint32_t>(2);
    uint32_t n_tiles_height = get_arg_val<uint32_t>(3);
    uint32_t batch_size = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    constexpr uint32_t cb_bypass = tt::CBIndex::c_17;
    const uint32_t tile_size_bytes = get_tile_size(cb_out0);
    constexpr auto dst_args = TensorAccessorArgs<0>();
    const auto dst = TensorAccessor(dst_args, dst_addr, tile_size_bytes);


    uint32_t batch_tiles_wh = n_tiles_width * n_tiles_height;
    for(uint32_t b = 0; b < batch_size; b++) {
        for(uint32_t h = 0; h < n_tiles_height; h++) {
            for(uint32_t w = 0; w < n_tiles_width_active/2; w++) {
                cb_wait_front(cb_out0, 2);
                uint32_t tile_idx = b * batch_tiles_wh + h * n_tiles_width + w;
                uint32_t tile_idx2 =b * batch_tiles_wh+  h * n_tiles_width + (w + n_tiles_width_active/2);
                uint32_t cb_out_addr = get_read_ptr(cb_out0);
                noc_async_write_tile(tile_idx, dst, cb_out_addr);
                noc_async_write_tile(tile_idx2, dst, cb_out_addr + tile_size_bytes);
                noc_async_write_barrier();
                cb_pop_front(cb_out0, 2);
            }

            for(uint32_t w = n_tiles_width_active; w < n_tiles_width; w++) {
                cb_wait_front(cb_bypass, 1);
                uint32_t cb_bypass_addr = get_read_ptr(cb_bypass);
                uint32_t tile_idx = b * batch_tiles_wh + h * n_tiles_width + w;
                noc_async_write_tile(tile_idx, dst, cb_bypass_addr);
                noc_async_write_barrier();
                cb_pop_front(cb_bypass, 1);
            }
        }
    }
}
