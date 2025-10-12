#include <cstdint>

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t n_tiles_width_active = get_arg_val<uint32_t>(1);
    uint32_t n_tiles_width = get_arg_val<uint32_t>(2);
    uint32_t n_tiles_height = get_arg_val<uint32_t>(3);
    uint32_t idx_addr = get_arg_val<uint32_t>(4);
    uint32_t batch_size = get_arg_val<uint32_t>(5);
    uint32_t active_begin = get_arg_val<uint32_t>(6);
    uint32_t active_end = get_arg_val<uint32_t>(7);
    uint32_t passive_begin = get_arg_val<uint32_t>(8);
    uint32_t passive_end = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_bypass = tt::CBIndex::c_17;
    const uint32_t tile_size_bytes = get_tile_size(cb_in0);
    constexpr auto src_args = TensorAccessorArgs<0>();
    const auto src = TensorAccessor(src_args, src_addr, tile_size_bytes);

    constexpr auto idx_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const uint32_t idx_page_size_bytes = get_tile_size(cb_in1);
    const auto idx = TensorAccessor(idx_args, idx_addr, n_tiles_height*32*sizeof(int32_t));

    uint32_t last_h = (uint32_t)-1;
    for(uint32_t active_id=active_begin; active_id<active_end; active_id++) {
        uint32_t h = active_id / (n_tiles_width_active/2);
        uint32_t w = active_id % (n_tiles_width_active/2);

        if(last_h != h) {
            uint32_t batch_active_tiles_wh = (n_tiles_width_active/2) * n_tiles_height;
            uint32_t b = active_id / batch_active_tiles_wh;
            cb_reserve_back(cb_in1, 1);
            uint32_t cb_idx_addr = get_write_ptr(cb_in1);
            uint64_t read_addr = idx.get_noc_addr(b, 32*sizeof(int)*(h%n_tiles_height));
            noc_async_read(read_addr, cb_idx_addr, 32*sizeof(int));
            noc_async_read_barrier();
            cb_push_back(cb_in1, 1);
            last_h = h;
        }

        cb_reserve_back(cb_in0, 2);
        uint32_t cb_src_addr = get_write_ptr(cb_in0);
        uint32_t tile_idx =  h * n_tiles_width + w;
        noc_async_read_tile(tile_idx, src, cb_src_addr);
        uint32_t tile_idx2 = h * n_tiles_width + (w + n_tiles_width_active/2);
        noc_async_read_tile(tile_idx2, src, cb_src_addr + tile_size_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_in0, 2);
    }

    uint32_t n_tiles_width_passive = n_tiles_width - n_tiles_width_active;
    for(uint32_t passive_id = passive_begin; passive_id < passive_end; passive_id++) {
        uint32_t h = passive_id / n_tiles_width_passive;
        uint32_t w = passive_id % n_tiles_width_passive + n_tiles_width_active;
        uint32_t tile_idx = h * n_tiles_width + w;
        cb_reserve_back(cb_bypass, 1);
        uint32_t cb_bypass_addr = get_write_ptr(cb_bypass);
        noc_async_read_tile(tile_idx, src, cb_bypass_addr);
        noc_async_read_barrier();
        cb_push_back(cb_bypass, 1);
    }

}
