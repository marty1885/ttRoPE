#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/eltwise_unary/trigonometry.h"
#include <array>

#ifdef TRISC_MATH
using namespace sfpi;

inline vFloat approx_exp(vFloat x)
{
    return ckernel::sfpu::_sfpu_exp_21f_<true>(x);
}

inline vFloat vector_sin_phase(vFloat x)
{
    vFloat v = x;
    vInt whole_v = float_to_int16(v, 0);
    v -= int32_to_float(whole_v, 0);

    v = ckernel::sfpu::sfpu_sinpi<false>(v);
    v_if(whole_v & 1) { v = -v; }
    v_endif;
    return v;
}

inline void rope_face(int pos, int D, int vec_offset)
{
    float inv_d = 1.f/D;
    for (int i = 0; i < 4; i++) {
        vFloat block_lane_id = int32_to_float((vConstTileId & 15) + vec_offset); // No mod operator on SFPI, use bit hack
        vFloat exponent = block_lane_id * inv_d;

        // Standard RoPE forumla freq = 1.f / pow(10000, exponent). We don't have pow(x, y) in SFPU. Rewrite formula
        // freq = 1.f / exp(exponent * log(10000))
        // freq = exp(-exponent * log(10000.0f))
        vFloat term_to_exp = -exponent * 9.21034037f;
        vFloat freq = approx_exp(term_to_exp);

        // Standard RoPE math
        // FIXME: Somehow accuracy issue here. `freq` is fine. But after multipling with `freq` and everything blows up.
        vFloat angle = ckernel::sfpu::FRAC_1_PI * int32_to_float(pos) * freq;
        vFloat sin_angle = vector_sin_phase(angle);
        vFloat cos_angle = vector_sin_phase(0.5f - angle);

        // Thanks that dst interleaves lanes by default
        vFloat x = dst_reg[i*2];
        vFloat y = dst_reg[i*2+1];

        dst_reg[i*2] = x * cos_angle - y * sin_angle;
        dst_reg[i*2+1] = x * sin_angle + y * cos_angle;
        // result[offset + i] = x * cos_angle - y * sin_angle;
        // result[offset + i + 1] = x * sin_angle + y * cos_angle;
    }
}

inline void rope_tile(int pos, int D, int vec_offset)
{

    math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(0);
    math::set_addr_mod_base();
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
;
    for (int face = 0; face < 4; face++) {
        rope_face(pos, D, vec_offset + ((face % 2 == 0) ? 0 : 16));
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    }

    math::clear_dst_reg_addr();
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    math::clear_addr_mod_base();
}
#endif

namespace NAMESPACE {
void MAIN {

    uint32_t n_tiles_width_active = get_arg_val<uint32_t>(0);
    uint32_t n_tiles_width = get_arg_val<uint32_t>(1);
    uint32_t n_tiles_height = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_16);
    exp_tile_init();
    for(uint32_t i = 0; i < n_tiles_height; i++) {
        for(uint32_t j = 0; j < n_tiles_width; j++) {
            cb_wait_front(cb_in0, 1);
            tile_regs_acquire();
            copy_tile_init(cb_in0);
            copy_tile(cb_in0, 0, 0);
            if(j < n_tiles_width_active) {
                MATH(rope_tile(1000, n_tiles_width * 32, j * 32));
            }
            tile_regs_commit();
            tile_regs_wait();

            cb_reserve_back(cb_out0, 1);
            pack_reconfig_data_format(cb_out0);
            pack_tile(0, cb_out0);
            tile_regs_release();
            cb_push_back(cb_out0, 1);
            cb_pop_front(cb_in0, 1);
        }
    }

}
}
