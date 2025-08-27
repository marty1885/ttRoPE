#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/eltwise_unary/trigonometry.h"
#include <array>

#ifdef TRISC_MATH
using namespace sfpi;

// Implemented algorithm exp_f24 from https://ieeexplore.ieee.org/document/9810030
inline vFloat vector_exp(sfpi::vFloat val) {
    sfpi::vFloat y = 0.0f;
    // Intermediary values can overflow if input value is below -88.0f, which leads to output increasing again instead
    // of staying at 0. This overflow happens when `log2(e) * val < 127.0f`, which correspond to `val < 88.0f`
    v_if(val > -88.0f) {
        // The paper relies on the following formula (c.f. Section 2 and 3 of paper):
        // z = (bias + x * factor * N_m; where:
        // factor = 0x00b8aa3b (computed through log(e))
        // bias = 0x3f800000
        sfpi::vInt z = sfpu::_float_to_int32_(val * sfpi::vFloat(0x00b8aa3b) + sfpi::vFloat(0x3f800000));
        sfpi::vInt zii = exexp(sfpi::reinterpret<sfpi::vFloat>(z));         // Extract exponent
        sfpi::vInt zif = sfpi::exman9(sfpi::reinterpret<sfpi::vFloat>(z));  // Extract mantissa

        // Polynomial coefficients for approximation of exp on [1; 2]
        vFloat POLY_D1;
        vInt POLY_D2;
        vInt POLY_D3;

        v_if(zif > 0x00600000) {
            // Fourth segment (highest values of the mantissa)
            POLY_D1 = 0.52496276e-7f;
            POLY_D2 = 0x81354a;
            POLY_D3 = 0x10a440;
        }
        v_elseif(zif > 0x00400000) {
            // Third segment
            POLY_D1 = 0.4414393e-7f;
            POLY_D2 = 0xcdf4b4;
            POLY_D3 = 0x3e4d6;
        }
        v_elseif(zif > 0x00200000) {
            // Second segment
            POLY_D1 =0.37120473e-7f;
            POLY_D2 = 0x1113a74;
            POLY_D3 = 0x9f16;
        }
        v_else {
            // First segment
            POLY_D1 = 0.31214472e-7f;
            POLY_D2 = 0x151d842;
            // Note: The original C code has a float constant here WTF?
            // This following number is found via a guess the author did something
            // stupid and added a floating point notation after the real value.
            POLY_D3 = 328;
        }
        v_endif;

        sfpi::vFloat d1 = sfpi::vFloat(POLY_D1);
        sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(POLY_D2) + zif, 0);
        sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(POLY_D3) + zif, 0);
        d2 = d1 * d2;
        zif = sfpu::_float_to_int32_(d2 * d3);

        // Restore exponent
        zii = sfpi::reinterpret<sfpi::vInt>(
            sfpi::setexp(sfpi::reinterpret<sfpi::vFloat>(zif), 127U + zii));  // restore exponent

        y = sfpi::reinterpret<sfpi::vFloat>(zii);
    }
    v_endif;
    return y;
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

        // RoPE formula freq = exp(-exponent * log(10000.0f)).
        // The angle is calculated as angle = (pos * freq) / PI.
        // To improve FPU accuracy, we can rewrite this as:
        // angle = pos * exp(-exponent * log(10000) + log(1/PI))
        vFloat term_to_exp = -exponent * 9.21034037f; // log(10000) = 9.21034037, log(1/PI) = -1.14472988585
        vFloat freq = vector_exp(term_to_exp);

        // Standard RoPE math
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
