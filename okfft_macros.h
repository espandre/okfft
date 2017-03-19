/*
This file is part of OKFFT

BSD 3-Clause License

Copyright (c) 2012, 2013, Anthony M. Blake <amb@anthonix.com>
Copyright (c) 2012, The University of Waikato
Copyright (c) 2015, Jukka Ojanen <jukka.ojanen@kolumbus.fi>
Copyright (c) 2017, Espen Andreassen <espandre@gmail.com>

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of the organization nor the
  names of its contributors may be used to endorse or promote products
  derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL ANTHONY M. BLAKE BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once
#include "okfft.h"

#define okfft_sse_swap_pairs(x)   _mm_shuffle_ps(x, x, _MM_SHUFFLE(2, 3, 0, 1))
#define okfft_sse_dup_re(x)       _mm_shuffle_ps(x, x, _MM_SHUFFLE(2, 2, 0, 0))
#define okfft_sse_dup_im(x)       _mm_shuffle_ps(x, x, _MM_SHUFFLE(3, 3, 1, 1))
#define okfft_sse_swap_sign(x)    _mm_xor_ps(x, sse_sign_mask)
#define okfft_sse_unpack_lo(x, y) _mm_shuffle_ps(x, y, _MM_SHUFFLE(1, 0, 1, 0))
#define okfft_sse_unpack_hi(x, y) _mm_shuffle_ps(x, y, _MM_SHUFFLE(3, 2, 3, 2))
#define okfft_sse_blend(x, y)     _mm_shuffle_ps(x, y, _MM_SHUFFLE(3, 2, 1, 0))

#define okfft_sse_store4(base, r0, r1, r2, r3)              \
{                                                           \
    _mm_store_ps(base +  0, r0);                            \
    _mm_store_ps(base +  4, r1);                            \
    _mm_store_ps(base +  8, r2);                            \
    _mm_store_ps(base + 12, r3);                            \
}

#define OKFFT_SSE_KN(re, im, r0, r1, r2, r3)                \
{                                                           \
    __m128 uk = r0, uk2 = r1;                               \
    __m128 r2r = _mm_mul_ps(re, r2);                        \
    __m128 r3r = _mm_mul_ps(re, r3);                        \
                                                            \
    r2 = okfft_sse_swap_pairs(r2);                          \
    r3 = okfft_sse_swap_pairs(r3);                          \
                                                            \
    __m128 r2i = _mm_mul_ps(im, r2);                        \
    __m128 r3i = _mm_mul_ps(im, r3);                        \
                                                            \
    __m128 zk_p = _mm_sub_ps(r2r, r2i);                     \
    __m128 zk_n = _mm_add_ps(r3r, r3i);                     \
                                                            \
    __m128 zk   = _mm_add_ps(zk_p, zk_n);                   \
    __m128 zk_d = _mm_sub_ps(zk_p, zk_n);                   \
                                                            \
    r2 = _mm_sub_ps(uk, zk);                                \
    r0 = _mm_add_ps(uk, zk);                                \
                                                            \
    zk_d = okfft_sse_swap_sign(zk_d);                       \
    zk_d = okfft_sse_swap_pairs(zk_d);                      \
                                                            \
    r3 = _mm_add_ps(uk2, zk_d);                             \
    r1 = _mm_sub_ps(uk2, zk_d);                             \
}

#define OKFFT_SSE_KNKN(re0, im0, re1, im1, r00, r10, r20, r30, r01, r11, r21, r31)  \
{                                                           \
    __m128 uk0  = r00, uk20 = r10;                          \
    __m128 uk1  = r01, uk21 = r11;                          \
    __m128 r20r = _mm_mul_ps(re0, r20);                     \
    __m128 r21r = _mm_mul_ps(re1, r21);                     \
    __m128 r30r = _mm_mul_ps(re0, r30);                     \
    __m128 r31r = _mm_mul_ps(re1, r31);                     \
                                                            \
    r20 = okfft_sse_swap_pairs(r20);                        \
    r21 = okfft_sse_swap_pairs(r21);                        \
    r30 = okfft_sse_swap_pairs(r30);                        \
    r31 = okfft_sse_swap_pairs(r31);                        \
                                                            \
    __m128 r20i = _mm_mul_ps(im0, r20);                     \
    __m128 r21i = _mm_mul_ps(im1, r21);                     \
    __m128 r30i = _mm_mul_ps(im0, r30);                     \
    __m128 r31i = _mm_mul_ps(im1, r31);                     \
                                                            \
    __m128 zk_p0 = _mm_sub_ps(r20r, r20i);                  \
    __m128 zk_p1 = _mm_sub_ps(r21r, r21i);                  \
    __m128 zk_n0 = _mm_add_ps(r30r, r30i);                  \
    __m128 zk_n1 = _mm_add_ps(r31r, r31i);                  \
                                                            \
    __m128 zk0   = _mm_add_ps(zk_p0, zk_n0);                \
    __m128 zk1   = _mm_add_ps(zk_p1, zk_n1);                \
    __m128 zk_d0 = _mm_sub_ps(zk_p0, zk_n0);                \
    __m128 zk_d1 = _mm_sub_ps(zk_p1, zk_n1);                \
                                                            \
    r20 = _mm_sub_ps(uk0, zk0);                             \
    r21 = _mm_sub_ps(uk1, zk1);                             \
    r00 = _mm_add_ps(uk0, zk0);                             \
    r01 = _mm_add_ps(uk1, zk1);                             \
                                                            \
    zk_d0 = okfft_sse_swap_sign(zk_d0);                     \
    zk_d1 = okfft_sse_swap_sign(zk_d1);                     \
    zk_d0 = okfft_sse_swap_pairs(zk_d0);                    \
    zk_d1 = okfft_sse_swap_pairs(zk_d1);                    \
                                                            \
    r30 = _mm_add_ps(uk20, zk_d0);                          \
    r31 = _mm_add_ps(uk21, zk_d1);                          \
    r10 = _mm_sub_ps(uk20, zk_d0);                          \
    r11 = _mm_sub_ps(uk21, zk_d1);                          \
}

#define OKFFT_SSE_X4(data, lut)                             \
{                                                           \
    __m128 r00 = _mm_load_ps(data +  0);                    \
    __m128 r01 = _mm_load_ps(data +  4);                    \
    __m128 r10 = _mm_load_ps(data +  8);                    \
    __m128 r11 = _mm_load_ps(data + 12);                    \
    __m128 r20 = _mm_load_ps(data + 16);                    \
    __m128 r21 = _mm_load_ps(data + 20);                    \
    __m128 r30 = _mm_load_ps(data + 24);                    \
    __m128 r31 = _mm_load_ps(data + 28);                    \
                                                            \
    __m128 re0 = _mm_load_ps(lut +  0);                     \
    __m128 im0 = _mm_load_ps(lut +  4);                     \
    __m128 re1 = _mm_load_ps(lut +  8);                     \
    __m128 im1 = _mm_load_ps(lut + 12);                     \
                                                            \
    OKFFT_SSE_KNKN(re0, im0, re1, im1,  r00, r10, r20, r30, \
                                        r01, r11, r21, r31);\
                                                            \
    _mm_store_ps(data + 16, r20);                           \
    _mm_store_ps(data + 20, r21);                           \
    _mm_store_ps(data +  0, r00);                           \
    _mm_store_ps(data +  4, r01);                           \
                                                            \
    _mm_store_ps(data + 24, r30);                           \
    _mm_store_ps(data + 28, r31);                           \
    _mm_store_ps(data +  8, r10);                           \
    _mm_store_ps(data + 12, r11);                           \
}

#define OKFFT_SSE_X8(N, data, p_lut)                        \
{                                                           \
    size_t OFFS = N >> 2;                                   \
    const float *__restrict lut = (p_lut);                  \
    float *__restrict d0 = data + (0 * OFFS);               \
    float *__restrict d1 = data + (1 * OFFS);               \
    float *__restrict d2 = data + (2 * OFFS);               \
    float *__restrict d3 = data + (3 * OFFS);               \
    float *__restrict d4 = data + (4 * OFFS);               \
    float *__restrict d5 = data + (5 * OFFS);               \
    float *__restrict d6 = data + (6 * OFFS);               \
    float *__restrict d7 = data + (7 * OFFS);               \
                                                            \
    for (size_t i = 0; i < (N >> 4); i++)                   \
    {                                                       \
        __m128 r0 = _mm_load_ps(d0);                        \
        __m128 r1 = _mm_load_ps(d1);                        \
        __m128 r2 = _mm_load_ps(d2);                        \
        __m128 r3 = _mm_load_ps(d3);                        \
        __m128 r4 = _mm_load_ps(d4);                        \
        __m128 r5 = _mm_load_ps(d5);                        \
        __m128 r6 = _mm_load_ps(d6);                        \
        __m128 r7 = _mm_load_ps(d7);                        \
                                                            \
        __m128 re = _mm_load_ps(lut + 0);                   \
        __m128 im = _mm_load_ps(lut + 4);                   \
                                                            \
        OKFFT_SSE_KN(re, im, r0, r1, r2, r3);               \
                                                            \
        __m128 re0 = _mm_load_ps(lut +  8);                 \
        __m128 im0 = _mm_load_ps(lut + 12);                 \
        __m128 re1 = _mm_load_ps(lut + 16);                 \
        __m128 im1 = _mm_load_ps(lut + 20);                 \
                                                            \
        OKFFT_SSE_KNKN(re0, im0, re1, im1,  r0, r2, r4, r6, \
                                            r1, r3, r5, r7);\
                                                            \
        _mm_store_ps(d0, r0);                               \
        _mm_store_ps(d1, r1);                               \
        _mm_store_ps(d2, r2);                               \
        _mm_store_ps(d3, r3);                               \
        _mm_store_ps(d4, r4);                               \
        _mm_store_ps(d5, r5);                               \
        _mm_store_ps(d6, r6);                               \
        _mm_store_ps(d7, r7);                               \
                                                            \
        lut += 24;                                          \
        d0 += 4; d1 += 4; d2 += 4; d3 += 4;                 \
        d4 += 4; d5 += 4; d6 += 4; d7 += 4;                 \
    }                                                       \
}

#define OKFFT_SSE_TX2(a, b)                                 \
{                                                           \
    __m128 q0 = okfft_sse_unpack_lo(a, b);                  \
    __m128 q1 = okfft_sse_unpack_hi(a, b);                  \
    a = q0;                                                 \
    b = q1;                                                 \
}

#define OKFFT_SSE_L2(i0, i1, i2, i3, r0, r1, r2, r3)        \
{                                                           \
    __m128 t0 = _mm_load_ps(i0);                            \
    __m128 t1 = _mm_load_ps(i1);                            \
    __m128 t2 = _mm_load_ps(i2);                            \
    __m128 t3 = _mm_load_ps(i3);                            \
                                                            \
    r0 = _mm_add_ps(t0, t1);                                \
    r1 = _mm_sub_ps(t0, t1);                                \
    r2 = _mm_add_ps(t2, t3);                                \
    r3 = _mm_sub_ps(t2, t3);                                \
}

#define OKFFT_SSE_L4(i0, i1, i2, i3, r0, r1, r2, r3)        \
{                                                           \
    __m128 t0 = _mm_load_ps(i0);                            \
    __m128 t1 = _mm_load_ps(i1);                            \
    __m128 t2 = _mm_load_ps(i2);                            \
    __m128 t3 = _mm_load_ps(i3);                            \
                                                            \
    __m128 t4 = _mm_add_ps(t0, t1);                         \
    __m128 t5 = _mm_sub_ps(t0, t1);                         \
    __m128 t6 = _mm_add_ps(t2, t3);                         \
    __m128 t7 = _mm_sub_ps(t2, t3);                         \
                                                            \
    t7 = okfft_sse_swap_sign(t7);                           \
    t7 = okfft_sse_swap_pairs(t7);                          \
                                                            \
    r0 = _mm_add_ps(t4, t6);                                \
    r2 = _mm_sub_ps(t4, t6);                                \
    r1 = _mm_sub_ps(t5, t7);                                \
    r3 = _mm_add_ps(t5, t7);                                \
}

#define OKFFT_SSE_L44(i0, i1, i2, i3, r0, r1, r2, r3)       \
{                                                           \
    __m128 t0 = _mm_load_ps(i0);                            \
    __m128 t1 = _mm_load_ps(i1);                            \
    __m128 t2 = _mm_load_ps(i2);                            \
    __m128 t3 = _mm_load_ps(i3);                            \
                                                            \
    __m128 t4 = _mm_add_ps(t0, t1);                         \
    __m128 t5 = _mm_sub_ps(t0, t1);                         \
    __m128 t6 = _mm_add_ps(t2, t3);                         \
    __m128 t7 = _mm_sub_ps(t2, t3);                         \
                                                            \
    t7 = okfft_sse_swap_sign(t7);                           \
    t7 = okfft_sse_swap_pairs(t7);                          \
                                                            \
    t0 = _mm_add_ps(t4, t6);                                \
    t2 = _mm_sub_ps(t4, t6);                                \
    t1 = _mm_sub_ps(t5, t7);                                \
    t3 = _mm_add_ps(t5, t7);                                \
                                                            \
    OKFFT_SSE_TX2(t0, t1);                                  \
    OKFFT_SSE_TX2(t2, t3);                                  \
                                                            \
    r0 = t0;                                                \
    r2 = t1;                                                \
    r1 = t2;                                                \
    r3 = t3;                                                \
}

#define OKFFT_SSE_L42(i0, i1, i2, i3, r0, r1, r2, r3)       \
{                                                           \
    __m128 t0 = _mm_load_ps(i0);                            \
    __m128 t1 = _mm_load_ps(i1);                            \
    __m128 t6 = _mm_load_ps(i2);                            \
    __m128 t7 = _mm_load_ps(i3);                            \
                                                            \
    __m128 t2 = okfft_sse_blend(t6, t7);                    \
    __m128 t3 = okfft_sse_blend(t7, t6);                    \
                                                            \
    __m128 t4 = _mm_add_ps(t0, t1);                         \
    __m128 t5 = _mm_sub_ps(t0, t1);                         \
           t6 = _mm_add_ps(t2, t3);                         \
           t7 = _mm_sub_ps(t2, t3);                         \
                                                            \
    r2 = okfft_sse_unpack_hi(t4, t5);                       \
    r3 = okfft_sse_unpack_hi(t6, t7);                       \
                                                            \
    t7 = okfft_sse_swap_sign(t7);                           \
    t7 = okfft_sse_swap_pairs(t7);                          \
                                                            \
    t0 = _mm_add_ps(t4, t6);                                \
    t2 = _mm_sub_ps(t4, t6);                                \
    t1 = _mm_sub_ps(t5, t7);                                \
    t3 = _mm_add_ps(t5, t7);                                \
                                                            \
    r0 = okfft_sse_unpack_lo(t0, t1);                       \
    r1 = okfft_sse_unpack_lo(t2, t3);                       \
}

#define OKFFT_SSE_L24(i0, i1, i2, i3, r0, r1, r2, r3)       \
{                                                           \
    __m128 t0 = _mm_load_ps(i0);                            \
    __m128 t1 = _mm_load_ps(i1);                            \
    __m128 t2 = _mm_load_ps(i2);                            \
    __m128 t3 = _mm_load_ps(i3);                            \
                                                            \
    __m128 t4 = _mm_add_ps(t0, t1);                         \
    __m128 t5 = _mm_sub_ps(t0, t1);                         \
    __m128 t6 = _mm_add_ps(t2, t3);                         \
    __m128 t7 = _mm_sub_ps(t2, t3);                         \
                                                            \
    r0 = okfft_sse_unpack_lo(t4, t5);                       \
    r1 = okfft_sse_unpack_lo(t6, t7);                       \
                                                            \
    t5 = okfft_sse_swap_sign(t5);                           \
    t5 = okfft_sse_swap_pairs(t5);                          \
                                                            \
    t0 = _mm_add_ps(t6, t4);                                \
    t2 = _mm_sub_ps(t6, t4);                                \
    t1 = _mm_sub_ps(t7, t5);                                \
    t3 = _mm_add_ps(t7, t5);                                \
                                                            \
    r3 = okfft_sse_unpack_hi(t0, t1);                       \
    r2 = okfft_sse_unpack_hi(t2, t3);                       \
}

#define OKFFT_SSE_K0(r0, r1, r2, r3)                        \
{                                                           \
    __m128 t0 = r0;                                         \
    __m128 t1 = r1;                                         \
                                                            \
    __m128  t2 = _mm_add_ps(r2, r3);                        \
    __m128  t3 = _mm_sub_ps(r2, r3);                        \
            t3 = okfft_sse_swap_sign(t3);                   \
            t3 = okfft_sse_swap_pairs(t3);                  \
                                                            \
    r0 = _mm_add_ps(t0, t2);                                \
    r2 = _mm_sub_ps(t0, t2);                                \
    r1 = _mm_sub_ps(t1, t3);                                \
    r3 = _mm_add_ps(t1, t3);                                \
}

#define OKFFT_SSE_LEAF_EE(out, os, in, is)                  \
{                                                           \
    const float *__restrict LUT = sse_constants;            \
    __m128 r0, r1, r2, r3, r4, r5, r6, r7;                  \
    float *__restrict out0 = out + os[0];                   \
    float *__restrict out1 = out + os[1];                   \
                                                            \
    OKFFT_SSE_L4(in + is[0], in + is[1], in + is[2], in + is[3], r0, r1, r2, r3);   \
    OKFFT_SSE_L2(in + is[4], in + is[5], in + is[6], in + is[7], r4, r5, r6, r7);   \
                                                            \
    __m128 re = _mm_load_ps(LUT + 0);                       \
    __m128 im = _mm_load_ps(LUT + 4);                       \
    OKFFT_SSE_K0(r0, r2, r4, r6);                           \
    OKFFT_SSE_KN(re, im, r1, r3, r5, r7);                   \
                                                            \
    OKFFT_SSE_TX2(r0, r1);                                  \
    OKFFT_SSE_TX2(r2, r3);                                  \
    OKFFT_SSE_TX2(r4, r5);                                  \
    OKFFT_SSE_TX2(r6, r7);                                  \
                                                            \
    okfft_sse_store4(out0, r0, r2, r4, r6);                 \
    okfft_sse_store4(out1, r1, r3, r5, r7);                 \
}

#define OKFFT_SSE_LEAF_EO(out, os, in, is)                  \
{                                                           \
    const float *__restrict LUT = sse_constants;            \
    __m128 r0, r1, r2, r3, r4, r5, r6, r7;                  \
    float *__restrict out0 = out + os[0];                   \
    float *__restrict out1 = out + os[1];                   \
                                                            \
    OKFFT_SSE_L44(in + is[0], in + is[1], in + is[2], in + is[3], r0, r1, r2, r3);  \
    OKFFT_SSE_L24(in + is[4], in + is[5], in + is[6], in + is[7], r4, r5, r6, r7);  \
                                                            \
    okfft_sse_store4(out1, r2, r3, r7, r6);                 \
                                                            \
    __m128 re = _mm_load_ps(LUT +  8);                      \
    __m128 im = _mm_load_ps(LUT + 12);                      \
    OKFFT_SSE_KN(re, im, r0, r1, r4, r5);                   \
                                                            \
    okfft_sse_store4(out0, r0, r1, r4, r5);                 \
}

#define OKFFT_SSE_LEAF_OE(out, os, in, is)                  \
{                                                           \
    const float *__restrict LUT = sse_constants;            \
    __m128 r0, r1, r2, r3, r4, r5, r6, r7;                  \
    float *__restrict out0 = out + os[0];                   \
    float *__restrict out1 = out + os[1];                   \
                                                            \
    OKFFT_SSE_L42(in + is[0], in + is[1], in + is[2], in + is[3], r0, r1, r2, r3);  \
    OKFFT_SSE_L44(in + is[6], in + is[7], in + is[4], in + is[5], r4, r5, r6, r7);  \
                                                            \
    okfft_sse_store4(out0, r0, r1, r4, r5);                 \
                                                            \
    __m128 re = _mm_load_ps(LUT +  8);                      \
    __m128 im = _mm_load_ps(LUT + 12);                      \
    OKFFT_SSE_KN(re, im, r6, r7, r2, r3);                   \
                                                            \
    okfft_sse_store4(out1, r6, r7, r2, r3);                 \
}

#define OKFFT_SSE_LEAF_OO(out, os, in, is)                  \
{                                                           \
    __m128 r0, r1, r2, r3, r4, r5, r6, r7;                  \
    float *__restrict out0 = out + os[0];                   \
    float *__restrict out1 = out + os[1];                   \
                                                            \
    OKFFT_SSE_L44(in + is[0], in + is[1], in + is[2], in + is[3], r0, r1, r2, r3);  \
    OKFFT_SSE_L44(in + is[6], in + is[7], in + is[4], in + is[5], r4, r5, r6, r7);  \
                                                            \
    okfft_sse_store4(out0, r0, r1, r4, r5);                 \
    okfft_sse_store4(out1, r2, r3, r6, r7);                 \
}

#define OKFFT_SSE_LEAF_EE2(out, os, in, is)                 \
{                                                           \
    const float *__restrict LUT = sse_constants;            \
    __m128 r0, r1, r2, r3, r4, r5, r6, r7;                  \
    float *__restrict out0 = out + os[0];                   \
    float *__restrict out1 = out + os[1];                   \
                                                            \
    OKFFT_SSE_L4(in + is[6], in + is[7], in + is[4], in + is[5], r0, r1, r2, r3);   \
    OKFFT_SSE_L2(in + is[0], in + is[1], in + is[3], in + is[2], r4, r5, r6, r7);   \
                                                            \
    __m128 re = _mm_load_ps(LUT + 0);                       \
    __m128 im = _mm_load_ps(LUT + 4);                       \
    OKFFT_SSE_K0(r0, r2, r4, r6);                           \
    OKFFT_SSE_KN(re, im, r1, r3, r5, r7);                   \
                                                            \
    OKFFT_SSE_TX2(r0, r1);                                  \
    OKFFT_SSE_TX2(r2, r3);                                  \
    OKFFT_SSE_TX2(r4, r5);                                  \
    OKFFT_SSE_TX2(r6, r7);                                  \
                                                            \
    okfft_sse_store4(out0, r0, r2, r4, r6);                 \
    okfft_sse_store4(out1, r1, r3, r5, r7);                 \
}

#define OKFFT_SSE_FP_EVEN(i0, i1, p, p_out, p_in)           \
{                                                           \
    float *__restrict out = p_out;                          \
    const float *__restrict in = p_in;                      \
    const ptrdiff_t *__restrict is = p->is;                 \
    const ptrdiff_t *__restrict os = p->offsets;            \
                                                            \
    for (size_t i = i0; i > 0; --i)                         \
    {                                                       \
        OKFFT_SSE_LEAF_EE(out, os, in, is);                 \
        in += 4;                                            \
        os += 2;                                            \
    }                                                       \
                                                            \
    OKFFT_SSE_LEAF_EO(out, os, in, is);                     \
    in += 4;                                                \
    os += 2;                                                \
                                                            \
    for (size_t i = i1; i > 0; --i)                         \
    {                                                       \
        OKFFT_SSE_LEAF_OO(out, os, in, is);                 \
        in += 4;                                            \
        os += 2;                                            \
    }                                                       \
                                                            \
    for (size_t i = i1; i > 0; --i)                         \
    {                                                       \
        OKFFT_SSE_LEAF_EE2(out, os, in, is);                \
        in += 4;                                            \
        os += 2;                                            \
    }                                                       \
}

#define OKFFT_SSE_FP_ODD(i0, i1, p, p_out, p_in)            \
{                                                           \
    float *__restrict out = p_out;                          \
    const float *__restrict in = p_in;                      \
    const ptrdiff_t *__restrict is = p->is;                 \
    const ptrdiff_t *__restrict os = p->offsets;            \
                                                            \
    for (size_t i = i0; i > 0; --i)                         \
    {                                                       \
        OKFFT_SSE_LEAF_EE(out, os, in, is);                 \
        in += 4;                                            \
        os += 2;                                            \
    }                                                       \
                                                            \
    for (size_t i = i1; i > 0; --i)                         \
    {                                                       \
        OKFFT_SSE_LEAF_OO(out, os, in, is);                 \
        in += 4;                                            \
        os += 2;                                            \
    }                                                       \
                                                            \
    OKFFT_SSE_LEAF_OE(out, os, in, is);                     \
    in += 4;                                                \
    os += 2;                                                \
                                                            \
    for (size_t i = i1; i > 0; --i)                         \
    {                                                       \
        OKFFT_SSE_LEAF_EE2(out, os, in, is);                \
        in += 4;                                            \
        os += 2;                                            \
    }                                                       \
}

#define OKFFT_SSE_XF_32(data)                       \
{                                                   \
    OKFFT_SSE_X8(32, data, ws1);                    \
}

#define OKFFT_SSE_XF_64(data)                       \
{                                                   \
    OKFFT_SSE_X4(data +  0, ws);                    \
    OKFFT_SSE_X4(data + 64, ws);                    \
    OKFFT_SSE_X4(data + 96, ws);                    \
    OKFFT_SSE_X8(64, data, ws + (ws_is[2] << 1));   \
}

#define OKFFT_SSE_XF_128(data)                      \
{                                                   \
    OKFFT_SSE_X8(32, data +   0, ws1);              \
    OKFFT_SSE_X4(data + 64, ws);                    \
    OKFFT_SSE_X4(data + 96, ws);                    \
    OKFFT_SSE_X8(32, data + 128, ws1);              \
    OKFFT_SSE_X8(32, data + 192, ws1);              \
    OKFFT_SSE_X8(128, data, ws + (ws_is[3] << 1));  \
}

#define OKFFT_SSE_XF_256(data)                      \
{                                                   \
    OKFFT_SSE_XF_64(data);                          \
    OKFFT_SSE_XF_32(data + 2 * 64);                 \
    OKFFT_SSE_XF_32(data + 3 * 64);                 \
    OKFFT_SSE_XF_64(data + 4 * 64);                 \
    OKFFT_SSE_XF_64(data + 6 * 64);                 \
    OKFFT_SSE_X8(256, data, ws + (ws_is[4] << 1));  \
}

#define OKFFT_SSE_XF_512(data)                      \
{                                                   \
    OKFFT_SSE_XF_128(data);                         \
    OKFFT_SSE_XF_64(data +  2 * 128);               \
    OKFFT_SSE_XF_64(data +  3 * 128);               \
    OKFFT_SSE_XF_128(data + 4 * 128);               \
    OKFFT_SSE_XF_128(data + 6 * 128);               \
    OKFFT_SSE_X8(512, data, ws + (ws_is[5] << 1));  \
}

#define OKFFT_SSE_XF_1024(data)                     \
{                                                   \
    OKFFT_SSE_XF_256(data);                         \
    OKFFT_SSE_XF_128(data + 2 * 256);               \
    OKFFT_SSE_XF_128(data + 3 * 256);               \
    OKFFT_SSE_XF_256(data + 4 * 256);               \
    OKFFT_SSE_XF_256(data + 6 * 256);               \
    OKFFT_SSE_X8(1024, data, ws + (ws_is[6] << 1)); \
}

#ifdef OKFFT_HAS_AVX

#define okfft_avx_swap_pairs(x)   _mm256_shuffle_ps(x, x, _MM_SHUFFLE(2, 3, 0, 1))
#define okfft_avx_dup_re(x)       _mm256_shuffle_ps(x, x, _MM_SHUFFLE(2, 2, 0, 0))
#define okfft_avx_dup_im(x)       _mm256_shuffle_ps(x, x, _MM_SHUFFLE(3, 3, 1, 1))
#define okfft_avx_swap_sign(x)    _mm256_xor_ps(x, avx_sign_mask)
#define okfft_avx_unpack_lo(x, y) _mm256_shuffle_ps(x, y, _MM_SHUFFLE(1, 0, 1, 0))
#define okfft_avx_unpack_hi(x, y) _mm256_shuffle_ps(x, y, _MM_SHUFFLE(3, 2, 3, 2))
#define okfft_avx_blend(x, y)     _mm256_shuffle_ps(x, y, _MM_SHUFFLE(3, 2, 1, 0))

#define okfft_avx_store4(base0, base1, r0, r1, r2, r3)  \
{                                                       \
    __m256 q0 = _mm256_permute2f128_ps(r0, r1, 0x20);   \
    __m256 q1 = _mm256_permute2f128_ps(r2, r3, 0x20);   \
    __m256 q2 = _mm256_permute2f128_ps(r0, r1, 0x31);   \
    __m256 q3 = _mm256_permute2f128_ps(r2, r3, 0x31);   \
                                                        \
    _mm256_store_ps(base0 +  0, q0);                    \
    _mm256_store_ps(base0 +  8, q1);                    \
    _mm256_store_ps(base1 +  0, q2);                    \
    _mm256_store_ps(base1 +  8, q3);                    \
}

#define OKFFT_AVX_KN(re, im, r0, r1, r2, r3)                \
{                                                           \
    __m256 uk = r0, uk2 = r1;                               \
    __m256 r2r = _mm256_mul_ps(re, r2);                     \
    __m256 r3r = _mm256_mul_ps(re, r3);                     \
                                                            \
    r2 = okfft_avx_swap_pairs(r2);                          \
    r3 = okfft_avx_swap_pairs(r3);                          \
                                                            \
    __m256 r2i = _mm256_mul_ps(im, r2);                     \
    __m256 r3i = _mm256_mul_ps(im, r3);                     \
                                                            \
    __m256 zk_p = _mm256_sub_ps(r2r, r2i);                  \
    __m256 zk_n = _mm256_add_ps(r3r, r3i);                  \
                                                            \
    __m256 zk   = _mm256_add_ps(zk_p, zk_n);                \
    __m256 zk_d = _mm256_sub_ps(zk_p, zk_n);                \
                                                            \
    r2 = _mm256_sub_ps(uk, zk);                             \
    r0 = _mm256_add_ps(uk, zk);                             \
                                                            \
    zk_d = okfft_avx_swap_sign(zk_d);                       \
    zk_d = okfft_avx_swap_pairs(zk_d);                      \
                                                            \
    r3 = _mm256_add_ps(uk2, zk_d);                          \
    r1 = _mm256_sub_ps(uk2, zk_d);                          \
}

#define OKFFT_AVX_KNKN(re0, im0, re1, im1, r00, r10, r20, r30, r01, r11, r21, r31)  \
{                                                       \
    __m256 uk0  = r00, uk20 = r10;                      \
    __m256 uk1  = r01, uk21 = r11;                      \
    __m256 r20r = _mm256_mul_ps(re0, r20);              \
    __m256 r21r = _mm256_mul_ps(re1, r21);              \
    __m256 r30r = _mm256_mul_ps(re0, r30);              \
    __m256 r31r = _mm256_mul_ps(re1, r31);              \
                                                        \
    r20 = okfft_avx_swap_pairs(r20);                    \
    r21 = okfft_avx_swap_pairs(r21);                    \
    r30 = okfft_avx_swap_pairs(r30);                    \
    r31 = okfft_avx_swap_pairs(r31);                    \
                                                        \
    __m256 r20i = _mm256_mul_ps(im0, r20);              \
    __m256 r21i = _mm256_mul_ps(im1, r21);              \
    __m256 r30i = _mm256_mul_ps(im0, r30);              \
    __m256 r31i = _mm256_mul_ps(im1, r31);              \
                                                        \
    __m256 zk_p0 = _mm256_sub_ps(r20r, r20i);           \
    __m256 zk_p1 = _mm256_sub_ps(r21r, r21i);           \
    __m256 zk_n0 = _mm256_add_ps(r30r, r30i);           \
    __m256 zk_n1 = _mm256_add_ps(r31r, r31i);           \
                                                        \
    __m256 zk0   = _mm256_add_ps(zk_p0, zk_n0);         \
    __m256 zk1   = _mm256_add_ps(zk_p1, zk_n1);         \
    __m256 zk_d0 = _mm256_sub_ps(zk_p0, zk_n0);         \
    __m256 zk_d1 = _mm256_sub_ps(zk_p1, zk_n1);         \
                                                        \
    r20 = _mm256_sub_ps(uk0, zk0);                      \
    r21 = _mm256_sub_ps(uk1, zk1);                      \
    r00 = _mm256_add_ps(uk0, zk0);                      \
    r01 = _mm256_add_ps(uk1, zk1);                      \
                                                        \
    zk_d0 = okfft_avx_swap_sign(zk_d0);                 \
    zk_d1 = okfft_avx_swap_sign(zk_d1);                 \
    zk_d0 = okfft_avx_swap_pairs(zk_d0);                \
    zk_d1 = okfft_avx_swap_pairs(zk_d1);                \
                                                        \
    r30 = _mm256_add_ps(uk20, zk_d0);                   \
    r31 = _mm256_add_ps(uk21, zk_d1);                   \
    r10 = _mm256_sub_ps(uk20, zk_d0);                   \
    r11 = _mm256_sub_ps(uk21, zk_d1);                   \
}

#define OKFFT_AVX_X4(data, lut)                             \
{                                                           \
    __m256 re = _mm256_load_ps(lut + 0);                    \
    __m256 im = _mm256_load_ps(lut + 8);                    \
                                                            \
    __m256 r0 = _mm256_load_ps(data +  0);                  \
    __m256 r1 = _mm256_load_ps(data +  8);                  \
    __m256 r2 = _mm256_load_ps(data + 16);                  \
    __m256 r3 = _mm256_load_ps(data + 24);                  \
                                                            \
    OKFFT_AVX_KN(re, im, r0, r1, r2, r3);                   \
                                                            \
    _mm256_store_ps(data +  0, r0);                         \
    _mm256_store_ps(data +  8, r1);                         \
    _mm256_store_ps(data + 16, r2);                         \
    _mm256_store_ps(data + 24, r3);                         \
}

#define OKFFT_AVX_X4X4(data, lut)                           \
{                                                           \
    __m256 re = _mm256_load_ps(lut + 0);                    \
    __m256 im = _mm256_load_ps(lut + 8);                    \
                                                            \
    __m256 r00 = _mm256_load_ps(data +  0);                 \
    __m256 r10 = _mm256_load_ps(data +  8);                 \
    __m256 r20 = _mm256_load_ps(data + 16);                 \
    __m256 r30 = _mm256_load_ps(data + 24);                 \
    __m256 r01 = _mm256_load_ps(data + 32);                 \
    __m256 r11 = _mm256_load_ps(data + 40);                 \
    __m256 r21 = _mm256_load_ps(data + 48);                 \
    __m256 r31 = _mm256_load_ps(data + 56);                 \
                                                            \
    OKFFT_AVX_KNKN(re, im, re, im,  r00, r10, r20, r30,     \
                                    r01, r11, r21, r31);    \
                                                            \
    _mm256_store_ps(data +  0, r00);                        \
    _mm256_store_ps(data +  8, r10);                        \
    _mm256_store_ps(data + 16, r20);                        \
    _mm256_store_ps(data + 24, r30);                        \
    _mm256_store_ps(data + 32, r01);                        \
    _mm256_store_ps(data + 40, r11);                        \
    _mm256_store_ps(data + 48, r21);                        \
    _mm256_store_ps(data + 56, r31);                        \
}

#define OKFFT_AVX_X8(N, data, p_lut)                        \
{                                                           \
    const size_t OFFS = N / 4;                              \
    const float *__restrict lut = (p_lut);                  \
    float *__restrict d0 = data + (0 * OFFS);               \
    float *__restrict d1 = data + (1 * OFFS);               \
    float *__restrict d2 = data + (2 * OFFS);               \
    float *__restrict d3 = data + (3 * OFFS);               \
    float *__restrict d4 = data + (4 * OFFS);               \
    float *__restrict d5 = data + (5 * OFFS);               \
    float *__restrict d6 = data + (6 * OFFS);               \
    float *__restrict d7 = data + (7 * OFFS);               \
                                                            \
    for (size_t i = 0; i < N / 32; i++)                     \
    {                                                       \
        __m256 re = _mm256_load_ps(lut +  0);               \
        __m256 im = _mm256_load_ps(lut +  8);               \
        __m256 re0 = _mm256_load_ps(lut + 16);              \
        __m256 im0 = _mm256_load_ps(lut + 24);              \
        __m256 re1 = _mm256_load_ps(lut + 32);              \
        __m256 im1 = _mm256_load_ps(lut + 40);              \
                                                            \
        __m256 r0 = _mm256_load_ps(d0);                     \
        __m256 r1 = _mm256_load_ps(d1);                     \
        __m256 r2 = _mm256_load_ps(d2);                     \
        __m256 r3 = _mm256_load_ps(d3);                     \
        __m256 r4 = _mm256_load_ps(d4);                     \
        __m256 r5 = _mm256_load_ps(d5);                     \
        __m256 r6 = _mm256_load_ps(d6);                     \
        __m256 r7 = _mm256_load_ps(d7);                     \
                                                            \
        OKFFT_AVX_KN(re, im, r0, r1, r2, r3);               \
        OKFFT_AVX_KNKN(re0, im0, re1, im1,  r0, r2, r4, r6, \
                                            r1, r3, r5, r7);\
                                                            \
        _mm256_store_ps(d0, r0);                            \
        _mm256_store_ps(d1, r1);                            \
        _mm256_store_ps(d2, r2);                            \
        _mm256_store_ps(d3, r3);                            \
        _mm256_store_ps(d4, r4);                            \
        _mm256_store_ps(d5, r5);                            \
        _mm256_store_ps(d6, r6);                            \
        _mm256_store_ps(d7, r7);                            \
                                                            \
        lut += 48;                                          \
        d0 += 8; d1 += 8; d2 += 8; d3 += 8;                 \
        d4 += 8; d5 += 8; d6 += 8; d7 += 8;                 \
    }                                                       \
}

#define OKFFT_AVX_X8_32(data, p_lut)                        \
{                                                           \
    const float *__restrict lut = (p_lut);                  \
    __m256 re  = _mm256_load_ps(lut +  0);                  \
    __m256 im  = _mm256_load_ps(lut +  8);                  \
    __m256 re0 = _mm256_load_ps(lut + 16);                  \
    __m256 im0 = _mm256_load_ps(lut + 24);                  \
    __m256 re1 = _mm256_load_ps(lut + 32);                  \
    __m256 im1 = _mm256_load_ps(lut + 40);                  \
                                                            \
    __m256 r0 = _mm256_load_ps(data +  0);                  \
    __m256 r1 = _mm256_load_ps(data +  8);                  \
    __m256 r2 = _mm256_load_ps(data + 16);                  \
    __m256 r3 = _mm256_load_ps(data + 24);                  \
    __m256 r4 = _mm256_load_ps(data + 32);                  \
    __m256 r5 = _mm256_load_ps(data + 40);                  \
    __m256 r6 = _mm256_load_ps(data + 48);                  \
    __m256 r7 = _mm256_load_ps(data + 56);                  \
                                                            \
    OKFFT_AVX_KN(re, im, r0, r1, r2, r3);                   \
    OKFFT_AVX_KNKN(re0, im0, re1, im1,  r0, r2, r4, r6,     \
                                        r1, r3, r5, r7);    \
                                                            \
    _mm256_store_ps(data +  0, r0);                         \
    _mm256_store_ps(data +  8, r1);                         \
    _mm256_store_ps(data + 16, r2);                         \
    _mm256_store_ps(data + 24, r3);                         \
    _mm256_store_ps(data + 32, r4);                         \
    _mm256_store_ps(data + 40, r5);                         \
    _mm256_store_ps(data + 48, r6);                         \
    _mm256_store_ps(data + 56, r7);                         \
}

#define OKFFT_AVX_TX2(a, b)                             \
{                                                       \
    __m256 q0 = okfft_avx_unpack_lo(a, b);              \
    __m256 q1 = okfft_avx_unpack_hi(a, b);              \
    a = q0;                                             \
    b = q1;                                             \
}

#define OKFFT_AVX_L2(i0, i1, i2, i3, r0, r1, r2, r3)    \
{                                                       \
    __m256 t0 = _mm256_load_ps(i0);                     \
    __m256 t1 = _mm256_load_ps(i1);                     \
    __m256 t2 = _mm256_load_ps(i2);                     \
    __m256 t3 = _mm256_load_ps(i3);                     \
                                                        \
    r0 = _mm256_add_ps(t0, t1);                         \
    r1 = _mm256_sub_ps(t0, t1);                         \
    r2 = _mm256_add_ps(t2, t3);                         \
    r3 = _mm256_sub_ps(t2, t3);                         \
}

#define OKFFT_AVX_L4(i0, i1, i2, i3, r0, r1, r2, r3)    \
{                                                       \
    __m256 t0 = _mm256_load_ps(i0);                     \
    __m256 t1 = _mm256_load_ps(i1);                     \
    __m256 t2 = _mm256_load_ps(i2);                     \
    __m256 t3 = _mm256_load_ps(i3);                     \
                                                        \
    __m256 t4 = _mm256_add_ps(t0, t1);                  \
    __m256 t5 = _mm256_sub_ps(t0, t1);                  \
    __m256 t6 = _mm256_add_ps(t2, t3);                  \
    __m256 t7 = _mm256_sub_ps(t2, t3);                  \
                                                        \
    t7 = okfft_avx_swap_sign(t7);                       \
    t7 = okfft_avx_swap_pairs(t7);                      \
                                                        \
    r0 = _mm256_add_ps(t4, t6);                         \
    r2 = _mm256_sub_ps(t4, t6);                         \
    r1 = _mm256_sub_ps(t5, t7);                         \
    r3 = _mm256_add_ps(t5, t7);                         \
}

#define OKFFT_AVX_L44(i0, i1, i2, i3, r0, r1, r2, r3)   \
{                                                       \
    __m256 t0 = _mm256_load_ps(i0);                     \
    __m256 t1 = _mm256_load_ps(i1);                     \
    __m256 t2 = _mm256_load_ps(i2);                     \
    __m256 t3 = _mm256_load_ps(i3);                     \
                                                        \
    __m256 t4 = _mm256_add_ps(t0, t1);                  \
    __m256 t5 = _mm256_sub_ps(t0, t1);                  \
    __m256 t6 = _mm256_add_ps(t2, t3);                  \
    __m256 t7 = _mm256_sub_ps(t2, t3);                  \
                                                        \
    t7 = okfft_avx_swap_sign(t7);                       \
    t7 = okfft_avx_swap_pairs(t7);                      \
                                                        \
    t0 = _mm256_add_ps(t4, t6);                         \
    t2 = _mm256_sub_ps(t4, t6);                         \
    t1 = _mm256_sub_ps(t5, t7);                         \
    t3 = _mm256_add_ps(t5, t7);                         \
                                                        \
    OKFFT_AVX_TX2(t0, t1);                              \
    OKFFT_AVX_TX2(t2, t3);                              \
                                                        \
    r0 = t0;                                            \
    r2 = t1;                                            \
    r1 = t2;                                            \
    r3 = t3;                                            \
}

#define OKFFT_AVX_K0(r0, r1, r2, r3)                    \
{                                                       \
    __m256 t0 = r0;                                     \
    __m256 t1 = r1;                                     \
                                                        \
    __m256 t2 = _mm256_add_ps(r2, r3);                  \
    __m256 t3 = _mm256_sub_ps(r2, r3);                  \
                                                        \
    t3 = okfft_avx_swap_sign(t3);                       \
    t3 = okfft_avx_swap_pairs(t3);                      \
                                                        \
    r0 = _mm256_add_ps(t0, t2);                         \
    r2 = _mm256_sub_ps(t0, t2);                         \
    r1 = _mm256_sub_ps(t1, t3);                         \
    r3 = _mm256_add_ps(t1, t3);                         \
}

#define OKFFT_AVX_LEAF_EE(out, os, in, is)              \
{                                                       \
    const float *__restrict LUT = avx_constants;        \
    __m256 r0, r1, r2, r3, r4, r5, r6, r7;              \
    float *__restrict out00 = out + os[0];              \
    float *__restrict out01 = out + os[1];              \
    float *__restrict out10 = out + os[2];              \
    float *__restrict out11 = out + os[3];              \
                                                        \
    OKFFT_AVX_L4(in + is[0], in + is[1], in + is[2], in + is[3], r0, r1, r2, r3);   \
    OKFFT_AVX_L2(in + is[4], in + is[5], in + is[6], in + is[7], r4, r5, r6, r7);   \
                                                        \
    __m256 re = _mm256_load_ps(LUT + 0);                \
    __m256 im = _mm256_load_ps(LUT + 8);                \
    OKFFT_AVX_K0(r0, r2, r4, r6);                       \
    OKFFT_AVX_KN(re, im, r1, r3, r5, r7);               \
                                                        \
    OKFFT_AVX_TX2(r0, r1);                              \
    OKFFT_AVX_TX2(r2, r3);                              \
    OKFFT_AVX_TX2(r4, r5);                              \
    OKFFT_AVX_TX2(r6, r7);                              \
                                                        \
    okfft_avx_store4(out00, out10, r0, r2, r4, r6);     \
    okfft_avx_store4(out01, out11, r1, r3, r5, r7);     \
}

#define OKFFT_AVX_LEAF_OO(out, os, in, is)              \
{                                                       \
    __m256 r0, r1, r2, r3, r4, r5, r6, r7;              \
    float *__restrict out00 = out + os[0];              \
    float *__restrict out01 = out + os[1];              \
    float *__restrict out10 = out + os[2];              \
    float *__restrict out11 = out + os[3];              \
                                                        \
    OKFFT_AVX_L44(in + is[0], in + is[1], in + is[2], in + is[3], r0, r1, r2, r3);  \
    OKFFT_AVX_L44(in + is[6], in + is[7], in + is[4], in + is[5], r4, r5, r6, r7);  \
                                                        \
    okfft_avx_store4(out00, out10, r0, r1, r4, r5);     \
    okfft_avx_store4(out01, out11, r2, r3, r6, r7);     \
}

#define OKFFT_AVX_LEAF_EE2(out, os, in, is)             \
{                                                       \
    const float *__restrict LUT = avx_constants;        \
    __m256 r0, r1, r2, r3, r4, r5, r6, r7;              \
    float *__restrict out00 = out + os[0];              \
    float *__restrict out01 = out + os[1];              \
    float *__restrict out10 = out + os[2];              \
    float *__restrict out11 = out + os[3];              \
                                                        \
    OKFFT_AVX_L4(in + is[6], in + is[7], in + is[4], in + is[5], r0, r1, r2, r3);   \
    OKFFT_AVX_L2(in + is[0], in + is[1], in + is[3], in + is[2], r4, r5, r6, r7);   \
                                                        \
    __m256 re = _mm256_load_ps(LUT + 0);                \
    __m256 im = _mm256_load_ps(LUT + 8);                \
    OKFFT_AVX_K0(r0, r2, r4, r6);                       \
    OKFFT_AVX_KN(re, im, r1, r3, r5, r7);               \
                                                        \
    OKFFT_AVX_TX2(r0, r1);                              \
    OKFFT_AVX_TX2(r2, r3);                              \
    OKFFT_AVX_TX2(r4, r5);                              \
    OKFFT_AVX_TX2(r6, r7);                              \
                                                        \
    okfft_avx_store4(out00, out10, r0, r2, r4, r6);     \
    okfft_avx_store4(out01, out11, r1, r3, r5, r7);     \
}

#define OKFFT_AVX_FP_EVEN(i0, i1, p, p_out, p_in)       \
{                                                       \
    float *__restrict out = p_out;                      \
    const float *__restrict in = p_in;                  \
    const ptrdiff_t *__restrict is = p->is;             \
    const ptrdiff_t *__restrict os = p->offsets;        \
                                                        \
    for (size_t i = i0 >> 1; i > 0; --i)                \
    {                                                   \
        OKFFT_AVX_LEAF_EE(out, os, in, is);             \
        in += 8; os += 4;                               \
    }                                                   \
                                                        \
    OKFFT_SSE_LEAF_EE(out, os, in, is);                 \
    in += 4; os += 2;                                   \
                                                        \
    OKFFT_SSE_LEAF_EO(out, os, in, is);                 \
    in += 4; os += 2;                                   \
                                                        \
    for (size_t i = i1 >> 1; i > 0; --i)                \
    {                                                   \
        OKFFT_AVX_LEAF_OO(out, os, in, is);             \
        in += 8; os += 4;                               \
    }                                                   \
                                                        \
    OKFFT_SSE_LEAF_OO(out, os, in, is);                 \
    in += 4; os += 2;                                   \
                                                        \
    for (size_t i = i1 >> 1; i > 0; --i)                \
    {                                                   \
        OKFFT_AVX_LEAF_EE2(out, os, in, is);            \
        in += 8; os += 4;                               \
    }                                                   \
                                                        \
    OKFFT_SSE_LEAF_EE2(out, os, in, is);                \
}

#define OKFFT_AVX_FP_ODD(i0, i1, p, p_out, p_in)        \
{                                                       \
    float *__restrict out = p_out;                      \
    const float *__restrict in = p_in;                  \
    const ptrdiff_t *__restrict is = p->is;             \
    const ptrdiff_t *__restrict os = p->offsets;        \
                                                        \
    for (size_t i = i0 >> 1; i > 0; --i)                \
    {                                                   \
        OKFFT_AVX_LEAF_EE(out, os, in, is);             \
        in += 8;                                        \
        os += 4;                                        \
    }                                                   \
                                                        \
    OKFFT_SSE_LEAF_EE(out, os, in, is);                 \
    in += 4; os += 2;                                   \
                                                        \
    for (size_t i = i1 >> 1; i > 0; --i)                \
    {                                                   \
        OKFFT_AVX_LEAF_OO(out, os, in, is);             \
        in += 8;                                        \
        os += 4;                                        \
    }                                                   \
                                                        \
    OKFFT_SSE_LEAF_OE(out, os, in, is);                 \
    in += 4;                                            \
    os += 2;                                            \
                                                        \
    for (size_t i = i1 >> 1; i > 0; --i)                \
    {                                                   \
        OKFFT_AVX_LEAF_EE2(out, os, in, is);            \
        in += 8;                                        \
        os += 4;                                        \
    }                                                   \
}

#define OKFFT_AVX_XF_32(data)                       \
{                                                   \
    OKFFT_AVX_X8_32(data, ws1);                     \
}

#define OKFFT_AVX_XF_64(data)                       \
{                                                   \
    OKFFT_AVX_X4(data + 0,  ws);                    \
    OKFFT_AVX_X4X4(data + 64, ws);                  \
    OKFFT_AVX_X8(64, data, ws + (ws_is[2] << 1));   \
}

#define OKFFT_AVX_XF_128(data)                      \
{                                                   \
    OKFFT_AVX_X4X4(data + 64, ws);                  \
    OKFFT_AVX_X8_32(data +   0, ws1);               \
    OKFFT_AVX_X8_32(data + 128, ws1);               \
    OKFFT_AVX_X8_32(data + 192, ws1);               \
    OKFFT_AVX_X8(128, data, ws + (ws_is[3] << 1));  \
}

#define OKFFT_AVX_XF_256(data)                      \
{                                                   \
    OKFFT_AVX_XF_64(data);                          \
    OKFFT_AVX_XF_32(data + 2 * 64);                 \
    OKFFT_AVX_XF_32(data + 3 * 64);                 \
    OKFFT_AVX_XF_64(data + 4 * 64);                 \
    OKFFT_AVX_XF_64(data + 6 * 64);                 \
    OKFFT_AVX_X8(256, data, ws + (ws_is[4] << 1));  \
}

#define OKFFT_AVX_XF_512(data)                      \
{                                                   \
    OKFFT_AVX_XF_128(data);                         \
    OKFFT_AVX_XF_64(data +  2 * 128);               \
    OKFFT_AVX_XF_64(data +  3 * 128);               \
    OKFFT_AVX_XF_128(data + 4 * 128);               \
    OKFFT_AVX_XF_128(data + 6 * 128);               \
    OKFFT_AVX_X8(512, data, ws + (ws_is[5] << 1));  \
}

#define OKFFT_AVX_XF_1024(data)                     \
{                                                   \
    OKFFT_AVX_XF_256(data);                         \
    OKFFT_AVX_XF_128(data + 2 * 256);               \
    OKFFT_AVX_XF_128(data + 3 * 256);               \
    OKFFT_AVX_XF_256(data + 4 * 256);               \
    OKFFT_AVX_XF_256(data + 6 * 256);               \
    OKFFT_AVX_X8(1024, data, ws + (ws_is[6] << 1)); \
}

#endif