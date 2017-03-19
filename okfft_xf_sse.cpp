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

#include "okfft.h"
#include "okfft_macros.h"

#ifdef _MSC_VER
    #include <intrin.h>
    #define okfft_force_inline __forceinline
    #define OKFFT_ALIGN(x) __declspec(align(x))
#else
    #include <x86intrin.h>
    #define okfft_force_inline inline __attribute__((always_inline))
    #define OKFFT_ALIGN(x) __attribute__((aligned(x)))
#endif

#define OKFFT_SQRT_HALF 0.7071067811865475244008443621048490392848359376884740f

static const OKFFT_ALIGN(16) float okfft_sse_inv_constants[16] =
{
    OKFFT_SQRT_HALF,     OKFFT_SQRT_HALF,   OKFFT_SQRT_HALF,     OKFFT_SQRT_HALF,
    OKFFT_SQRT_HALF,    -OKFFT_SQRT_HALF,   OKFFT_SQRT_HALF,    -OKFFT_SQRT_HALF,
    1.0f,                1.0f,              OKFFT_SQRT_HALF,     OKFFT_SQRT_HALF,
    0.0f,                0.0f,              OKFFT_SQRT_HALF,    -OKFFT_SQRT_HALF,
};

static const OKFFT_ALIGN(16) float okfft_sse_fwd_constants[16] =
{
     OKFFT_SQRT_HALF,    OKFFT_SQRT_HALF,    OKFFT_SQRT_HALF,   OKFFT_SQRT_HALF,
    -OKFFT_SQRT_HALF,    OKFFT_SQRT_HALF,   -OKFFT_SQRT_HALF,   OKFFT_SQRT_HALF,
     1.0f,               1.0f,               OKFFT_SQRT_HALF,   OKFFT_SQRT_HALF,
     0.0f,               0.0f,              -OKFFT_SQRT_HALF,   OKFFT_SQRT_HALF,
};

static const __m128 okfft_sse_fwd_sign_mask = _mm_set_ps(-0.f, 0.f, -0.f, 0.f);
static const __m128 okfft_sse_inv_sign_mask = _mm_set_ps(0.f, -0.f, 0.f, -0.f);

static okfft_force_inline size_t okfft_sse_ilog2(size_t N)
{
#ifdef _MSC_VER
    unsigned long l2;
    _BitScanReverse64(&l2, N);
    return l2;
#else
    return __builtin_ctzll(N);
#endif
}

// ================= FORWARDS ===================================

static okfft_force_inline void okfft_sse_xf_fwd_32(const okfft_plan_t *plan, float *__restrict data)
{
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;
    OKFFT_SSE_X8(32, data, ws + (ws_is[1] << 1));
}

static okfft_force_inline void okfft_sse_xf_fwd_64(const okfft_plan_t *plan, float *__restrict data)
{
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;

    OKFFT_SSE_X4(data +  0, ws);
    OKFFT_SSE_X4(data + 64, ws);
    OKFFT_SSE_X4(data + 96, ws);
    OKFFT_SSE_X8(64, data, ws + (ws_is[2] << 1));
}

static okfft_force_inline void okfft_sse_xf_fwd_128(const okfft_plan_t *plan, float *__restrict data)
{
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const ptrdiff_t *ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;
    const float *__restrict ws1 = ws + (ws_is[1] << 1);

    OKFFT_SSE_X4(data + 64, ws);
    OKFFT_SSE_X4(data + 96, ws);
    OKFFT_SSE_X8(32, data + 0, ws1);
    OKFFT_SSE_X8(32, data + 128, ws1);
    OKFFT_SSE_X8(32, data + 192, ws1);
    OKFFT_SSE_X8(128, data, ws + (ws_is[3] << 1));
}

static okfft_force_inline void okfft_sse_xf_fwd_256(const okfft_plan_t *plan, float *__restrict data)
{
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;
    const float *__restrict ws1 = ws + (ws_is[1] << 1);
    OKFFT_SSE_XF_256(data)
}

static okfft_force_inline void okfft_sse_xf_fwd_512(const okfft_plan_t *plan, float *__restrict data)
{
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;
    const float *__restrict ws1 = ws + (ws_is[1] << 1);
    OKFFT_SSE_XF_512(data)
}

static inline void okfft_sse_xf_fwd_1k(const okfft_plan_t *plan, float *__restrict data)
{
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;
    const float *__restrict ws1 = ws + (ws_is[1] << 1);
    OKFFT_SSE_XF_1024(data)
}

static inline void okfft_sse_xf_fwd_2k(const okfft_plan_t *plan, float *__restrict data)
{
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;

    okfft_sse_xf_fwd_512(plan, data);
    okfft_sse_xf_fwd_256(plan, data + 2 * 512);
    okfft_sse_xf_fwd_256(plan, data + 3 * 512);
    okfft_sse_xf_fwd_512(plan, data + 4 * 512);
    okfft_sse_xf_fwd_512(plan, data + 6 * 512);
    OKFFT_SSE_X8(2 * 1024, data, ws + (ws_is[7] << 1));
}

static inline void okfft_sse_xf_fwd_4k(const okfft_plan_t *plan, float *__restrict data)
{
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;

    okfft_sse_xf_fwd_1k(plan, data);
    okfft_sse_xf_fwd_512(plan, data + 2 * 1024);
    okfft_sse_xf_fwd_512(plan, data + 3 * 1024);
    okfft_sse_xf_fwd_1k(plan, data + 4 * 1024);
    okfft_sse_xf_fwd_1k(plan, data + 6 * 1024);
    OKFFT_SSE_X8(4 * 1024, data, ws + (ws_is[8] << 1));
}

static inline void okfft_sse_xf_fwd_8k(const okfft_plan_t *plan, float *__restrict data)
{
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;

    okfft_sse_xf_fwd_2k(plan, data);
    okfft_sse_xf_fwd_1k(plan, data + 2 * 2 * 1024);
    okfft_sse_xf_fwd_1k(plan, data + 3 * 2 * 1024);
    okfft_sse_xf_fwd_2k(plan, data + 4 * 2 * 1024);
    okfft_sse_xf_fwd_2k(plan, data + 6 * 2 * 1024);
    OKFFT_SSE_X8(8 * 1024, data, ws + (ws_is[9] << 1));
}

static inline void okfft_sse_xf_fwd_16k(const okfft_plan_t *plan, float *__restrict data)
{
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;

    okfft_sse_xf_fwd_4k(plan, data);
    okfft_sse_xf_fwd_2k(plan, data + 2 * 4 * 1024);
    okfft_sse_xf_fwd_2k(plan, data + 3 * 4 * 1024);
    okfft_sse_xf_fwd_4k(plan, data + 4 * 4 * 1024);
    okfft_sse_xf_fwd_4k(plan, data + 6 * 4 * 1024);
    OKFFT_SSE_X8(16 * 1024, data, ws + (ws_is[10] << 1));
}

static inline void okfft_sse_xf_fwd_rec(const okfft_plan_t *plan, float *__restrict data, size_t N)
{
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;

    if (N > 64 * 1024)
    {
        size_t N2 = N >> 1;
        size_t N4 = N >> 2;
        size_t N8 = N >> 3;

        okfft_sse_xf_fwd_rec(plan, data, N4);
        okfft_sse_xf_fwd_rec(plan, data + N2, N8);
        okfft_sse_xf_fwd_rec(plan, data + N2 + N4, N8);
        okfft_sse_xf_fwd_rec(plan, data + N, N4);
        okfft_sse_xf_fwd_rec(plan, data + N + N2, N4);
        OKFFT_SSE_X8(N, data, ws + (ws_is[okfft_sse_ilog2(N) - 4] << 1));
    }
    else if (N == 64 * 1024)
    {
        okfft_sse_xf_fwd_16k(plan, data);
        okfft_sse_xf_fwd_8k(plan, data + 2 * 16 * 1024);
        okfft_sse_xf_fwd_8k(plan, data + 3 * 16 * 1024);
        okfft_sse_xf_fwd_16k(plan, data + 4 * 16 * 1024);
        okfft_sse_xf_fwd_16k(plan, data + 6 * 16 * 1024);
        OKFFT_SSE_X8(64 * 1024, data, ws + (ws_is[12] << 1));
    }
    else if (N == 32 * 1024)
    {
        okfft_sse_xf_fwd_8k(plan, data);
        okfft_sse_xf_fwd_4k(plan, data + 2 * 8 * 1024);
        okfft_sse_xf_fwd_4k(plan, data + 3 * 8 * 1024);
        okfft_sse_xf_fwd_8k(plan, data + 4 * 8 * 1024);
        okfft_sse_xf_fwd_8k(plan, data + 6 * 8 * 1024);
        OKFFT_SSE_X8(32 * 1024, data, ws + (ws_is[11] << 1));
    }
    else if (N == 16 * 1024)
        okfft_sse_xf_fwd_16k(plan, data);
}

void okfft_sse_fwd_32(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const float *__restrict sse_constants = okfft_sse_fwd_constants;
    OKFFT_SSE_FP_ODD(1, 0, plan, output, input);
    okfft_sse_xf_fwd_32(plan, output);
}

void okfft_sse_fwd_64(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const float *__restrict sse_constants = okfft_sse_fwd_constants;
    OKFFT_SSE_FP_EVEN(1, 1, plan, output, input);
    okfft_sse_xf_fwd_64(plan, output);
}

void okfft_sse_fwd_128(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const float *__restrict sse_constants = okfft_sse_fwd_constants;
    OKFFT_SSE_FP_ODD(3, 2, plan, output, input);
    okfft_sse_xf_fwd_128(plan, output);
}

void okfft_sse_fwd_256(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const float *__restrict sse_constants = okfft_sse_fwd_constants;
    OKFFT_SSE_FP_EVEN(5, 5, plan, output, input);
    okfft_sse_xf_fwd_256(plan, output);
}

void okfft_sse_fwd_512(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const float *__restrict sse_constants = okfft_sse_fwd_constants;
    OKFFT_SSE_FP_ODD(11, 10, plan, output, input);
    okfft_sse_xf_fwd_512(plan, output);
}

void okfft_sse_fwd_1024(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const float *__restrict sse_constants = okfft_sse_fwd_constants;
    OKFFT_SSE_FP_EVEN(21, 21, plan, output, input);
    okfft_sse_xf_fwd_1k(plan, output);
}

void okfft_sse_fwd_2048(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const float *__restrict sse_constants = okfft_sse_fwd_constants;
    OKFFT_SSE_FP_ODD(43, 42, plan, output, input);
    okfft_sse_xf_fwd_2k(plan, output);
}

void okfft_sse_fwd_4096(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const float *__restrict sse_constants = okfft_sse_fwd_constants;
    OKFFT_SSE_FP_EVEN(85, 85, plan, output, input);
    okfft_sse_xf_fwd_4k(plan, output);
}

void okfft_sse_fwd_8192(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const float *__restrict sse_constants = okfft_sse_fwd_constants;
    OKFFT_SSE_FP_ODD(171, 170, plan, output, input);
    okfft_sse_xf_fwd_8k(plan, output);
}

void okfft_sse_fwd_generic(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const float *__restrict sse_constants = okfft_sse_fwd_constants;
    size_t i0 = plan->i0, i1 = plan->i1;
    if (okfft_sse_ilog2(plan->N) & 1) // check if ilog2(N) is odd
    {
        OKFFT_SSE_FP_ODD(i0, i1, plan, output, input)
    }
    else
    {
        OKFFT_SSE_FP_EVEN(i0, i1, plan, output, input)
    }

    okfft_sse_xf_fwd_rec(plan, output, plan->N);
}

void okfft_sse_fwd_real(float *__restrict output, float *__restrict buffer, const float *__restrict A, const float *__restrict B, size_t N)
{
    buffer[N + 0] = buffer[0];
    buffer[N + 1] = buffer[1];

    for (size_t i = 0; i < N; i += 16)
    {
        __m128 x00 = _mm_load_ps(buffer + i +  0);
        __m128 x10 = _mm_load_ps(buffer + i +  4);
        __m128 x01 = _mm_load_ps(buffer + i +  8);
        __m128 x11 = _mm_load_ps(buffer + i + 12);

        __m128 y00 = _mm_loadu_ps(buffer + N - i -  2);
        __m128 y10 = _mm_loadu_ps(buffer + N - i -  6);
        __m128 y01 = _mm_loadu_ps(buffer + N - i - 10);
        __m128 y11 = _mm_loadu_ps(buffer + N - i - 14);

        __m128 xre0 = _mm_shuffle_ps(x00, x10, _MM_SHUFFLE(2, 0, 2, 0));
        __m128 xim0 = _mm_shuffle_ps(x00, x10, _MM_SHUFFLE(3, 1, 3, 1));
        __m128 yre0 = _mm_shuffle_ps(y00, y10, _MM_SHUFFLE(0, 2, 0, 2));
        __m128 yim0 = _mm_shuffle_ps(y00, y10, _MM_SHUFFLE(1, 3, 1, 3));

        __m128 xre1 = _mm_shuffle_ps(x01, x11, _MM_SHUFFLE(2, 0, 2, 0));
        __m128 xim1 = _mm_shuffle_ps(x01, x11, _MM_SHUFFLE(3, 1, 3, 1));
        __m128 yre1 = _mm_shuffle_ps(y01, y11, _MM_SHUFFLE(0, 2, 0, 2));
        __m128 yim1 = _mm_shuffle_ps(y01, y11, _MM_SHUFFLE(1, 3, 1, 3));

        __m128 are0 = _mm_load_ps(A + i +  0);
        __m128 aim0 = _mm_load_ps(A + i +  4);
        __m128 are1 = _mm_load_ps(A + i +  8);
        __m128 aim1 = _mm_load_ps(A + i + 12);

        __m128 bre0 = _mm_load_ps(B + i +  0);
        __m128 bim0 = _mm_load_ps(B + i +  4);
        __m128 bre1 = _mm_load_ps(B + i +  8);
        __m128 bim1 = _mm_load_ps(B + i + 12);

        __m128 m000 = _mm_mul_ps(xre0, are0);
        __m128 m100 = _mm_mul_ps(xim0, aim0);
        __m128 m200 = _mm_mul_ps(yre0, bre0);
        __m128 m300 = _mm_mul_ps(yim0, bim0);

        __m128 m010 = _mm_mul_ps(xim0, are0);
        __m128 m110 = _mm_mul_ps(xre0, aim0);
        __m128 m210 = _mm_mul_ps(yre0, bim0);
        __m128 m310 = _mm_mul_ps(yim0, bre0);

        __m128 m001 = _mm_mul_ps(xre1, are1);
        __m128 m101 = _mm_mul_ps(xim1, aim1);
        __m128 m201 = _mm_mul_ps(yre1, bre1);
        __m128 m301 = _mm_mul_ps(yim1, bim1);

        __m128 m011 = _mm_mul_ps(xim1, are1);
        __m128 m111 = _mm_mul_ps(xre1, aim1);
        __m128 m211 = _mm_mul_ps(yre1, bim1);
        __m128 m311 = _mm_mul_ps(yim1, bre1);

        __m128 re00 = _mm_sub_ps(m000, m100);
        __m128 re10 = _mm_add_ps(m200, m300);
        __m128 im00 = _mm_add_ps(m010, m110);
        __m128 im10 = _mm_sub_ps(m210, m310);

        __m128 re01 = _mm_sub_ps(m001, m101);
        __m128 re11 = _mm_add_ps(m201, m301);
        __m128 im01 = _mm_add_ps(m011, m111);
        __m128 im11 = _mm_sub_ps(m211, m311);

        __m128 re0  = _mm_add_ps(re00, re10);
        __m128 im0  = _mm_add_ps(im00, im10);

        __m128 re1  = _mm_add_ps(re01, re11);
        __m128 im1  = _mm_add_ps(im01, im11);

        __m128 o00  = _mm_unpacklo_ps(re0, im0);
        __m128 o10  = _mm_unpackhi_ps(re0, im0);

        __m128 o01 = _mm_unpacklo_ps(re1, im1);
        __m128 o11 = _mm_unpackhi_ps(re1, im1);

        _mm_store_ps(output + i +  0, o00);
        _mm_store_ps(output + i +  4, o10);
        _mm_store_ps(output + i +  8, o01);
        _mm_store_ps(output + i + 12, o11);
    }

    output[N + 0] = buffer[0] - buffer[1];
    output[N + 1] = 0.0f;
}

// ================= BACKWARDS ==================================

static okfft_force_inline void okfft_sse_xf_inv_32(const okfft_plan_t *plan, float *__restrict data)
{
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;
    OKFFT_SSE_X8(32, data, ws + (ws_is[1] << 1));
}

static okfft_force_inline void okfft_sse_xf_inv_64(const okfft_plan_t *plan, float *__restrict data)
{
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;

    OKFFT_SSE_X4(data + 0, ws);
    OKFFT_SSE_X4(data + 64, ws);
    OKFFT_SSE_X4(data + 96, ws);
    OKFFT_SSE_X8(64, data, ws + (ws_is[2] << 1));
}

static okfft_force_inline void okfft_sse_xf_inv_128(const okfft_plan_t *plan, float *__restrict data)
{
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const ptrdiff_t *ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;
    const float *__restrict ws1 = ws + (ws_is[1] << 1);

    OKFFT_SSE_X4(data + 64, ws);
    OKFFT_SSE_X4(data + 96, ws);
    OKFFT_SSE_X8(32, data + 0, ws1);
    OKFFT_SSE_X8(32, data + 128, ws1);
    OKFFT_SSE_X8(32, data + 192, ws1);
    OKFFT_SSE_X8(128, data, ws + (ws_is[3] << 1));
}

static okfft_force_inline void okfft_sse_xf_inv_256(const okfft_plan_t *plan, float *__restrict data)
{
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;
    const float *__restrict ws1 = ws + (ws_is[1] << 1);
    OKFFT_SSE_XF_256(data)
}

static okfft_force_inline void okfft_sse_xf_inv_512(const okfft_plan_t *plan, float *__restrict data)
{
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;
    const float *__restrict ws1 = ws + (ws_is[1] << 1);
    OKFFT_SSE_XF_512(data)
}

static inline void okfft_sse_xf_inv_1k(const okfft_plan_t *plan, float *__restrict data)
{
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;
    const float *__restrict ws1 = ws + (ws_is[1] << 1);
    OKFFT_SSE_XF_1024(data)
}

static inline void okfft_sse_xf_inv_2k(const okfft_plan_t *plan, float *__restrict data)
{
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;

    okfft_sse_xf_inv_512(plan, data);
    okfft_sse_xf_inv_256(plan, data + 2 * 512);
    okfft_sse_xf_inv_256(plan, data + 3 * 512);
    okfft_sse_xf_inv_512(plan, data + 4 * 512);
    okfft_sse_xf_inv_512(plan, data + 6 * 512);
    OKFFT_SSE_X8(2 * 1024, data, ws + (ws_is[7] << 1));
}

static inline void okfft_sse_xf_inv_4k(const okfft_plan_t *plan, float *__restrict data)
{
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;

    okfft_sse_xf_inv_1k(plan, data);
    okfft_sse_xf_inv_512(plan, data + 2 * 1024);
    okfft_sse_xf_inv_512(plan, data + 3 * 1024);
    okfft_sse_xf_inv_1k(plan, data + 4 * 1024);
    okfft_sse_xf_inv_1k(plan, data + 6 * 1024);
    OKFFT_SSE_X8(4 * 1024, data, ws + (ws_is[8] << 1));
}

static inline void okfft_sse_xf_inv_8k(const okfft_plan_t *plan, float *__restrict data)
{
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;

    okfft_sse_xf_inv_2k(plan, data);
    okfft_sse_xf_inv_1k(plan, data + 2 * 2 * 1024);
    okfft_sse_xf_inv_1k(plan, data + 3 * 2 * 1024);
    okfft_sse_xf_inv_2k(plan, data + 4 * 2 * 1024);
    okfft_sse_xf_inv_2k(plan, data + 6 * 2 * 1024);
    OKFFT_SSE_X8(8 * 1024, data, ws + (ws_is[9] << 1));
}

static inline void okfft_sse_xf_inv_16k(const okfft_plan_t *plan, float *__restrict data)
{
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;

    okfft_sse_xf_inv_4k(plan, data);
    okfft_sse_xf_inv_2k(plan, data + 2 * 4 * 1024);
    okfft_sse_xf_inv_2k(plan, data + 3 * 4 * 1024);
    okfft_sse_xf_inv_4k(plan, data + 4 * 4 * 1024);
    okfft_sse_xf_inv_4k(plan, data + 6 * 4 * 1024);
    OKFFT_SSE_X8(16 * 1024, data, ws + (ws_is[10] << 1));
}

static inline void okfft_sse_xf_inv_rec(const okfft_plan_t *plan, float *__restrict data, size_t N)
{
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;

    if (N > 64 * 1024)
    {
        size_t N2 = N >> 1;
        size_t N4 = N >> 2;
        size_t N8 = N >> 3;

        okfft_sse_xf_inv_rec(plan, data, N4);
        okfft_sse_xf_inv_rec(plan, data + N2, N8);
        okfft_sse_xf_inv_rec(plan, data + N2 + N4, N8);
        okfft_sse_xf_inv_rec(plan, data + N, N4);
        okfft_sse_xf_inv_rec(plan, data + N + N2, N4);
        OKFFT_SSE_X8(N, data, ws + (ws_is[okfft_sse_ilog2(N) - 4] << 1));
    }
    else if (N == 64 * 1024)
    {
        okfft_sse_xf_inv_16k(plan, data);
        okfft_sse_xf_inv_8k(plan, data + 2 * 16 * 1024);
        okfft_sse_xf_inv_8k(plan, data + 3 * 16 * 1024);
        okfft_sse_xf_inv_16k(plan, data + 4 * 16 * 1024);
        okfft_sse_xf_inv_16k(plan, data + 6 * 16 * 1024);
        OKFFT_SSE_X8(64 * 1024, data, ws + (ws_is[12] << 1));
    }
    else if (N == 32 * 1024)
    {
        okfft_sse_xf_inv_8k(plan, data);
        okfft_sse_xf_inv_4k(plan, data + 2 * 8 * 1024);
        okfft_sse_xf_inv_4k(plan, data + 3 * 8 * 1024);
        okfft_sse_xf_inv_8k(plan, data + 4 * 8 * 1024);
        okfft_sse_xf_inv_8k(plan, data + 6 * 8 * 1024);
        OKFFT_SSE_X8(32 * 1024, data, ws + (ws_is[11] << 1));
    }
    else if (N == 16 * 1024)
        okfft_sse_xf_inv_16k(plan, data);
}


void okfft_sse_inv_32(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const float *__restrict sse_constants = okfft_sse_inv_constants;
    OKFFT_SSE_FP_ODD(1, 0, plan, output, input);
    okfft_sse_xf_inv_32(plan, output);
}

void okfft_sse_inv_64(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const float *__restrict sse_constants = okfft_sse_inv_constants;
    OKFFT_SSE_FP_EVEN(1, 1, plan, output, input);
    okfft_sse_xf_inv_64(plan, output);
}

void okfft_sse_inv_128(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const float *__restrict sse_constants = okfft_sse_inv_constants;
    OKFFT_SSE_FP_ODD(3, 2, plan, output, input);
    okfft_sse_xf_inv_128(plan, output);
}

void okfft_sse_inv_256(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const float *__restrict sse_constants = okfft_sse_inv_constants;
    OKFFT_SSE_FP_EVEN(5, 5, plan, output, input);
    okfft_sse_xf_inv_256(plan, output);
}

void okfft_sse_inv_512(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const float *__restrict sse_constants = okfft_sse_inv_constants;
    OKFFT_SSE_FP_ODD(11, 10, plan, output, input);
    okfft_sse_xf_inv_512(plan, output);
}

void okfft_sse_inv_1024(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const float *__restrict sse_constants = okfft_sse_inv_constants;
    OKFFT_SSE_FP_EVEN(21, 21, plan, output, input);
    okfft_sse_xf_inv_1k(plan, output);
}

void okfft_sse_inv_2048(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const float *__restrict sse_constants = okfft_sse_inv_constants;
    OKFFT_SSE_FP_ODD(43, 42, plan, output, input);
    okfft_sse_xf_inv_2k(plan, output);
}

void okfft_sse_inv_4096(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const float *__restrict sse_constants = okfft_sse_inv_constants;
    OKFFT_SSE_FP_EVEN(85, 85, plan, output, input);
    okfft_sse_xf_inv_4k(plan, output);
}

void okfft_sse_inv_8192(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const float *__restrict sse_constants = okfft_sse_inv_constants;
    OKFFT_SSE_FP_ODD(171, 170, plan, output, input);
    okfft_sse_xf_inv_8k(plan, output);
}

void okfft_sse_inv_generic(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const float *__restrict sse_constants = okfft_sse_inv_constants;
    size_t i0 = plan->i0, i1 = plan->i1;
    if (okfft_sse_ilog2(plan->N) & 1) // check if ilog2(N) is odd
    {
        OKFFT_SSE_FP_ODD(i0, i1, plan, output, input)
    }
    else
    {
        OKFFT_SSE_FP_EVEN(i0, i1, plan, output, input)
    }

    okfft_sse_xf_inv_rec(plan, output, plan->N);
}

void okfft_sse_inv_real(float *__restrict output, const float *__restrict input, const float *__restrict A, const float *__restrict B, size_t N)
{
    for (size_t i = 0; i < N; i += 16)
    {
        __m128 x00 = _mm_load_ps(input + i +  0);
        __m128 x10 = _mm_load_ps(input + i +  4);
        __m128 x01 = _mm_load_ps(input + i +  8);
        __m128 x11 = _mm_load_ps(input + i + 12);

        __m128 y00 = _mm_loadu_ps(input + N - i -  2);
        __m128 y10 = _mm_loadu_ps(input + N - i -  6);
        __m128 y01 = _mm_loadu_ps(input + N - i - 10);
        __m128 y11 = _mm_loadu_ps(input + N - i - 14);

        __m128 xre0 = _mm_shuffle_ps(x00, x10, _MM_SHUFFLE(2, 0, 2, 0));
        __m128 xim0 = _mm_shuffle_ps(x00, x10, _MM_SHUFFLE(3, 1, 3, 1));
        __m128 yre0 = _mm_shuffle_ps(y00, y10, _MM_SHUFFLE(0, 2, 0, 2));
        __m128 yim0 = _mm_shuffle_ps(y00, y10, _MM_SHUFFLE(1, 3, 1, 3));

        __m128 xre1 = _mm_shuffle_ps(x01, x11, _MM_SHUFFLE(2, 0, 2, 0));
        __m128 xim1 = _mm_shuffle_ps(x01, x11, _MM_SHUFFLE(3, 1, 3, 1));
        __m128 yre1 = _mm_shuffle_ps(y01, y11, _MM_SHUFFLE(0, 2, 0, 2));
        __m128 yim1 = _mm_shuffle_ps(y01, y11, _MM_SHUFFLE(1, 3, 1, 3));

        __m128 are0 = _mm_load_ps(A + i + 0);
        __m128 aim0 = _mm_load_ps(A + i + 4);
        __m128 are1 = _mm_load_ps(A + i + 8);
        __m128 aim1 = _mm_load_ps(A + i + 12);

        __m128 bre0 = _mm_load_ps(B + i + 0);
        __m128 bim0 = _mm_load_ps(B + i + 4);
        __m128 bre1 = _mm_load_ps(B + i + 8);
        __m128 bim1 = _mm_load_ps(B + i + 12);

        __m128 m000 = _mm_mul_ps(xre0, are0);
        __m128 m100 = _mm_mul_ps(xim0, aim0);
        __m128 m200 = _mm_mul_ps(yre0, bre0);
        __m128 m300 = _mm_mul_ps(yim0, bim0);

        __m128 m010 = _mm_mul_ps(xim0, are0);
        __m128 m110 = _mm_mul_ps(xre0, aim0);
        __m128 m210 = _mm_mul_ps(yre0, bim0);
        __m128 m310 = _mm_mul_ps(yim0, bre0);

        __m128 m001 = _mm_mul_ps(xre1, are1);
        __m128 m101 = _mm_mul_ps(xim1, aim1);
        __m128 m201 = _mm_mul_ps(yre1, bre1);
        __m128 m301 = _mm_mul_ps(yim1, bim1);

        __m128 m011 = _mm_mul_ps(xim1, are1);
        __m128 m111 = _mm_mul_ps(xre1, aim1);
        __m128 m211 = _mm_mul_ps(yre1, bim1);
        __m128 m311 = _mm_mul_ps(yim1, bre1);

        __m128 re00 = _mm_add_ps(m000, m100);
        __m128 re10 = _mm_sub_ps(m200, m300);
        __m128 im00 = _mm_sub_ps(m010, m110);
        __m128 im10 = _mm_add_ps(m210, m310);

        __m128 re01 = _mm_add_ps(m001, m101);
        __m128 re11 = _mm_sub_ps(m201, m301);
        __m128 im01 = _mm_sub_ps(m011, m111);
        __m128 im11 = _mm_add_ps(m211, m311);

        __m128 re0 = _mm_add_ps(re00, re10);
        __m128 im0 = _mm_sub_ps(im00, im10);

        __m128 re1 = _mm_add_ps(re01, re11);
        __m128 im1 = _mm_sub_ps(im01, im11);

        __m128 o00 = _mm_unpacklo_ps(re0, im0);
        __m128 o10 = _mm_unpackhi_ps(re0, im0);

        __m128 o01 = _mm_unpacklo_ps(re1, im1);
        __m128 o11 = _mm_unpackhi_ps(re1, im1);

        _mm_store_ps(output + i + 0, o00);
        _mm_store_ps(output + i + 4, o10);
        _mm_store_ps(output + i + 8, o01);
        _mm_store_ps(output + i + 12, o11);
    }
}