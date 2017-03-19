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

#ifdef OKFFT_HAS_AVX

#ifdef _MSC_VER
    #include <intrin.h>
    #define okfft_force_inline __forceinline
    #define OKFFT_ALIGN(x) __declspec(align(x))
#else
    #include <x86intrin.h>
    #define okfft_force_inline inline __attribute__((always_inline))
    #define OKFFT_ALIGN(x) __attribute__((aligned(x)))
#endif

#ifdef _MSC_VER
#else
#endif

#define OKFFT_SQRT_HALF 0.7071067811865475244008443621048490392848359376884740f

// need sse constants for AVX leafs too!
static const OKFFT_ALIGN(16) float okfft_sse_fwd_constants[16] =
{
     OKFFT_SQRT_HALF,    OKFFT_SQRT_HALF,    OKFFT_SQRT_HALF,   OKFFT_SQRT_HALF,
    -OKFFT_SQRT_HALF,    OKFFT_SQRT_HALF,   -OKFFT_SQRT_HALF,   OKFFT_SQRT_HALF,
     1.0f,               1.0f,               OKFFT_SQRT_HALF,   OKFFT_SQRT_HALF,
     0.0f,               0.0f,              -OKFFT_SQRT_HALF,   OKFFT_SQRT_HALF,
};

static const OKFFT_ALIGN(16) float okfft_sse_inv_constants[16] =
{
    OKFFT_SQRT_HALF,     OKFFT_SQRT_HALF,   OKFFT_SQRT_HALF,     OKFFT_SQRT_HALF,
    OKFFT_SQRT_HALF,    -OKFFT_SQRT_HALF,   OKFFT_SQRT_HALF,    -OKFFT_SQRT_HALF,
    1.0f,                1.0f,              OKFFT_SQRT_HALF,     OKFFT_SQRT_HALF,
    0.0f,                0.0f,              OKFFT_SQRT_HALF,    -OKFFT_SQRT_HALF,
};

static const __m128 okfft_sse_fwd_sign_mask = _mm_set_ps(-0.f, 0.f, -0.f, 0.f);
static const __m128 okfft_sse_inv_sign_mask = _mm_set_ps(0.f, -0.f, 0.f, -0.f);

static const OKFFT_ALIGN(32) float okfft_avx_fwd_constants[32] =
{
     OKFFT_SQRT_HALF,    OKFFT_SQRT_HALF,    OKFFT_SQRT_HALF,   OKFFT_SQRT_HALF,     OKFFT_SQRT_HALF,    OKFFT_SQRT_HALF,    OKFFT_SQRT_HALF,   OKFFT_SQRT_HALF,
    -OKFFT_SQRT_HALF,    OKFFT_SQRT_HALF,   -OKFFT_SQRT_HALF,   OKFFT_SQRT_HALF,    -OKFFT_SQRT_HALF,    OKFFT_SQRT_HALF,   -OKFFT_SQRT_HALF,   OKFFT_SQRT_HALF,
     1.0f,               1.0f,               OKFFT_SQRT_HALF,   OKFFT_SQRT_HALF,     1.0f,               1.0f,               OKFFT_SQRT_HALF,   OKFFT_SQRT_HALF,
     0.0f,               0.0f,              -OKFFT_SQRT_HALF,   OKFFT_SQRT_HALF,     0.0f,               0.0f,              -OKFFT_SQRT_HALF,   OKFFT_SQRT_HALF,
};

static const OKFFT_ALIGN(32) float okfft_avx_inv_constants[32] =
{
    OKFFT_SQRT_HALF,     OKFFT_SQRT_HALF,   OKFFT_SQRT_HALF,     OKFFT_SQRT_HALF,   OKFFT_SQRT_HALF,     OKFFT_SQRT_HALF,   OKFFT_SQRT_HALF,     OKFFT_SQRT_HALF,
    OKFFT_SQRT_HALF,    -OKFFT_SQRT_HALF,   OKFFT_SQRT_HALF,    -OKFFT_SQRT_HALF,   OKFFT_SQRT_HALF,    -OKFFT_SQRT_HALF,   OKFFT_SQRT_HALF,    -OKFFT_SQRT_HALF,
    1.0f,                1.0f,              OKFFT_SQRT_HALF,     OKFFT_SQRT_HALF,   1.0f,                1.0f,              OKFFT_SQRT_HALF,     OKFFT_SQRT_HALF,
    0.0f,                0.0f,              OKFFT_SQRT_HALF,    -OKFFT_SQRT_HALF,   0.0f,                0.0f,              OKFFT_SQRT_HALF,    -OKFFT_SQRT_HALF,
};

static const __m256 okfft_avx_fwd_sign_mask = _mm256_set_ps(-0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f);
static const __m256 okfft_avx_inv_sign_mask = _mm256_set_ps(0.f, -0.f, 0.f, -0.f, 0.f, -0.f, 0.f, -0.f);

static okfft_force_inline size_t okfft_avx_ilog2(size_t N)
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

static okfft_force_inline void okfft_avx_xf_fwd_32(const okfft_plan_t *plan, float *__restrict data)
{
    const __m256 avx_sign_mask = okfft_avx_fwd_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;
    const float *__restrict ws1 = ws + (ws_is[1] << 1);
    OKFFT_AVX_XF_32(data);
}

static okfft_force_inline void okfft_avx_xf_fwd_64(const okfft_plan_t *plan, float *__restrict data)
{
    const __m256 avx_sign_mask = okfft_avx_fwd_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;
    OKFFT_AVX_XF_64(data);
}

static okfft_force_inline void okfft_avx_xf_fwd_128(const okfft_plan_t *plan, float *__restrict data)
{
    const __m256 avx_sign_mask = okfft_avx_fwd_sign_mask;
    const ptrdiff_t *ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;
    const float *__restrict ws1 = ws + (ws_is[1] << 1);
    OKFFT_AVX_XF_128(data);
}

static okfft_force_inline void okfft_avx_xf_fwd_256(const okfft_plan_t *plan, float *__restrict data)
{
    const __m256 avx_sign_mask = okfft_avx_fwd_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;
    const float *__restrict ws1 = ws + (ws_is[1] << 1);
    OKFFT_AVX_XF_256(data)
}

static okfft_force_inline void okfft_avx_xf_fwd_512(const okfft_plan_t *plan, float *__restrict data)
{
    const __m256 avx_sign_mask = okfft_avx_fwd_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;
    const float *__restrict ws1 = ws + (ws_is[1] << 1);
    OKFFT_AVX_XF_512(data)
}

static inline void okfft_avx_xf_fwd_1k(const okfft_plan_t *plan, float *__restrict data)
{
    const __m256 avx_sign_mask = okfft_avx_fwd_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;
    const float *__restrict ws1 = ws + (ws_is[1] << 1);
    OKFFT_AVX_XF_1024(data)
}

static inline void okfft_avx_xf_fwd_2k(const okfft_plan_t *plan, float *__restrict data)
{
    const __m256 avx_sign_mask = okfft_avx_fwd_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;

    okfft_avx_xf_fwd_512(plan, data);
    okfft_avx_xf_fwd_256(plan, data + 2 * 512);
    okfft_avx_xf_fwd_256(plan, data + 3 * 512);
    okfft_avx_xf_fwd_512(plan, data + 4 * 512);
    okfft_avx_xf_fwd_512(plan, data + 6 * 512);
    OKFFT_AVX_X8(2 * 1024, data, ws + (ws_is[7] << 1));
}

static inline void okfft_avx_xf_fwd_4k(const okfft_plan_t *plan, float *__restrict data)
{
    const __m256 avx_sign_mask = okfft_avx_fwd_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;

    okfft_avx_xf_fwd_1k(plan, data);
    okfft_avx_xf_fwd_512(plan, data + 2 * 1024);
    okfft_avx_xf_fwd_512(plan, data + 3 * 1024);
    okfft_avx_xf_fwd_1k(plan, data + 4 * 1024);
    okfft_avx_xf_fwd_1k(plan, data + 6 * 1024);
    OKFFT_AVX_X8(4 * 1024, data, ws + (ws_is[8] << 1));
}

static inline void okfft_avx_xf_fwd_8k(const okfft_plan_t *plan, float *__restrict data)
{
    const __m256 avx_sign_mask = okfft_avx_fwd_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;

    okfft_avx_xf_fwd_2k(plan, data);
    okfft_avx_xf_fwd_1k(plan, data + 2 * 2 * 1024);
    okfft_avx_xf_fwd_1k(plan, data + 3 * 2 * 1024);
    okfft_avx_xf_fwd_2k(plan, data + 4 * 2 * 1024);
    okfft_avx_xf_fwd_2k(plan, data + 6 * 2 * 1024);
    OKFFT_AVX_X8(8 * 1024, data, ws + (ws_is[9] << 1));
}

static inline void okfft_avx_xf_fwd_16k(const okfft_plan_t *plan, float *__restrict data)
{
    const __m256 avx_sign_mask = okfft_avx_fwd_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;

    okfft_avx_xf_fwd_4k(plan, data);
    okfft_avx_xf_fwd_2k(plan, data + 2 * 4 * 1024);
    okfft_avx_xf_fwd_2k(plan, data + 3 * 4 * 1024);
    okfft_avx_xf_fwd_4k(plan, data + 4 * 4 * 1024);
    okfft_avx_xf_fwd_4k(plan, data + 6 * 4 * 1024);
    OKFFT_AVX_X8(16 * 1024, data, ws + (ws_is[10] << 1));
}

static inline void okfft_avx_xf_fwd_rec(const okfft_plan_t *plan, float *__restrict data, size_t N)
{
    const __m256 avx_sign_mask = okfft_avx_fwd_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;

    if (N > 64 * 1024)
    {
        size_t N2 = N >> 1;
        size_t N4 = N >> 2;
        size_t N8 = N >> 3;

        okfft_avx_xf_fwd_rec(plan, data, N4);
        okfft_avx_xf_fwd_rec(plan, data + N2, N8);
        okfft_avx_xf_fwd_rec(plan, data + N2 + N4, N8);
        okfft_avx_xf_fwd_rec(plan, data + N, N4);
        okfft_avx_xf_fwd_rec(plan, data + N + N2, N4);
        OKFFT_AVX_X8(N, data, ws + (ws_is[okfft_avx_ilog2(N) - 4] << 1));
    }
    else if (N == 64 * 1024)
    {
        okfft_avx_xf_fwd_16k(plan, data);
        okfft_avx_xf_fwd_8k(plan, data + 2 * 16 * 1024);
        okfft_avx_xf_fwd_8k(plan, data + 3 * 16 * 1024);
        okfft_avx_xf_fwd_16k(plan, data + 4 * 16 * 1024);
        okfft_avx_xf_fwd_16k(plan, data + 6 * 16 * 1024);
        OKFFT_AVX_X8(64 * 1024, data, ws + (ws_is[12] << 1));
    }
    else if (N == 32 * 1024)
    {
        okfft_avx_xf_fwd_8k(plan, data);
        okfft_avx_xf_fwd_4k(plan, data + 2 * 8 * 1024);
        okfft_avx_xf_fwd_4k(plan, data + 3 * 8 * 1024);
        okfft_avx_xf_fwd_8k(plan, data + 4 * 8 * 1024);
        okfft_avx_xf_fwd_8k(plan, data + 6 * 8 * 1024);
        OKFFT_AVX_X8(32 * 1024, data, ws + (ws_is[11] << 1));
    }
    else if (N == 16 * 1024)
        okfft_avx_xf_fwd_16k(plan, data);
}

void okfft_avx_fwd_32(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    _mm256_zeroupper();
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const float *__restrict sse_constants = okfft_sse_fwd_constants;
    OKFFT_SSE_FP_ODD(1, 0, plan, output, input);
    okfft_avx_xf_fwd_32(plan, output);
    _mm256_zeroupper();
}

void okfft_avx_fwd_64(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    _mm256_zeroupper();
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const float *__restrict sse_constants = okfft_sse_fwd_constants;
    OKFFT_SSE_FP_EVEN(1, 1, plan, output, input);
    okfft_avx_xf_fwd_64(plan, output);
    _mm256_zeroupper();
}

void okfft_avx_fwd_128(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    _mm256_zeroupper();
    const __m256 avx_sign_mask = okfft_avx_fwd_sign_mask;
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const float *__restrict sse_constants = okfft_sse_fwd_constants;
    const float *__restrict avx_constants = okfft_avx_fwd_constants;
    OKFFT_AVX_FP_ODD(3, 2, plan, output, input);
    okfft_avx_xf_fwd_128(plan, output);
    _mm256_zeroupper();
}

void okfft_avx_fwd_256(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    _mm256_zeroupper();
    const __m256 avx_sign_mask = okfft_avx_fwd_sign_mask;
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const float *__restrict sse_constants = okfft_sse_fwd_constants;
    const float *__restrict avx_constants = okfft_avx_fwd_constants;
    OKFFT_AVX_FP_EVEN(5, 5, plan, output, input);
    okfft_avx_xf_fwd_256(plan, output);
    _mm256_zeroupper();
}

void okfft_avx_fwd_512(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    _mm256_zeroupper();
    const __m256 avx_sign_mask = okfft_avx_fwd_sign_mask;
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const float *__restrict sse_constants = okfft_sse_fwd_constants;
    const float *__restrict avx_constants = okfft_avx_fwd_constants;
    OKFFT_AVX_FP_ODD(11, 10, plan, output, input);
    okfft_avx_xf_fwd_512(plan, output);
    _mm256_zeroupper();
}

void okfft_avx_fwd_1024(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    _mm256_zeroupper();
    const __m256 avx_sign_mask = okfft_avx_fwd_sign_mask;
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const float *__restrict sse_constants = okfft_sse_fwd_constants;
    const float *__restrict avx_constants = okfft_avx_fwd_constants;
    OKFFT_AVX_FP_EVEN(21, 21, plan, output, input);
    okfft_avx_xf_fwd_1k(plan, output);
    _mm256_zeroupper();
}

void okfft_avx_fwd_2048(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    _mm256_zeroupper();
    const __m256 avx_sign_mask = okfft_avx_fwd_sign_mask;
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const float *__restrict sse_constants = okfft_sse_fwd_constants;
    const float *__restrict avx_constants = okfft_avx_fwd_constants;
    OKFFT_AVX_FP_ODD(43, 42, plan, output, input);
    okfft_avx_xf_fwd_2k(plan, output);
    _mm256_zeroupper();
}

void okfft_avx_fwd_4096(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    _mm256_zeroupper();
    const __m256 avx_sign_mask = okfft_avx_fwd_sign_mask;
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const float *__restrict sse_constants = okfft_sse_fwd_constants;
    const float *__restrict avx_constants = okfft_avx_fwd_constants;
    OKFFT_AVX_FP_EVEN(85, 85, plan, output, input);
    okfft_avx_xf_fwd_4k(plan, output);
    _mm256_zeroupper();
}

void okfft_avx_fwd_8192(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    _mm256_zeroupper();
    const __m256 avx_sign_mask = okfft_avx_fwd_sign_mask;
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const float *__restrict sse_constants = okfft_sse_fwd_constants;
    const float *__restrict avx_constants = okfft_avx_fwd_constants;
    OKFFT_AVX_FP_ODD(171, 170, plan, output, input);
    okfft_avx_xf_fwd_8k(plan, output);
    _mm256_zeroupper();
}

void okfft_avx_fwd_generic(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    _mm256_zeroupper();
    const __m256 avx_sign_mask = okfft_avx_fwd_sign_mask;
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const float *__restrict sse_constants = okfft_sse_fwd_constants;
    const float *__restrict avx_constants = okfft_avx_fwd_constants;

    size_t i0 = plan->i0, i1 = plan->i1;
    if (okfft_avx_ilog2(plan->N) & 1) // check if ilog2(N) is odd
    {
        OKFFT_AVX_FP_ODD(i0, i1, plan, output, input);
    }
    else
    {
        OKFFT_AVX_FP_EVEN(i0, i1, plan, output, input);
    }

    okfft_avx_xf_fwd_rec(plan, output, plan->N);
    _mm256_zeroupper();
}

void okfft_avx_fwd_real(float *__restrict output, float *__restrict buffer, const float *__restrict A, const float *__restrict B, size_t N)
{
    _mm256_zeroupper();
    buffer[N + 0] = buffer[0];
    buffer[N + 1] = buffer[1];

    for (size_t i = 0; i < N; i += 32)
    {
        __m256 x00 = _mm256_load_ps(buffer + i +  0);
        __m256 x10 = _mm256_load_ps(buffer + i +  8);
        __m256 x01 = _mm256_load_ps(buffer + i + 16);
        __m256 x11 = _mm256_load_ps(buffer + i + 24);

        __m256 y00 = _mm256_loadu_ps(buffer + N - i -  6);
        __m256 y10 = _mm256_loadu_ps(buffer + N - i - 14);
        __m256 y01 = _mm256_loadu_ps(buffer + N - i - 22);
        __m256 y11 = _mm256_loadu_ps(buffer + N - i - 30);

        __m256 xre0 = _mm256_shuffle_ps(x00, x10, _MM_SHUFFLE(2, 0, 2, 0));
        __m256 xim0 = _mm256_shuffle_ps(x00, x10, _MM_SHUFFLE(3, 1, 3, 1));
        __m256 yre0 = _mm256_shuffle_ps(y00, y10, _MM_SHUFFLE(0, 2, 0, 2));
        __m256 yim0 = _mm256_shuffle_ps(y00, y10, _MM_SHUFFLE(1, 3, 1, 3));

        __m256 xre1 = _mm256_shuffle_ps(x01, x11, _MM_SHUFFLE(2, 0, 2, 0));
        __m256 xim1 = _mm256_shuffle_ps(x01, x11, _MM_SHUFFLE(3, 1, 3, 1));
        __m256 yre1 = _mm256_shuffle_ps(y01, y11, _MM_SHUFFLE(0, 2, 0, 2));
        __m256 yim1 = _mm256_shuffle_ps(y01, y11, _MM_SHUFFLE(1, 3, 1, 3));

        yre0 = _mm256_permute2f128_ps(yre0, yre0, 1);
        yim0 = _mm256_permute2f128_ps(yim0, yim0, 1);
        yre1 = _mm256_permute2f128_ps(yre1, yre1, 1);
        yim1 = _mm256_permute2f128_ps(yim1, yim1, 1);

        __m256 are0 = _mm256_load_ps(A + i +  0);
        __m256 aim0 = _mm256_load_ps(A + i +  8);
        __m256 are1 = _mm256_load_ps(A + i + 16);
        __m256 aim1 = _mm256_load_ps(A + i + 24);

        __m256 bre0 = _mm256_load_ps(B + i +  0);
        __m256 bim0 = _mm256_load_ps(B + i +  8);
        __m256 bre1 = _mm256_load_ps(B + i + 16);
        __m256 bim1 = _mm256_load_ps(B + i + 24);

        __m256 m000 = _mm256_mul_ps(xre0, are0);
        __m256 m100 = _mm256_mul_ps(xim0, aim0);
        __m256 m200 = _mm256_mul_ps(yre0, bre0);
        __m256 m300 = _mm256_mul_ps(yim0, bim0);

        __m256 m010 = _mm256_mul_ps(xim0, are0);
        __m256 m110 = _mm256_mul_ps(xre0, aim0);
        __m256 m210 = _mm256_mul_ps(yre0, bim0);
        __m256 m310 = _mm256_mul_ps(yim0, bre0);

        __m256 m001 = _mm256_mul_ps(xre1, are1);
        __m256 m101 = _mm256_mul_ps(xim1, aim1);
        __m256 m201 = _mm256_mul_ps(yre1, bre1);
        __m256 m301 = _mm256_mul_ps(yim1, bim1);

        __m256 m011 = _mm256_mul_ps(xim1, are1);
        __m256 m111 = _mm256_mul_ps(xre1, aim1);
        __m256 m211 = _mm256_mul_ps(yre1, bim1);
        __m256 m311 = _mm256_mul_ps(yim1, bre1);
        
        __m256 re00 = _mm256_sub_ps(m000, m100);
        __m256 re10 = _mm256_add_ps(m200, m300);
        __m256 im00 = _mm256_add_ps(m010, m110);
        __m256 im10 = _mm256_sub_ps(m210, m310);

        __m256 re01 = _mm256_sub_ps(m001, m101);
        __m256 re11 = _mm256_add_ps(m201, m301);
        __m256 im01 = _mm256_add_ps(m011, m111);
        __m256 im11 = _mm256_sub_ps(m211, m311);

        __m256 re0  = _mm256_add_ps(re00, re10);
        __m256 im0  = _mm256_add_ps(im00, im10);

        __m256 re1  = _mm256_add_ps(re01, re11);
        __m256 im1  = _mm256_add_ps(im01, im11);

        __m256 o00  = _mm256_unpacklo_ps(re0, im0);
        __m256 o10  = _mm256_unpackhi_ps(re0, im0);

        __m256 o01  = _mm256_unpacklo_ps(re1, im1);
        __m256 o11  = _mm256_unpackhi_ps(re1, im1);

        _mm256_store_ps(output + i +  0, o00);
        _mm256_store_ps(output + i +  8, o10);
        _mm256_store_ps(output + i + 16, o01);
        _mm256_store_ps(output + i + 24, o11);
    }
    
    output[N + 0] = buffer[0] - buffer[1];
    output[N + 1] = 0.0f;

    _mm256_zeroupper();
}

// ================= BACKWARDS ==================================

static okfft_force_inline void okfft_avx_xf_inv_32(const okfft_plan_t *plan, float *__restrict data)
{
    const __m256 avx_sign_mask = okfft_avx_inv_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;
    const float *__restrict ws1 = ws + (ws_is[1] << 1);
    OKFFT_AVX_XF_32(data);
}

static okfft_force_inline void okfft_avx_xf_inv_64(const okfft_plan_t *plan, float *__restrict data)
{
    const __m256 avx_sign_mask = okfft_avx_inv_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;
    OKFFT_AVX_XF_64(data);
}

static okfft_force_inline void okfft_avx_xf_inv_128(const okfft_plan_t *plan, float *__restrict data)
{
    const __m256 avx_sign_mask = okfft_avx_inv_sign_mask;
    const ptrdiff_t *ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;
    const float *__restrict ws1 = ws + (ws_is[1] << 1);
    OKFFT_AVX_XF_128(data);
}

static okfft_force_inline void okfft_avx_xf_inv_256(const okfft_plan_t *plan, float *__restrict data)
{
    const __m256 avx_sign_mask = okfft_avx_inv_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;
    const float *__restrict ws1 = ws + (ws_is[1] << 1);
    OKFFT_AVX_XF_256(data)
}

static okfft_force_inline void okfft_avx_xf_inv_512(const okfft_plan_t *plan, float *__restrict data)
{
    const __m256 avx_sign_mask = okfft_avx_inv_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;
    const float *__restrict ws1 = ws + (ws_is[1] << 1);
    OKFFT_AVX_XF_512(data)
}

static inline void okfft_avx_xf_inv_1k(const okfft_plan_t *plan, float *__restrict data)
{
    const __m256 avx_sign_mask = okfft_avx_inv_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;
    const float *__restrict ws1 = ws + (ws_is[1] << 1);
    OKFFT_AVX_XF_1024(data)
}

static inline void okfft_avx_xf_inv_2k(const okfft_plan_t *plan, float *__restrict data)
{
    const __m256 avx_sign_mask = okfft_avx_inv_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;

    okfft_avx_xf_inv_512(plan, data);
    okfft_avx_xf_inv_256(plan, data + 2 * 512);
    okfft_avx_xf_inv_256(plan, data + 3 * 512);
    okfft_avx_xf_inv_512(plan, data + 4 * 512);
    okfft_avx_xf_inv_512(plan, data + 6 * 512);
    OKFFT_AVX_X8(2 * 1024, data, ws + (ws_is[7] << 1));
}

static inline void okfft_avx_xf_inv_4k(const okfft_plan_t *plan, float *__restrict data)
{
    const __m256 avx_sign_mask = okfft_avx_inv_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;

    okfft_avx_xf_inv_1k(plan, data);
    okfft_avx_xf_inv_512(plan, data + 2 * 1024);
    okfft_avx_xf_inv_512(plan, data + 3 * 1024);
    okfft_avx_xf_inv_1k(plan, data + 4 * 1024);
    okfft_avx_xf_inv_1k(plan, data + 6 * 1024);
    OKFFT_AVX_X8(4 * 1024, data, ws + (ws_is[8] << 1));
}

static inline void okfft_avx_xf_inv_8k(const okfft_plan_t *plan, float *__restrict data)
{
    const __m256 avx_sign_mask = okfft_avx_inv_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;

    okfft_avx_xf_inv_2k(plan, data);
    okfft_avx_xf_inv_1k(plan, data + 2 * 2 * 1024);
    okfft_avx_xf_inv_1k(plan, data + 3 * 2 * 1024);
    okfft_avx_xf_inv_2k(plan, data + 4 * 2 * 1024);
    okfft_avx_xf_inv_2k(plan, data + 6 * 2 * 1024);
    OKFFT_AVX_X8(8 * 1024, data, ws + (ws_is[9] << 1));
}

static inline void okfft_avx_xf_inv_16k(const okfft_plan_t *plan, float *__restrict data)
{
    const __m256 avx_sign_mask = okfft_avx_inv_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;

    okfft_avx_xf_inv_4k(plan, data);
    okfft_avx_xf_inv_2k(plan, data + 2 * 4 * 1024);
    okfft_avx_xf_inv_2k(plan, data + 3 * 4 * 1024);
    okfft_avx_xf_inv_4k(plan, data + 4 * 4 * 1024);
    okfft_avx_xf_inv_4k(plan, data + 6 * 4 * 1024);
    OKFFT_AVX_X8(16 * 1024, data, ws + (ws_is[10] << 1));
}

static inline void okfft_avx_xf_inv_rec(const okfft_plan_t *plan, float *__restrict data, size_t N)
{
    const __m256 avx_sign_mask = okfft_avx_inv_sign_mask;
    const ptrdiff_t *__restrict ws_is = plan->ws_is;
    const float *__restrict ws = plan->ws;

    if (N > 64 * 1024)
    {
        size_t N2 = N >> 1;
        size_t N4 = N >> 2;
        size_t N8 = N >> 3;

        okfft_avx_xf_inv_rec(plan, data, N4);
        okfft_avx_xf_inv_rec(plan, data + N2, N8);
        okfft_avx_xf_inv_rec(plan, data + N2 + N4, N8);
        okfft_avx_xf_inv_rec(plan, data + N, N4);
        okfft_avx_xf_inv_rec(plan, data + N + N2, N4);
        OKFFT_AVX_X8(N, data, ws + (ws_is[okfft_avx_ilog2(N) - 4] << 1));
    }
    else if (N == 64 * 1024)
    {
        okfft_avx_xf_inv_16k(plan, data);
        okfft_avx_xf_inv_8k(plan, data + 2 * 16 * 1024);
        okfft_avx_xf_inv_8k(plan, data + 3 * 16 * 1024);
        okfft_avx_xf_inv_16k(plan, data + 4 * 16 * 1024);
        okfft_avx_xf_inv_16k(plan, data + 6 * 16 * 1024);
        OKFFT_AVX_X8(64 * 1024, data, ws + (ws_is[12] << 1));
    }
    else if (N == 32 * 1024)
    {
        okfft_avx_xf_inv_8k(plan, data);
        okfft_avx_xf_inv_4k(plan, data + 2 * 8 * 1024);
        okfft_avx_xf_inv_4k(plan, data + 3 * 8 * 1024);
        okfft_avx_xf_inv_8k(plan, data + 4 * 8 * 1024);
        okfft_avx_xf_inv_8k(plan, data + 6 * 8 * 1024);
        OKFFT_AVX_X8(32 * 1024, data, ws + (ws_is[11] << 1));
    }
    else if (N == 16 * 1024)
        okfft_avx_xf_inv_16k(plan, data);
}

void okfft_avx_inv_32(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    _mm256_zeroupper();
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const float *__restrict sse_constants = okfft_sse_inv_constants;
    OKFFT_SSE_FP_ODD(1, 0, plan, output, input);
    okfft_avx_xf_inv_32(plan, output);
    _mm256_zeroupper();
}

void okfft_avx_inv_64(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    _mm256_zeroupper();
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const float *__restrict sse_constants = okfft_sse_inv_constants;
    OKFFT_SSE_FP_EVEN(1, 1, plan, output, input);
    okfft_avx_xf_inv_64(plan, output);
    _mm256_zeroupper();
}

void okfft_avx_inv_128(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    _mm256_zeroupper();
    const __m256 avx_sign_mask = okfft_avx_inv_sign_mask;
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const float *__restrict sse_constants = okfft_sse_inv_constants;
    const float *__restrict avx_constants = okfft_avx_inv_constants;
    OKFFT_AVX_FP_ODD(3, 2, plan, output, input);
    okfft_avx_xf_inv_128(plan, output);
    _mm256_zeroupper();
}

void okfft_avx_inv_256(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    _mm256_zeroupper();
    const __m256 avx_sign_mask = okfft_avx_inv_sign_mask;
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const float *__restrict sse_constants = okfft_sse_inv_constants;
    const float *__restrict avx_constants = okfft_avx_inv_constants;
    OKFFT_AVX_FP_EVEN(5, 5, plan, output, input);
    okfft_avx_xf_inv_256(plan, output);
    _mm256_zeroupper();
}

void okfft_avx_inv_512(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    _mm256_zeroupper();
    const __m256 avx_sign_mask = okfft_avx_inv_sign_mask;
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const float *__restrict sse_constants = okfft_sse_inv_constants;
    const float *__restrict avx_constants = okfft_avx_inv_constants;
    OKFFT_AVX_FP_ODD(11, 11, plan, output, input);
    okfft_avx_xf_inv_512(plan, output);
    _mm256_zeroupper();
}

void okfft_avx_inv_1024(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    _mm256_zeroupper();
    const __m256 avx_sign_mask = okfft_avx_inv_sign_mask;
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const float *__restrict sse_constants = okfft_sse_inv_constants;
    const float *__restrict avx_constants = okfft_avx_inv_constants;
    OKFFT_AVX_FP_EVEN(21, 21, plan, output, input);
    okfft_avx_xf_inv_1k(plan, output);
    _mm256_zeroupper();
}

void okfft_avx_inv_2048(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    _mm256_zeroupper();
    const __m256 avx_sign_mask = okfft_avx_inv_sign_mask;
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const float *__restrict sse_constants = okfft_sse_inv_constants;
    const float *__restrict avx_constants = okfft_avx_inv_constants;
    OKFFT_AVX_FP_ODD(43, 42, plan, output, input);
    okfft_avx_xf_inv_2k(plan, output);
    _mm256_zeroupper();
}

void okfft_avx_inv_4096(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    _mm256_zeroupper();
    const __m256 avx_sign_mask = okfft_avx_inv_sign_mask;
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const float *__restrict sse_constants = okfft_sse_inv_constants;
    const float *__restrict avx_constants = okfft_avx_inv_constants;
    OKFFT_AVX_FP_EVEN(85, 85, plan, output, input);
    okfft_avx_xf_inv_4k(plan, output);
    _mm256_zeroupper();
}

void okfft_avx_inv_8192(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    _mm256_zeroupper();
    const __m256 avx_sign_mask = okfft_avx_inv_sign_mask;
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const float *__restrict sse_constants = okfft_sse_inv_constants;
    const float *__restrict avx_constants = okfft_avx_inv_constants;
    OKFFT_AVX_FP_ODD(171, 170, plan, output, input);
    okfft_avx_xf_inv_8k(plan, output);
    _mm256_zeroupper();
}

void okfft_avx_inv_generic(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    _mm256_zeroupper();
    const __m256 avx_sign_mask = okfft_avx_inv_sign_mask;
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const float *__restrict sse_constants = okfft_sse_inv_constants;
    const float *__restrict avx_constants = okfft_avx_inv_constants;
    size_t i0 = plan->i0, i1 = plan->i1;
    if (okfft_avx_ilog2(plan->N) & 1) // check if ilog2(N) is odd
    {
        OKFFT_AVX_FP_ODD(i0, i1, plan, output, input)
    }
    else
    {
        OKFFT_AVX_FP_EVEN(i0, i1, plan, output, input)
    }

    okfft_avx_xf_inv_rec(plan, output, plan->N);
    _mm256_zeroupper();
}

void okfft_avx_inv_real(float *__restrict output, const float *__restrict input, const float *__restrict A, const float *__restrict B, size_t N)
{
    _mm256_zeroupper();


    for (size_t i = 0; i < N; i += 32)
    {
        __m256 x00 = _mm256_load_ps(input + i + 0);
        __m256 x10 = _mm256_load_ps(input + i + 8);
        __m256 x01 = _mm256_load_ps(input + i + 16);
        __m256 x11 = _mm256_load_ps(input + i + 24);

        __m256 y00 = _mm256_loadu_ps(input + N - i - 6);
        __m256 y10 = _mm256_loadu_ps(input + N - i - 14);
        __m256 y01 = _mm256_loadu_ps(input + N - i - 22);
        __m256 y11 = _mm256_loadu_ps(input + N - i - 30);

        __m256 xre0 = _mm256_shuffle_ps(x00, x10, _MM_SHUFFLE(2, 0, 2, 0));
        __m256 xim0 = _mm256_shuffle_ps(x00, x10, _MM_SHUFFLE(3, 1, 3, 1));
        __m256 yre0 = _mm256_shuffle_ps(y00, y10, _MM_SHUFFLE(0, 2, 0, 2));
        __m256 yim0 = _mm256_shuffle_ps(y00, y10, _MM_SHUFFLE(1, 3, 1, 3));

        __m256 xre1 = _mm256_shuffle_ps(x01, x11, _MM_SHUFFLE(2, 0, 2, 0));
        __m256 xim1 = _mm256_shuffle_ps(x01, x11, _MM_SHUFFLE(3, 1, 3, 1));
        __m256 yre1 = _mm256_shuffle_ps(y01, y11, _MM_SHUFFLE(0, 2, 0, 2));
        __m256 yim1 = _mm256_shuffle_ps(y01, y11, _MM_SHUFFLE(1, 3, 1, 3));

        yre0 = _mm256_permute2f128_ps(yre0, yre0, 1);
        yim0 = _mm256_permute2f128_ps(yim0, yim0, 1);
        yre1 = _mm256_permute2f128_ps(yre1, yre1, 1);
        yim1 = _mm256_permute2f128_ps(yim1, yim1, 1);

        __m256 are0 = _mm256_load_ps(A + i + 0);
        __m256 aim0 = _mm256_load_ps(A + i + 8);
        __m256 are1 = _mm256_load_ps(A + i + 16);
        __m256 aim1 = _mm256_load_ps(A + i + 24);

        __m256 bre0 = _mm256_load_ps(B + i + 0);
        __m256 bim0 = _mm256_load_ps(B + i + 8);
        __m256 bre1 = _mm256_load_ps(B + i + 16);
        __m256 bim1 = _mm256_load_ps(B + i + 24);

        __m256 m000 = _mm256_mul_ps(xre0, are0);
        __m256 m100 = _mm256_mul_ps(xim0, aim0);
        __m256 m200 = _mm256_mul_ps(yre0, bre0);
        __m256 m300 = _mm256_mul_ps(yim0, bim0);

        __m256 m010 = _mm256_mul_ps(xim0, are0);
        __m256 m110 = _mm256_mul_ps(xre0, aim0);
        __m256 m210 = _mm256_mul_ps(yre0, bim0);
        __m256 m310 = _mm256_mul_ps(yim0, bre0);

        __m256 m001 = _mm256_mul_ps(xre1, are1);
        __m256 m101 = _mm256_mul_ps(xim1, aim1);
        __m256 m201 = _mm256_mul_ps(yre1, bre1);
        __m256 m301 = _mm256_mul_ps(yim1, bim1);

        __m256 m011 = _mm256_mul_ps(xim1, are1);
        __m256 m111 = _mm256_mul_ps(xre1, aim1);
        __m256 m211 = _mm256_mul_ps(yre1, bim1);
        __m256 m311 = _mm256_mul_ps(yim1, bre1);

        __m256 re00 = _mm256_add_ps(m000, m100);
        __m256 re10 = _mm256_sub_ps(m200, m300);
        __m256 im00 = _mm256_sub_ps(m010, m110);
        __m256 im10 = _mm256_add_ps(m210, m310);

        __m256 re01 = _mm256_add_ps(m001, m101);
        __m256 re11 = _mm256_sub_ps(m201, m301);
        __m256 im01 = _mm256_sub_ps(m011, m111);
        __m256 im11 = _mm256_add_ps(m211, m311);

        __m256 re0 = _mm256_add_ps(re00, re10);
        __m256 im0 = _mm256_sub_ps(im00, im10);

        __m256 re1 = _mm256_add_ps(re01, re11);
        __m256 im1 = _mm256_sub_ps(im01, im11);

        __m256 o00 = _mm256_unpacklo_ps(re0, im0);
        __m256 o10 = _mm256_unpackhi_ps(re0, im0);

        __m256 o01 = _mm256_unpacklo_ps(re1, im1);
        __m256 o11 = _mm256_unpackhi_ps(re1, im1);

        _mm256_store_ps(output + i +  0, o00);
        _mm256_store_ps(output + i +  8, o10);
        _mm256_store_ps(output + i + 16, o01);
        _mm256_store_ps(output + i + 24, o11);
    }

    _mm256_zeroupper();
}

#endif