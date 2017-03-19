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

#include <stdio.h>  // for printf (default log)
#include <stdlib.h> // for qsort
#include <string.h> // for memset

#ifdef _MSC_VER
    #include <intrin.h>
    #define okfft_force_inline __forceinline
    #define OKFFT_ALIGN(x) __declspec(align(x))
#else
    #include <x86intrin.h>
    #define okfft_force_inline inline __attribute__((always_inline))
    #define OKFFT_ALIGN(x) __attribute__((aligned(x)))
#endif

// predefined xform prototypes (implementations are found in okfft_xf_sse.cpp, and okfft_xf_avx.cpp)
#ifdef OKFFT_HAS_AVX

void okfft_avx_fwd_32(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_avx_fwd_64(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_avx_fwd_128(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_avx_fwd_256(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_avx_fwd_512(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_avx_fwd_1024(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_avx_fwd_2048(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_avx_fwd_4096(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_avx_fwd_8192(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_avx_fwd_generic(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);

void okfft_avx_fwd_real(float *__restrict output, float *__restrict buffer, const float *__restrict A, const float *__restrict B, size_t N);

void okfft_avx_inv_32(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_avx_inv_64(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_avx_inv_128(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_avx_inv_256(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_avx_inv_512(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_avx_inv_1024(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_avx_inv_2048(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_avx_inv_4096(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_avx_inv_8192(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_avx_inv_generic(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);

void okfft_avx_inv_real(float *__restrict output, const float *__restrict buffer, const float *__restrict A, const float *__restrict B, size_t N);

#endif

#ifdef OKFFT_HAS_SSE

void okfft_sse_fwd_32(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_sse_fwd_64(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_sse_fwd_128(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_sse_fwd_256(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_sse_fwd_512(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_sse_fwd_1024(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_sse_fwd_2048(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_sse_fwd_4096(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_sse_fwd_8192(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_sse_fwd_generic(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);

void okfft_sse_fwd_real(float *__restrict output, float *__restrict buffer, const float *__restrict A, const float *__restrict B, size_t N);

void okfft_sse_inv_32(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_sse_inv_64(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_sse_inv_128(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_sse_inv_256(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_sse_inv_512(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_sse_inv_1024(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_sse_inv_2048(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_sse_inv_4096(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_sse_inv_8192(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);
void okfft_sse_inv_generic(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);

void okfft_sse_inv_real(float *__restrict output, const float *__restrict buffer, const float *__restrict A, const float *__restrict B, size_t N);

#endif

#define OKFFT_FLAG_INVERSE_XFORM    1
#define OKFFT_FLAG_AVX              2

static void okfft_init_offsets(okfft_plan_t *p, size_t N);
static void okfft_init_indices(okfft_plan_t *p, size_t N);
static void okfft_init_twiddles(okfft_plan_t *p, size_t N, bool is_inverse);
static void okfft_init_real_coeffs(okfft_plan_t *p, size_t N, bool is_inverse);

static const size_t leaf_N = 8;

#ifdef OKFFT_HAS_AVX
static bool okfft_cpu_has_avx()
{
    int data[4];
    #ifdef _MSC_VER
        __cpuid(data, 1);    
    #else
        #define __cpuid(func,ax,bx,cx,dx)\
	        __asm__ __volatile__ ("cpuid":\
	            "=a" (ax), "=b" (bx), "=c" (cx), "=d" (dx) : "a" (func));
        __cpuid(1, data[0], data[1], data[2], data[3]);
    #endif
    return (data[2] & (1 << 28)) != 0;
}
#endif

okfft_plan_t *okfft_create_plan(size_t N, OKFFT_DIRECTION dir)
{
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    
    if (!(dir == OKFFT_DIR_FORWARD || dir == OKFFT_DIR_INVERSE))
    {
        OKFFT_LOG("Invalid direction for FFT (expected 'OKFFT_DIR_FORWARD' or 'OKFFT_DIR_INVERSE').");
        return NULL;
    }

    if (N < 32)
    {
        OKFFT_LOG("Minimum FFT size is 32, size %zu provided.", N);
        return NULL;
    }

    if ((N & (N - 1)))
    {
        OKFFT_LOG("FFT size must be a power of two. Size %zu provided.", N);
        return NULL;
    }

    #if defined(OKFFT_HAS_AVX) && !defined(OKFFT_HAS_SSE)
    if (!okfft_cpu_has_avx())
    {
        OKFFT_LOG("Cpu does not support AVX, but OKFFT was built without SSE support.");
        return NULL;
    }
    #endif

    okfft_plan_t *plan = (okfft_plan_t *) OKFFT_ALLOC_PLAN(sizeof(*plan));
    memset(plan, 0, sizeof(*plan));

    plan->N = N;

    size_t imm = N / leaf_N / 3;
    plan->i0 = imm + 1;
    plan->i1 = imm;

    if (((N / leaf_N) % 3) > 1)
        plan->i1++;

    plan->i0 /= 2;
    plan->i1 /= 2;

    if (dir == OKFFT_DIR_INVERSE)
        plan->flags |= OKFFT_FLAG_INVERSE_XFORM;

    #ifdef OKFFT_HAS_AVX
    if (okfft_cpu_has_avx())
        plan->flags |= OKFFT_FLAG_AVX;
    #endif

    okfft_init_offsets(plan, N);
    okfft_init_indices(plan, N);
    okfft_init_twiddles(plan, N, dir == OKFFT_DIR_INVERSE);

    if (dir == OKFFT_DIR_FORWARD)
    {
        #ifdef OKFFT_HAS_AVX
        if (okfft_cpu_has_avx())
        {
            switch (N)
            {
                case   32: plan->xform = okfft_avx_fwd_32;      break;
                case   64: plan->xform = okfft_avx_fwd_64;      break;
                case  128: plan->xform = okfft_avx_fwd_128;     break;
                case  256: plan->xform = okfft_avx_fwd_256;     break;
                case  512: plan->xform = okfft_avx_fwd_512;     break;
                case 1024: plan->xform = okfft_avx_fwd_1024;    break;
                case 2048: plan->xform = okfft_avx_fwd_2048;    break;
                case 4096: plan->xform = okfft_avx_fwd_4096;    break;
                case 8192: plan->xform = okfft_avx_fwd_8192;    break;
                default:   plan->xform = okfft_avx_fwd_generic; break;
            }
        }
        else
        #endif
        {
        #ifdef OKFFT_HAS_SSE
            switch (N)
            {
                case   32: plan->xform = okfft_sse_fwd_32;      break;
                case   64: plan->xform = okfft_sse_fwd_64;      break;
                case  128: plan->xform = okfft_sse_fwd_128;     break;
                case  256: plan->xform = okfft_sse_fwd_256;     break;
                case  512: plan->xform = okfft_sse_fwd_512;     break;
                case 1024: plan->xform = okfft_sse_fwd_1024;    break;
                case 2048: plan->xform = okfft_sse_fwd_2048;    break;
                case 4096: plan->xform = okfft_sse_fwd_4096;    break;
                case 8192: plan->xform = okfft_sse_fwd_8192;    break;
                default:   plan->xform = okfft_sse_fwd_generic; break;
            }
        #endif
        }

    }
    else
    {
        #ifdef OKFFT_HAS_AVX
        if (okfft_cpu_has_avx())
        {
            switch (N)
            {
                case   32: plan->xform = okfft_avx_inv_32;      break;
                case   64: plan->xform = okfft_avx_inv_64;      break;
                case  128: plan->xform = okfft_avx_inv_128;     break;
                case  256: plan->xform = okfft_avx_inv_256;     break;
                case  512: plan->xform = okfft_avx_inv_512;     break;
                case 1024: plan->xform = okfft_avx_inv_1024;    break;
                case 2048: plan->xform = okfft_avx_inv_2048;    break;
                case 4096: plan->xform = okfft_avx_inv_4096;    break;
                case 8192: plan->xform = okfft_avx_inv_8192;    break;
                default:   plan->xform = okfft_avx_inv_generic; break;
            }
        }
        else
        #endif
        {
            #ifdef OKFFT_HAS_SSE
            switch (N)
            {
                case   32: plan->xform = okfft_sse_inv_32;      break;
                case   64: plan->xform = okfft_sse_inv_64;      break;
                case  128: plan->xform = okfft_sse_inv_128;     break;
                case  256: plan->xform = okfft_sse_inv_256;     break;
                case  512: plan->xform = okfft_sse_inv_512;     break;
                case 1024: plan->xform = okfft_sse_inv_1024;    break;
                case 2048: plan->xform = okfft_sse_inv_2048;    break;
                case 4096: plan->xform = okfft_sse_inv_4096;    break;
                case 8192: plan->xform = okfft_sse_inv_8192;    break;
                default:   plan->xform = okfft_sse_inv_generic; break;
            }
            #endif
        }
    }

    return plan;
}

okfft_plan_t *okfft_create_plan_real(size_t N, OKFFT_DIRECTION dir)
{
    okfft_plan_t *plan = okfft_create_plan(N / 2, dir);

    if (plan)
        okfft_init_real_coeffs(plan, N, dir == OKFFT_DIR_INVERSE);

    return plan;
}

okfft_buffer_t okfft_create_buffer(size_t N)
{
    okfft_buffer_t s = { (float *) OKFFT_ALLOC_BUFFER((N + 2) * sizeof(float)) };
    return s;
}

void okfft_destroy_buffer(okfft_buffer_t *s)
{
    OKFFT_FREE_BUFFER(s->buffer);
    s->buffer = NULL;
}

void okfft_destroy_plan(okfft_plan_t *plan)
{
    OKFFT_FREE_ALIGNED_DATA(plan->ws);
    OKFFT_FREE_ALIGNED_DATA(plan->ws_is);
    OKFFT_FREE_DATA(plan->offsets);
    
    if (plan->A)
    {
        // real xform, free real coeffs
        OKFFT_FREE_ALIGNED_DATA(plan->A);
        OKFFT_FREE_ALIGNED_DATA(plan->B);
    }

    memset(plan, 0, sizeof(*plan));
}

void okfft_execute(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input)
{
    plan->xform(plan, output, input);
}

void okfft_execute_real(const okfft_plan_t *plan, okfft_buffer_t *state, float *__restrict output, const float *__restrict input)
{
    #if OKFFT_HAS_AVX 
    if (plan->flags & OKFFT_FLAG_AVX)
    {
        if (plan->flags & OKFFT_FLAG_INVERSE_XFORM)
        {
            okfft_avx_inv_real(state->buffer, input, plan->A, plan->B, plan->N << 1);
            plan->xform(plan, output, state->buffer);
        }
        else
        {
            plan->xform(plan, state->buffer, input);
            okfft_avx_fwd_real(output, state->buffer, plan->A, plan->B, plan->N << 1);
        }
    }
    else
    #endif
    {
        #ifdef OKFFT_HAS_SSE
        if (plan->flags & OKFFT_FLAG_INVERSE_XFORM)
        {
            okfft_sse_inv_real(state->buffer, input, plan->A, plan->B, plan->N << 1);
            plan->xform(plan, output, state->buffer);
        }
        else
        {
            plan->xform(plan, state->buffer, input);
            okfft_sse_fwd_real(output, state->buffer, plan->A, plan->B, plan->N << 1);
        }
        #endif
    }
}

// calculation functions

static void okfft_elab_odd(ptrdiff_t *const offs, size_t N, ptrdiff_t in_offs, ptrdiff_t out_offs, ptrdiff_t stride)
{
    if (N <= 16)
    {
        offs[(out_offs / 4) + 0] = in_offs * 2;
        offs[(out_offs / 4) + 1] = out_offs;

        if (N == 16)
        {
            offs[(out_offs / 4) + 2] = (in_offs + stride) * 2;
            offs[(out_offs / 4) + 3] = out_offs + 8;
        }
    }
    else
    {
        okfft_elab_odd(offs, N / 2, in_offs,          out_offs,               stride * 2);
        okfft_elab_odd(offs, N / 4, in_offs + stride, out_offs +     (N / 2), stride * 4);
        okfft_elab_odd(offs, N / 4, in_offs - stride, out_offs + 3 * (N / 4), stride * 4);
    }
}

static void okfft_elab_even(ptrdiff_t *const offs, ptrdiff_t N)
{
    offs[0] = 0;
    offs[1] = 0;
    offs[2] = N / 8;
    offs[3] = 8;
    offs[4] = (N / 16);
    offs[5] = 16;
    offs[6] = -(N / 16);
    offs[7] = 24;

    ptrdiff_t stride = 1;
    for (; N > 32; N /= 2, stride *= 2)
    {
        okfft_elab_odd(offs, N / 4,  stride,     (N / 2), stride * 4);
        okfft_elab_odd(offs, N / 4, -stride, 3 * (N / 4), stride * 4);
    }
}

static int okfft_offset_cmp(const void *pa, const void *pb)
{
    ptrdiff_t a = *(const ptrdiff_t *)pa;
    ptrdiff_t b = *(const ptrdiff_t *)pb;
    return (a > b) - (a < b);
}

static void okfft_init_offsets(okfft_plan_t *plan, size_t N)
{
    const size_t offset_count = N / leaf_N;
    ptrdiff_t *offsets = (ptrdiff_t *) OKFFT_ALLOC_DATA(offset_count * sizeof(ptrdiff_t));
    ptrdiff_t *tmp = (ptrdiff_t *) OKFFT_ALLOC_TEMP_DATA(2 * offset_count * sizeof(ptrdiff_t));

    okfft_elab_even(tmp, (ptrdiff_t) N);

    for (size_t i = 0; i < 2 * offset_count; i += 2)
    {
        if (tmp[i] < 0)
            tmp[i] += N;
    }

    qsort(tmp, offset_count, 2 * sizeof(*tmp), okfft_offset_cmp);

    for (size_t i = 0; i < offset_count; i++)
        offsets[i] = 2 * tmp[2 * i + 1];

    OKFFT_FREE_TEMP_DATA(tmp);
    plan->offsets = offsets;
}

static void okfft_init_indices(okfft_plan_t *p, size_t N)
{
    const size_t N2 = N >> 1;
    const size_t N4 = N >> 2;

    p->is[0] = 0;
    p->is[1] = N;
    p->is[2] = N2;
    p->is[3] = N2 * 3;
    p->is[4] = N4;
    p->is[5] = N4 * 5;
    p->is[6] = N4 * 7;
    p->is[7] = N4 * 3;
}

static inline size_t okfft_ilog2(size_t N)
{
#ifdef _MSC_VER
    unsigned long l2;
    _BitScanForward64(&l2, N);
    return l2;
#else
    return __builtin_ctzll(N);
#endif
}

// START TWIDDLES
// twiddle generation from Jukka Ojanen -- https://github.com/linkotec/ffts

static const OKFFT_ALIGN(32) uint64_t half_secant[66] =
{
    0x3fe0000000000000, 0x3be3bd3cc9be45de, 0x3fe0000000000000, 0x3c03bd3cc9be45de,
    0x3fe0000000000000, 0x3c23bd3cc9be45de, 0x3fe0000000000000, 0x3c43bd3cc9be45de,
    0x3fe0000000000000, 0x3c63bd3cc9be45de, 0x3fe0000000000000, 0x3c83bd3cc9be45df,
    0x3fe0000000000001, 0x3c7de9e64df22efd, 0x3fe0000000000005, 0xbc60b0cd906e8725,
    0x3fe0000000000014, 0xbc80b0cd906e8357, 0x3fe000000000004f, 0xbc5619b20dce83c9,
    0x3fe000000000013c, 0xbc7619b20dc6e79a, 0x3fe00000000004ef, 0x3c83cc9be4af1240,
    0x3fe00000000013bd, 0x3c7e64df2d14c08a, 0x3fe0000000004ef5, 0xbc59b20b47a85465,
    0x3fe0000000013bd4, 0xbc79b203ab79c897, 0x3fe000000004ef4f, 0x3c79386b15019a96,
    0x3fe000000013bd3d, 0xbc7b16b77d6dbf4b, 0x3fe00000004ef4f3, 0x3c741ee4f30832e0,
    0x3fe00000013bd3cd, 0xbc83f41ed3bcd4bb, 0x3fe0000004ef4f34, 0xbc82ef06dd75aebb,
    0x3fe0000013bd3cde, 0x3c52d979b2b41b3d, 0x3fe000004ef4f46c, 0xbc851db34f0fb458,
    0x3fe000013bd3e0e7, 0x3c58dbab8a0ce3f0, 0x3fe00004ef507722, 0x3c83e3512a8ec295,
    0x3fe00013bd5114f9, 0x3c8b3ca4c4c0d92d, 0x3fe0004ef637de7d, 0x3c45974eb74de729,
    0x3fe0013be8190891, 0xbc814c2026edf4da, 0x3fe004f09436640e, 0x3c8091abe2b34b50,
    0x3fe013d19c61d971, 0x3c7f7df76ce01b8e, 0x3fe0503ed17cba53, 0xbc69760974ad7633,
    0x3fe1517a7bdb3895, 0xbc8008d182f9091b, 0x0000000000000000, 0x0000000000000000,
    0x0000000000000000, 0x0000000000000000,
};

static const OKFFT_ALIGN(32) uint64_t cos_sin_table[132] =
{
    0x3ff0000000000000, 0x3df921fb54442d18, 0xbbf3bd3cc9be45de, 0x3a91a390bb77974f,
    0x3ff0000000000000, 0x3e0921fb54442d18, 0xbc13bd3cc9be45de, 0x3aa19bd054a14928,
    0x3ff0000000000000, 0x3e1921fb54442d18, 0xbc33bd3cc9be45de, 0x3ab17cceb948108a,
    0x3ff0000000000000, 0x3e2921fb54442d18, 0xbc53bd3cc9be45de, 0x3ac100c84be32e14,
    0x3ff0000000000000, 0x3e3921fb54442d18, 0xbc73bd3cc9be45de, 0x3ace215d2c9f4879,
    0x3fefffffffffffff, 0x3e4921fb54442d18, 0x3c8885866c837443, 0x3acd411f0005f376,
    0x3feffffffffffffe, 0x3e5921fb54442d18, 0xbc8de9e64df22ef1, 0xbaf7b1539937209e,
    0x3feffffffffffff6, 0x3e6921fb54442d16, 0x3c70b0cd906e88aa, 0xbb03b7c0fe19968a,
    0x3fefffffffffffd9, 0x3e7921fb54442d0e, 0xbc8e9e64df22ed26, 0xbaee8bb48d1b6ffb,
    0x3fefffffffffff62, 0x3e8921fb54442cef, 0x3c6619b20dd18f0f, 0xbb00e1337f2b20fb,
    0x3feffffffffffd88, 0x3e9921fb54442c73, 0x3c8619b20dd314b2, 0xbb174e98619fdf6e,
    0x3feffffffffff621, 0x3ea921fb54442a83, 0x3c8866c83764acf5, 0xbb388215f5b2407f,
    0x3fefffffffffd886, 0x3eb921fb544422c2, 0xbc8e64df20e7a944, 0x3b5a09617b9b9f23,
    0x3fefffffffff6216, 0x3ec921fb544403c1, 0x3c69b20e52ee25ea, 0xbb5999d94df6a86a,
    0x3feffffffffd8858, 0x3ed921fb544387ba, 0x3c89b20fd8910ead, 0x3b77d9db0809d04d,
    0x3feffffffff62162, 0x3ee921fb544197a1, 0xbc8937a8438d3925, 0xbb858b02a5d27f7a,
    0x3fefffffffd88586, 0x3ef921fb5439d73a, 0x3c8b22e494b3ddd2, 0xbb863c7ff8a3b73d,
    0x3fefffffff62161a, 0x3f0921fb541ad59e, 0xbc835c137ea469b2, 0x3bae9860b8cee262,
    0x3feffffffd885867, 0x3f1921fb539ecf31, 0xbc77d55623a32e63, 0x3b96b111fcd23a30,
    0x3feffffff621619c, 0x3f2921fb51aeb57c, 0xbc87507dbbbd8fe6, 0xbbca6e1d4916c435,
    0x3fefffffd8858675, 0x3f3921fb49ee4ea6, 0xbc879f0e54748eab, 0x3bde894d744a453e,
    0x3fefffff62161a34, 0x3f4921fb2aecb360, 0xbc6136dcb1f9b9c4, 0x3be876157e566b4c,
    0x3feffffd88586ee6, 0x3f5921faaee6472e, 0x3c81af64f173ae5b, 0xbbfee52e284a9df8,
    0x3feffff621621d02, 0x3f6921f8becca4ba, 0xbc76acfcebc82813, 0x3c02ba407bcab5b2,
    0x3fefffd8858e8a92, 0x3f7921f0fe670071, 0x3c8359c71883bcf7, 0x3bfab967fe6b7a9b,
    0x3fefff62169b92db, 0x3f8921d1fcdec784, 0x3c85dda3c81fbd0d, 0x3c29878ebe836d9d,
    0x3feffd886084cd0d, 0x3f992155f7a3667e, 0xbc81354d4556e4cb, 0xbbfb1d63091a0130,
    0x3feff621e3796d7e, 0x3fa91f65f10dd814, 0xbc6c57bc2e24aa15, 0xbc2912bd0d569a90,
    0x3fefd88da3d12526, 0x3fb917a6bc29b42c, 0xbc887df6378811c7, 0xbc3e2718d26ed688,
    0x3fef6297cff75cb0, 0x3fc8f8b83c69a60b, 0x3c7562172a361fd3, 0xbc626d19b9ff8d82,
    0x3fed906bcf328d46, 0x3fd87de2a6aea963, 0x3c7457e610231ac2, 0xbc672cedd3d5a610,
    0x3fe6a09e667f3bcd, 0x3fe6a09e667f3bcd, 0xbc8bdd3413b26456, 0xbc8bdd3413b26456,
    0x0000000000000000, 0x3ff0000000000000, 0x0000000000000000, 0x0000000000000000,
};

typedef float cplx[2];
typedef ptrdiff_t ssize_t;

static void okfft_generate_twiddle_table(cplx *table, size_t table_size)
{
    table[0][0] =  1.0f;
    table[0][1] = -0.f;

    if (table_size == 2)
    {
        table[1][0] =  0.70710677f;
        table[1][1] = -0.70710677f;
        return;
    }
    
    size_t log2   = okfft_ilog2(table_size);
    size_t offset = 32 - log2;

    const __m128d *__restrict ct = (const __m128d *) &cos_sin_table[4 * offset];
    const double *__restrict hs  = (const double *) &half_secant[2 * offset];

    OKFFT_ALIGN(16) __m128d w[32];
    OKFFT_ALIGN(16) __m128d h[32];

    // init from lut
    for (size_t i = 0; i <= log2; i++)
    {
        w[i] = ct[2 * i];
        h[i] = _mm_set1_pd(hs[2 * i]); // duplicate the high part
    }

    static const __m128d sign_swap = { 0.0, -0.0 };

    // generate sin / cos tables with max .5 ulp error
    for (int i = 1; i < (int) table_size / 2; i++)
    {
        log2 = okfft_ilog2(i); // trailing zeros in index        
        __m128d wvl = w[log2];

        __m128d v1 = _mm_shuffle_pd(wvl, wvl, 1);
        __m128d v0 = _mm_or_pd(wvl, sign_swap);
                v1 = _mm_or_pd(v1,  sign_swap);

        _mm_storel_pi((__m64 *) &table[i + 0],          _mm_cvtpd_ps(v0));
        _mm_storel_pi((__m64 *) &table[table_size - i], _mm_cvtpd_ps(v1));

        // skip and find next trailing zero
        offset = log2 + 2 + okfft_ilog2(~i >> (log2 + 2));
        w[log2] = _mm_mul_pd(h[log2], _mm_add_pd(w[log2 + 1], w[offset]));
    }

    table[table_size / 2][0] =  0.70710677f;
    table[table_size / 2][1] = -0.70710677f;
}

static void okfft_init_twiddles(okfft_plan_t *plan, size_t N, bool is_inverse)
{
#define dup_re(x) _mm_shuffle_ps(x, x, _MM_SHUFFLE(2, 2, 0, 0))
#define dup_im(x) _mm_shuffle_ps(x, x, _MM_SHUFFLE(3, 3, 1, 1))

    __m128 muli_sign;
    if (is_inverse)
        muli_sign = _mm_set_ps(0.0f, -0.0f, 0.0f, -0.0f);
    else
        muli_sign = _mm_set_ps(-0.0f, 0.0f, -0.0f, 0.0f);

    size_t lut_count = okfft_ilog2(N / leaf_N);
    size_t lut_size  = 16;
    for (size_t i = 1; i < lut_count; i++)
        lut_size += 48 * (1 << (i - 1));

    cplx *twiddles = (cplx *) OKFFT_ALLOC_ALIGNED_DATA(lut_size * sizeof(float));
    ptrdiff_t *twiddle_indices = (ptrdiff_t *) OKFFT_ALLOC_ALIGNED_DATA(lut_count * sizeof(ptrdiff_t));
    
    cplx *w = twiddles;

    // calculate factors
    size_t n = (size_t) leaf_N * 2;
    size_t m = (size_t) leaf_N << (lut_count - 2);

    cplx *tmp = (cplx *) OKFFT_ALLOC_TEMP_ALIGNED_DATA(m * sizeof(cplx));
    okfft_generate_twiddle_table(tmp, m);
    
    size_t stride = 1ull << (lut_count - 1);

    #ifdef OKFFT_HAS_AVX
        const bool needs_reorder = okfft_cpu_has_avx();
    #endif

    {
        OKFFT_ALIGN(16) cplx w0[4];

        for (size_t j = 0; j < 4; j++)
        {
            w0[j][0] = tmp[j * stride][0];
            w0[j][1] = tmp[j * stride][1];
        }

        __m128 t0 = _mm_load_ps((float *) w0 + 0);
        __m128 t1 = _mm_load_ps((float *) w0 + 4);

        __m128 re0 = dup_re(t0);
        __m128 re1 = dup_re(t1);

        __m128 im0 = dup_im(t0);
        __m128 im1 = dup_im(t1);

        im0 = _mm_xor_ps(im0, muli_sign);
        im1 = _mm_xor_ps(im1, muli_sign);

        #ifdef OKFFT_HAS_AVX
        if (needs_reorder)
        {
            // AVX x4 reorder
            _mm_store_ps((float *) w +  0, re0);
            _mm_store_ps((float *) w +  4, re1);
            _mm_store_ps((float *) w +  8, im0);
            _mm_store_ps((float *) w + 12, im1);
        }
        else
        #endif
        {
            _mm_store_ps((float *) w +  0, re0);
            _mm_store_ps((float *) w +  4, im0);
            _mm_store_ps((float *) w +  8, re1);
            _mm_store_ps((float *) w + 12, im1);
        }

        w += 8;
        n *= 2;
        stride >>= 1;
    }

    for (size_t i = 1; i < lut_count; i++) 
    {
        twiddle_indices[i] = w - twiddles;
        
        cplx *w0 = (cplx *) OKFFT_ALLOC_TEMP_ALIGNED_DATA(n / 8 * sizeof(cplx));
        cplx *w1 = (cplx *) OKFFT_ALLOC_TEMP_ALIGNED_DATA(n / 8 * sizeof(cplx));
        cplx *w2 = (cplx *) OKFFT_ALLOC_TEMP_ALIGNED_DATA(n / 8 * sizeof(cplx));
            
        for (size_t j = 0; j < n / 8; j++)
        {
            w0[j][0] = tmp[2 * j * stride][0];
            w0[j][1] = tmp[2 * j * stride][1];

            w1[j][0] = tmp[j * stride][0];
            w1[j][1] = tmp[j * stride][1];

            w2[j][0] = tmp[(j + (n / 8)) * stride][0];
            w2[j][1] = tmp[(j + (n / 8)) * stride][1];
        }

        #ifdef OKFFT_HAS_AVX
        if (needs_reorder)
        {
            // AVX x8 reorder
            for (size_t j = 0; j < n / 8; j += 4)
            {
                __m128 t00 = _mm_load_ps((float *)w0 + 2 * j);
                __m128 t10 = _mm_load_ps((float *)w1 + 2 * j);
                __m128 t20 = _mm_load_ps((float *)w2 + 2 * j);

                __m128 t01 = _mm_load_ps((float *)w0 + 2 * j + 4);
                __m128 t11 = _mm_load_ps((float *)w1 + 2 * j + 4);
                __m128 t21 = _mm_load_ps((float *)w2 + 2 * j + 4);

                __m128 re00 = dup_re(t00);
                __m128 re10 = dup_re(t10);
                __m128 re20 = dup_re(t20);

                __m128 re01 = dup_re(t01);
                __m128 re11 = dup_re(t11);
                __m128 re21 = dup_re(t21);

                __m128 im00 = dup_im(t00);
                __m128 im10 = dup_im(t10);
                __m128 im20 = dup_im(t20);

                __m128 im01 = dup_im(t01);
                __m128 im11 = dup_im(t11);
                __m128 im21 = dup_im(t21);

                im00 = _mm_xor_ps(im00, muli_sign);
                im10 = _mm_xor_ps(im10, muli_sign);
                im20 = _mm_xor_ps(im20, muli_sign);

                im01 = _mm_xor_ps(im01, muli_sign);
                im11 = _mm_xor_ps(im11, muli_sign);
                im21 = _mm_xor_ps(im21, muli_sign);

                _mm_store_ps((float *)w + 12 * j +  0, re00);
                _mm_store_ps((float *)w + 12 * j +  4, re01);

                _mm_store_ps((float *)w + 12 * j +  8, im00);
                _mm_store_ps((float *)w + 12 * j + 12, im01);

                _mm_store_ps((float *)w + 12 * j + 16, re10);
                _mm_store_ps((float *)w + 12 * j + 20, re11);

                _mm_store_ps((float *)w + 12 * j + 24, im10);
                _mm_store_ps((float *)w + 12 * j + 28, im11);

                _mm_store_ps((float *)w + 12 * j + 32, re20);
                _mm_store_ps((float *)w + 12 * j + 36, re21);

                _mm_store_ps((float *)w + 12 * j + 40, im20);
                _mm_store_ps((float *)w + 12 * j + 44, im21);
            }
        }
        else
        #endif
        {
            for (size_t j = 0; j < n / 8; j += 2) 
            {
                __m128 t0 = _mm_load_ps((float *) w0 + j * 2);
                __m128 t1 = _mm_load_ps((float *) w1 + j * 2);
                __m128 t2 = _mm_load_ps((float *) w2 + j * 2);

                __m128 re0 = dup_re(t0);
                __m128 re1 = dup_re(t1);
                __m128 re2 = dup_re(t2);

                __m128 im0 = dup_im(t0);
                __m128 im1 = dup_im(t1);
                __m128 im2 = dup_im(t2);

                im0 = _mm_xor_ps(im0, muli_sign);
                im1 = _mm_xor_ps(im1, muli_sign);
                im2 = _mm_xor_ps(im2, muli_sign);

                _mm_store_ps((float *) w + j * 12 +  0, re0);
                _mm_store_ps((float *) w + j * 12 +  4, im0);

                _mm_store_ps((float *) w + j * 12 +  8, re1);
                _mm_store_ps((float *) w + j * 12 + 12, im1);

                _mm_store_ps((float *) w + j * 12 + 16, re2);
                _mm_store_ps((float *) w + j * 12 + 20, im2);
            }
        }

        w += n / 8 * 3 * 2;

        OKFFT_FREE_TEMP_ALIGNED_DATA(w0);
        OKFFT_FREE_TEMP_ALIGNED_DATA(w1);
        OKFFT_FREE_TEMP_ALIGNED_DATA(w2);

        n *= 2;
        stride >>= 1;
    }
    
    OKFFT_FREE_TEMP_ALIGNED_DATA(tmp);
    
    plan->ws    = (float *) twiddles;
    plan->ws_is = twiddle_indices;

#undef dup_re
#undef dup_im
}

static void okfft_init_real_coeffs(okfft_plan_t *plan, size_t N, bool is_inverse)
{
    typedef double dbl_cplx[2];
    float * __restrict A = (float * __restrict) OKFFT_ALLOC_ALIGNED_DATA(N * sizeof(float));
    float * __restrict B = (float * __restrict) OKFFT_ALLOC_ALIGNED_DATA(N * sizeof(float));

    size_t log2   = okfft_ilog2(N);
    size_t offset = 34 - log2;

    const dbl_cplx * __restrict ct = (const dbl_cplx *) &cos_sin_table[4 * offset];
    const double *__restrict hs = (const double *) &half_secant[2 * offset];

    OKFFT_ALIGN(16) dbl_cplx w[32];

    // init from lut
    for (size_t i = 0; i <= log2; i++)
    {
        w[i][0] = ct[2 * i][0];
        w[i][1] = ct[2 * i][1];
    }

    if (is_inverse)
    {
        A[0] =  1.0f;
        A[1] = -1.0f;
        B[0] =  1.0f;
        B[1] =  1.0f;

        for (int i = 1; i < (int) N / 4; i++)
        {
            log2 = okfft_ilog2(i);

            float t1 = (float) w[log2][0];
            float t0 = (float) (1.0 - w[log2][1]);
            float t2 = (float) (1.0 + w[log2][1]);

            A[    2 * i + 0] =  t0;
            A[N - 2 * i + 0] =  t0;
            A[    2 * i + 1] = -t1;
            A[N - 2 * i + 1] =  t1;

            B[    2 * i + 0] =  t2;
            B[N - 2 * i + 0] =  t2;
            B[    2 * i + 1] =  t1;
            B[N - 2 * i + 1] = -t1;

            // skip and find next trailing zero
            offset = log2 + 2 + okfft_ilog2(~i >> (log2 + 2));
            w[log2][0] = hs[2 * log2] * (w[log2 + 1][0] + w[offset][0]);
            w[log2][1] = hs[2 * log2] * (w[log2 + 1][1] + w[offset][1]);
        }

        A[2 * N / 4 + 0] = 0.0f;
        A[2 * N / 4 + 1] = 0.0f;
        B[2 * N / 4 + 0] = 2.0f;
        B[2 * N / 4 + 1] = 0.0f;
    }
    else
    {
        A[0] =  0.5f;
        A[1] = -0.5f;
        B[0] =  0.5f;
        B[1] =  0.5f;

        for (ssize_t i = 1; i < (ssize_t) N / 4; i++)
        {
            log2 = okfft_ilog2(i);

            float t1 = (float) (0.5 * w[log2][0]);
            float t0 = (float) (0.5 * (1.0 - w[log2][1]));
            float t2 = (float) (0.5 * (1.0 + w[log2][1]));

            A[    2 * i + 0] =  t0;
            A[N - 2 * i + 0] =  t0;
            A[    2 * i + 1] = -t1;
            A[N - 2 * i + 1] =  t1;

            B[    2 * i + 0] =  t2;
            B[N - 2 * i + 0] =  t2;
            B[    2 * i + 1] =  t1;
            B[N - 2 * i + 1] = -t1;

            // skip and find next trailing zero
            offset = log2 + 2 + okfft_ilog2(~i >> (log2 + 2));
            w[log2][0] = hs[2 * log2] * (w[log2 + 1][0] + w[offset][0]);
            w[log2][1] = hs[2 * log2] * (w[log2 + 1][1] + w[offset][1]);
        }

        A[2 * N / 4 + 0] = 0.0f;
        A[2 * N / 4 + 1] = 0.0f;
        B[2 * N / 4 + 0] = 1.0f;
        B[2 * N / 4 + 1] = 0.0f;
    }

    // reorder A and B to avoid shuffling in the kernel! (avoids 4 cycles?)
    #ifdef OKFFT_HAS_AVX
    if (okfft_cpu_has_avx())
    {
        for (size_t i = 0; i < N; i += 16)
        {
            __m128 a0 = _mm_load_ps(A + i +  0);
            __m128 a1 = _mm_load_ps(A + i +  4);
            __m128 a2 = _mm_load_ps(A + i +  8);
            __m128 a3 = _mm_load_ps(A + i + 12);
            __m128 b0 = _mm_load_ps(B + i +  0);
            __m128 b1 = _mm_load_ps(B + i +  4);
            __m128 b2 = _mm_load_ps(B + i +  8);
            __m128 b3 = _mm_load_ps(B + i + 12);

            __m128 ar0 = _mm_shuffle_ps(a0, a2, _MM_SHUFFLE(2, 0, 2, 0));
            __m128 ar1 = _mm_shuffle_ps(a1, a3, _MM_SHUFFLE(2, 0, 2, 0));
            __m128 ai0 = _mm_shuffle_ps(a0, a2, _MM_SHUFFLE(3, 1, 3, 1));
            __m128 ai1 = _mm_shuffle_ps(a1, a3, _MM_SHUFFLE(3, 1, 3, 1));

            __m128 br0 = _mm_shuffle_ps(b0, b2, _MM_SHUFFLE(2, 0, 2, 0));
            __m128 br1 = _mm_shuffle_ps(b1, b3, _MM_SHUFFLE(2, 0, 2, 0));
            __m128 bi0 = _mm_shuffle_ps(b0, b2, _MM_SHUFFLE(3, 1, 3, 1));
            __m128 bi1 = _mm_shuffle_ps(b1, b3, _MM_SHUFFLE(3, 1, 3, 1));
            
            _mm_store_ps(A + i +  0, ar0);
            _mm_store_ps(A + i +  4, ar1);
            _mm_store_ps(A + i +  8, ai0);
            _mm_store_ps(A + i + 12, ai1);

            _mm_store_ps(B + i +  0, br0);
            _mm_store_ps(B + i +  4, br1);
            _mm_store_ps(B + i +  8, bi0);
            _mm_store_ps(B + i + 12, bi1);
        }
    }
    else
    #endif
    {
        for (size_t i = 0; i < N; i += 8)
        {
            __m128 a0 = _mm_load_ps(A + i + 0);
            __m128 a1 = _mm_load_ps(A + i + 4);
            __m128 b0 = _mm_load_ps(B + i + 0);
            __m128 b1 = _mm_load_ps(B + i + 4);

            __m128 are = _mm_shuffle_ps(a0, a1, _MM_SHUFFLE(2, 0, 2, 0));
            __m128 aim = _mm_shuffle_ps(a0, a1, _MM_SHUFFLE(3, 1, 3, 1));
            __m128 bre = _mm_shuffle_ps(b0, b1, _MM_SHUFFLE(2, 0, 2, 0));
            __m128 bim = _mm_shuffle_ps(b0, b1, _MM_SHUFFLE(3, 1, 3, 1));
            
            _mm_store_ps(A + i + 0, are);
            _mm_store_ps(A + i + 4, aim);
            _mm_store_ps(B + i + 0, bre);
            _mm_store_ps(B + i + 4, bim);
        }
    }

    plan->A = A;
    plan->B = B;
}
