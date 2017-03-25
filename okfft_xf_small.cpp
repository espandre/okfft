#include "okfft.h"
#include "okfft_macros.h"

#ifdef _MSC_VER
    #include <intrin.h>
    #define OKFFT_ALIGN(x) __declspec(align(x))
#else
    #include <x86intrin.h>
    #define OKFFT_ALIGN(x) __attribute__((aligned(x)))
#endif

#define OKFFT_SQRT_HALF 0.7071067811865475244008443621048490392848359376884740f

static const __m128 okfft_sse_fwd_sign_mask = _mm_set_ps(-0.f, 0.f, -0.f, 0.f);
static const __m128 okfft_sse_inv_sign_mask = _mm_set_ps(0.f, -0.f, 0.f, -0.f);

void okfft_small_1(const okfft_plan_t *, float *__restrict out, const float *__restrict in)
{
    out[0] = in[0];
    out[1] = in[1];
}

void okfft_small_2(const okfft_plan_t *, float *__restrict out, const float *__restrict in)
{
    const __m128 mask = { 0.f, 0.f, -0.f, -0.f };

    __m128 d = _mm_load_ps(in);
    __m128 d0 = _mm_shuffle_ps(d, d, _MM_SHUFFLE(1, 0, 1, 0));
    __m128 d1 = _mm_shuffle_ps(d, d, _MM_SHUFFLE(3, 2, 3, 2));
    d1 = _mm_xor_ps(d1, mask);

    __m128 r = _mm_add_ps(d0, d1);
    _mm_store_ps(out, r);
}

void okfft_small_fwd_4(const okfft_plan_t *, float *__restrict out, const float *__restrict in)
{
    const __m128 mask0 = { 0.f, 0.f, -0.f, -0.f };
    const __m128 mask1 = { 0.f, 0.f,  0.f, -0.f };

    // 2x radix2
    __m128 d0 = _mm_load_ps(in + 0);
    __m128 d1 = _mm_load_ps(in + 4);

    __m128 q0 = _mm_shuffle_ps(d0, d1, _MM_SHUFFLE(1, 0, 1, 0));
    __m128 q1 = _mm_shuffle_ps(d0, d1, _MM_SHUFFLE(3, 2, 3, 2));

    __m128 q2 = _mm_shuffle_ps(d0, d1, _MM_SHUFFLE(1, 0, 1, 0));
    __m128 q3 = _mm_shuffle_ps(d0, d1, _MM_SHUFFLE(3, 2, 3, 2));

    q1 = _mm_xor_ps(q1, mask0);
    q3 = _mm_xor_ps(q3, mask0);

    __m128 t4t5 = _mm_add_ps(q0, q1);
    __m128 t6t7 = _mm_add_ps(q2, q3);

    // radix 4
    t6t7 = _mm_shuffle_ps(t6t7, t6t7, _MM_SHUFFLE(2, 3, 1, 0));
    t6t7 = _mm_xor_ps(t6t7, mask1);

    __m128 r0 = _mm_add_ps(t4t5, t6t7);
    __m128 r1 = _mm_sub_ps(t4t5, t6t7);

    _mm_store_ps(out + 0, r0);
    _mm_store_ps(out + 4, r1);
}

void okfft_small_inv_4(const okfft_plan_t *, float *__restrict out, const float *__restrict in)
{
    const __m128 mask0 = { 0.f, 0.f, -0.f, -0.f };
    const __m128 mask1 = { 0.f, 0.f, -0.f,  0.f };

    // 2x radix2
    __m128 d0 = _mm_load_ps(in + 0);
    __m128 d1 = _mm_load_ps(in + 4);

    __m128 q0 = _mm_shuffle_ps(d0, d1, _MM_SHUFFLE(1, 0, 1, 0));
    __m128 q1 = _mm_shuffle_ps(d0, d1, _MM_SHUFFLE(3, 2, 3, 2));

    __m128 q2 = _mm_shuffle_ps(d0, d1, _MM_SHUFFLE(1, 0, 1, 0));
    __m128 q3 = _mm_shuffle_ps(d0, d1, _MM_SHUFFLE(3, 2, 3, 2));

    q1 = _mm_xor_ps(q1, mask0);
    q3 = _mm_xor_ps(q3, mask0);

    __m128 t4t5 = _mm_add_ps(q0, q1);
    __m128 t6t7 = _mm_add_ps(q2, q3);

    // radix 4
    t6t7 = _mm_shuffle_ps(t6t7, t6t7, _MM_SHUFFLE(2, 3, 1, 0));
    t6t7 = _mm_xor_ps(t6t7, mask1);

    __m128 r0 = _mm_add_ps(t4t5, t6t7);
    __m128 r1 = _mm_sub_ps(t4t5, t6t7);

    _mm_store_ps(out + 0, r0);
    _mm_store_ps(out + 4, r1);
}

#define OKFFT_COS_PI_8 0.9238795325112867561281831893967882868224166258636425f
#define OKFFT_SIN_PI_8 0.3826834323650897717284599840303988667613445624856270f

static const OKFFT_ALIGN(16) float okfft_small_fwd_constants[24] =
{
    1.0f, 1.0f,  OKFFT_SQRT_HALF, OKFFT_SQRT_HALF,
    -0.0f, 0.0f, -OKFFT_SQRT_HALF, OKFFT_SQRT_HALF,

    1.0f, 1.0f,  OKFFT_COS_PI_8, OKFFT_COS_PI_8,
    -0.0f, 0.0f, -OKFFT_SIN_PI_8, OKFFT_SIN_PI_8,

    OKFFT_SQRT_HALF, OKFFT_SQRT_HALF,  OKFFT_SIN_PI_8, OKFFT_SIN_PI_8,
    -OKFFT_SQRT_HALF, OKFFT_SQRT_HALF, -OKFFT_COS_PI_8, OKFFT_COS_PI_8
};

static const OKFFT_ALIGN(16) float okfft_small_inv_constants[24] =
{
    1.0f,  1.0f, OKFFT_SQRT_HALF,  OKFFT_SQRT_HALF,
    0.0f, -0.0f, OKFFT_SQRT_HALF, -OKFFT_SQRT_HALF,

    1.0f,  1.0f, OKFFT_COS_PI_8,  OKFFT_COS_PI_8,
    0.0f, -0.0f, OKFFT_SIN_PI_8, -OKFFT_SIN_PI_8,

    OKFFT_SQRT_HALF,  OKFFT_SQRT_HALF, OKFFT_SIN_PI_8,  OKFFT_SIN_PI_8,
    OKFFT_SQRT_HALF, -OKFFT_SQRT_HALF, OKFFT_COS_PI_8, -OKFFT_COS_PI_8
};

void okfft_small_fwd_8(const okfft_plan_t *, float *__restrict out, const float *__restrict in)
{
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const float *__restrict lut = okfft_small_fwd_constants;

    __m128 r01, r23, r45, r67;
    OKFFT_SSE_L42(in + 0, in + 8, in + 4, in + 12, r01, r23, r45, r67);

    __m128 re = _mm_load_ps(lut);
    __m128 im = _mm_load_ps(lut + 4);
    OKFFT_SSE_KN(re, im, r01, r23, r45, r67);

    okfft_sse_store4(out, r01, r23, r45, r67);
}

void okfft_small_inv_8(const okfft_plan_t *, float *__restrict out, const float *__restrict in)
{
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const float *__restrict lut = okfft_small_inv_constants;

    __m128 r01, r23, r45, r67;
    OKFFT_SSE_L42(in + 0, in + 8, in + 4, in + 12, r01, r23, r45, r67);

    __m128 re = _mm_load_ps(lut);
    __m128 im = _mm_load_ps(lut + 4);
    OKFFT_SSE_KN(re, im, r01, r23, r45, r67);

    okfft_sse_store4(out, r01, r23, r45, r67);
}

void okfft_small_fwd_16(const okfft_plan_t *, float *__restrict out, const float *__restrict in)
{
    const __m128 sse_sign_mask = okfft_sse_fwd_sign_mask;
    const float *__restrict lut = okfft_small_fwd_constants;

    __m128 r01, r23, r45, r67, r89, r1011, r1213, r1415;

    OKFFT_SSE_L44(in + 0, in + 16, in + 8, in + 24, r01, r23, r89, r1011);
    OKFFT_SSE_L24(in + 4, in + 20, in + 28, in + 12, r45, r67, r1415, r1213);

    __m128 re0 = _mm_load_ps(lut + 0);
    __m128 im0 = _mm_load_ps(lut + 4);
    __m128 re1 = _mm_load_ps(lut + 8);
    __m128 im1 = _mm_load_ps(lut + 12);
    __m128 re2 = _mm_load_ps(lut + 16);
    __m128 im2 = _mm_load_ps(lut + 20);

    OKFFT_SSE_KN(re0, im0, r01, r23, r45, r67);
    OKFFT_SSE_KNKN(re1, im1, re2, im2, r01, r45, r89, r1213, r23, r67, r1011, r1415);

    okfft_sse_store4(out + 0, r01, r23, r45, r67);
    okfft_sse_store4(out + 16, r89, r1011, r1213, r1415);
}

void okfft_small_inv_16(const okfft_plan_t *, float *__restrict out, const float *__restrict in)
{
    const __m128 sse_sign_mask = okfft_sse_inv_sign_mask;
    const float *__restrict lut = okfft_small_inv_constants;

    __m128 r01, r23, r45, r67, r89, r1011, r1213, r1415;

    OKFFT_SSE_L44(in + 0, in + 16, in + 8, in + 24, r01, r23, r89, r1011);
    OKFFT_SSE_L24(in + 4, in + 20, in + 28, in + 12, r45, r67, r1415, r1213);

    __m128 re0 = _mm_load_ps(lut + 0);
    __m128 im0 = _mm_load_ps(lut + 4);
    __m128 re1 = _mm_load_ps(lut + 8);
    __m128 im1 = _mm_load_ps(lut + 12);
    __m128 re2 = _mm_load_ps(lut + 16);
    __m128 im2 = _mm_load_ps(lut + 20);

    OKFFT_SSE_KN(re0, im0, r01, r23, r45, r67);
    OKFFT_SSE_KNKN(re1, im1, re2, im2, r01, r45, r89, r1213, r23, r67, r1011, r1415);

    okfft_sse_store4(out + 0, r01, r23, r45, r67);
    okfft_sse_store4(out + 16, r89, r1011, r1213, r1415);
}
