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
#include <stdint.h>
#include <stddef.h>

// comment out defines to configure
//#define OKFFT_HAS_AVX 1
#define OKFFT_HAS_SSE 1

#if !defined(OKFFT_HAS_AVX) && !defined(OKFFT_HAS_SSE)
    #error "Must enable avx xforms, sse xforms, or both!"
#endif

// override for custom memory allocation
#ifndef OKFFT_CUSTOM_ALLOC
    
    #define OKFFT_ALLOC_PLAN(size)      malloc(size)
    #define OKFFT_ALLOC_DATA(size)      malloc(size)

    // temp data is only used during plan construction
    #define OKFFT_ALLOC_TEMP_DATA(size) malloc(size)
    
    #define OKFFT_FREE_PLAN(ptr)        free(ptr)
    #define OKFFT_FREE_DATA(ptr)        free(ptr)
    #define OKFFT_FREE_TEMP_DATA(ptr)   free(ptr)

    #ifdef OKFFT_HAS_AVX
        #define OKFFT_ALLOC_ALIGNED_DATA(size) _mm_malloc(size, 32)
    #else
        #define OKFFT_ALLOC_ALIGNED_DATA(size) _mm_malloc(size, 16)
    #endif

    #define OKFFT_ALLOC_BUFFER(size)            OKFFT_ALLOC_ALIGNED_DATA(size)
    
    // temp data is only used during plan construction
    #define OKFFT_ALLOC_TEMP_ALIGNED_DATA(size) OKFFT_ALLOC_ALIGNED_DATA(size)

    #define OKFFT_FREE_ALIGNED_DATA(ptr)        _mm_free(ptr)
    #define OKFFT_FREE_TEMP_ALIGNED_DATA(ptr)   OKFFT_FREE_ALIGNED_DATA(ptr)
    #define OKFFT_FREE_BUFFER(ptr)              OKFFT_FREE_ALIGNED_DATA(ptr)
#endif

#ifndef OKFFT_LOG
    #define OKFFT_LOG(mesg, ...) printf(mesg, ##__VA_ARGS__)
#endif

struct okfft_plan_t;
typedef void (*okfft_xform_func_t)(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);

struct okfft_plan_t
{
    float *__restrict ws;               // twiddles
    ptrdiff_t *__restrict offsets;      // output indices
    
    ptrdiff_t is[8];                    // input indices
    ptrdiff_t *__restrict ws_is;        // twiddle factor indices

    size_t N;                           // transform size (used by the generic xform)
    size_t i0, i1;                      // base case loop sizes (used by the generic xform)
    
    okfft_xform_func_t xform;           // ptr to xform function

    float *__restrict A;                // coeffs for real valued xforms
    float *__restrict B;

    size_t flags;
};

struct okfft_buffer_t { float *buffer; };

// 'okfft_buffer_t' is needed to keep the real transforms thread safe!
okfft_buffer_t okfft_create_buffer(size_t N);
void okfft_destroy_buffer(okfft_buffer_t *buffer);

enum OKFFT_DIRECTION
{
    OKFFT_DIR_FORWARD,
    OKFFT_DIR_INVERSE
};

// complex -> complex
okfft_plan_t *okfft_create_plan(size_t N, OKFFT_DIRECTION dir);

// real -> complex
okfft_plan_t *okfft_create_plan_real(size_t N, OKFFT_DIRECTION dir);

void okfft_destroy_plan(okfft_plan_t *plan);

// for complex -> complex transforms
// thread safe for plan
void okfft_execute(const okfft_plan_t *plan, float *__restrict output, const float *__restrict input);

// for real -> complex / complex -> real transforms
// thread safe for plan (not state buffer!)
// NOTE: due to how this optimisation works, the complex -> real xform reads *N + 2* elements from 'input'. Easiest way to ensure the required capacity is to use a 'okfft_buffer_t'
void okfft_execute_real(const okfft_plan_t *plan, okfft_buffer_t *state, float *__restrict output, const float *__restrict input);
