# OKFFT

This project is a heavily reworked and simplified fork of a fork of FFTS. 

Fork: [https://github.com/linkotec/ffts](https://github.com/linkotec/ffts)

Original: [https://github.com/anthonix/ffts](https://github.com/anthonix/ffts)

It currently only does 1D transforms of power-of-twos.

However, it gained better vectorisation support in the form of AVX, and with that, increased performance (~40% increase).

## Performance

The performance measures uses the FFTW model of `mflops = 5 * N * log2(N) / usec_duration`.


The following figures were all recorded on an Intel i7 3930K running at 3.8 GHz:

AVX:

| Size  | GFLOPS |     Duration |
|------:|-------:|-------------:|
|    32 |  24.97 |     32.04 ns |
|    64 |  26.58 |     72.23 ns |
|   128 |  31.96 |    140.19 ns |
|   256 |  31.81 |    321.94 ns |
|   512 |  34.39 |    669.98 ns |
|  1024 |  33.13 |   1545.60 ns |
|  2048 |  28.62 |   3935.15 ns |
|  4096 |  26.39 |   9311.06 ns |
|  8192 |  24.19 |  22010.12 ns |
| 16384 |  21.83 |  52535.01 ns |
| 32768 |  21.01 | 116994.31 ns |
| 65536 |  19.32 | 271416.68 ns |

SSE:

| Size  | GFLOPS |     Duration |
|------:|-------:|-------------:|
|    32 |  22.08 |     36.23 ns |
|    64 |  24.19 |     79.38 ns |
|   128 |  23.98 |    186.85 ns |
|   256 |  24.25 |    422.26 ns |
|   512 |  24.81 |    928.56 ns |
|  1024 |  24.64 |   2077.96 ns |
|  2048 |  20.69 |   5444.88 ns |
|  4096 |  21.24 |  11572.62 ns |
|  8192 |  18.66 |  28534.90 ns |
| 16384 |  16.60 |  69094.78 ns |
| 32768 |  16.46 | 149308.98 ns |
| 65536 |  16.10 | 325683.48 ns |

## Building
Include the source files in your project.

## Configuration
The `okfft.h` header has a number of macros that allow for customisation and / or tuning.

### Vectorisation
By default, the FFT kernels only use SSE instructions for vectorisation, but by defining the macro `OKFFT_HAS_AVX`, AVX instructions may also be used.

Similarly, you can comment out the `OKFFT_HAS_SSE` define to remove the SSE kernels, which saves on library size.

__Note:__ The project contains cpu dispatch code to avoid using avx instructions if they are not supported, however, this requires compiling only the file `okfft_xf_avx.cpp` with AVX enabled.

__Note:__ At least one of `OKFFT_HAS_SSE` and `OKFFT_HAS_AVX` *must* be defined.


### Memory Allocation

Custom allocators can be used by changing the relevant macros in the 'okfft.h' header
