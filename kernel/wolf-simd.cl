/* $Id: simd.c 227 2010-06-16 17:28:38Z tp $ */
/*
 * SIMD implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2007-2010  Projet RNRT SAPHIR
 * 
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 * 
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ===========================(LICENSE END)=============================
 *
 * @author   Thomas Pornin <thomas.pornin@cryptolog.com>
 */

#ifndef WOLF_SIMD_CL
#define WOLF_SIMD_CL

#define COMPILER_UNROLLING

typedef uint u32;
typedef int s32;

//__constant static const s32 SIMD_Q[] = {
//  4, 28, -80, -120, -47, -126, 45, -123, -92, -127, -70, 23, -23, -24, 40, -125, 101, 122, 34, -24, -119, 110, -121, -112, 32, 24, 51, 73, -117, -64, -21, 42, -60, 16, 5, 85, 107, 52, -44, -96, 42, 127, -18, -108, -47, 26, 91, 117, 112, 46, 87, 79, 126, -120, 65, -24, 121, 29, 118, -7, -53, 85, -98, -117, 32, 115, -47, -116, 63, 16, -108, 49, -119, 57, -110, 4, -76, -76, -42, -86, 58, 115, 4, 4, -83, -51, -37, 116, 32, 15, 36, -42, 73, -99, 94, 87, 60, -20, 67, 12, -76, 55, 117, -68, -82, -80, 93, -20, 92, -21, -128, -91, -11, 84, -28, 76, 94, -124, 37, 93, 17, -78, -106, -29, 88, -15, -47, 102, -4, -28, 80, 120, 47, 126, -45, 123, 92, 127, 70, -23, 23, 24, -40, 125, -101, -122, -34, 24, 119, -110, 121, 112, -32, -24, -51, -73, 117, 64, 21, -42, 60, -16, -5, -85, -107, -52, 44, 96, -42, -127, 18, 108, 47, -26, -91, -117, -112, -46, -87, -79, -126, 120, -65, 24, -121, -29, -118, 7, 53, -85, 98, 117, -32, -115, 47, 116, -63, -16, 108, -49, 119, -57, 110, -4, 76, 76, 42, 86, -58, -115, -4, -4, 83, 51, 37, -116, -32, -15, -36, 42, -73, 99, -94, -87, -60, 20, -67, -12, 76, -55, -117, 68, 82, 80, -93, 20, -92, 21, 128, 91, 11, -84, 28, -76, -94, 124, -37, -93, -17, 78, 106, 29, -88, 15, 47, -102
//};

/*
 * The powers of 41 modulo 257. We use exponents from 0 to 255, inclusive.
 */
__constant static const s32 alpha_tab[] = {
    1,  41, 139,  45,  46,  87, 226,  14,  60, 147, 116, 130,
  190,  80, 196,  69,   2,  82,  21,  90,  92, 174, 195,  28,
  120,  37, 232,   3, 123, 160, 135, 138,   4, 164,  42, 180,
  184,  91, 133,  56, 240,  74, 207,   6, 246,  63,  13,  19,
    8,  71,  84, 103, 111, 182,   9, 112, 223, 148, 157,  12,
  235, 126,  26,  38,  16, 142, 168, 206, 222, 107,  18, 224,
  189,  39,  57,  24, 213, 252,  52,  76,  32,  27,  79, 155,
  187, 214,  36, 191, 121,  78, 114,  48, 169, 247, 104, 152,
   64,  54, 158,  53, 117, 171,  72, 125, 242, 156, 228,  96,
   81, 237, 208,  47, 128, 108,  59, 106, 234,  85, 144, 250,
  227,  55, 199, 192, 162, 217, 159,  94, 256, 216, 118, 212,
  211, 170,  31, 243, 197, 110, 141, 127,  67, 177,  61, 188,
  255, 175, 236, 167, 165,  83,  62, 229, 137, 220,  25, 254,
  134,  97, 122, 119, 253,  93, 215,  77,  73, 166, 124, 201,
   17, 183,  50, 251,  11, 194, 244, 238, 249, 186, 173, 154,
  146,  75, 248, 145,  34, 109, 100, 245,  22, 131, 231, 219,
  241, 115,  89,  51,  35, 150, 239,  33,  68, 218, 200, 233,
   44,   5, 205, 181, 225, 230, 178, 102,  70,  43, 221,  66,
  136, 179, 143, 209,  88,  10, 153, 105, 193, 203,  99, 204,
  140,  86, 185, 132,  15, 101,  29, 161, 176,  20,  49, 210,
  129, 149, 198, 151,  23, 172, 113,   7,  30, 202,  58,  65,
   95,  40,  98, 163
};

/*
 * Ranges:
 *   REDS1: from -32768..98302 to -383..383
 *   REDS2: from -2^31..2^31-1 to -32768..98302
 */
#define REDS1(x)    (((x) & 0xFF) - ((x) >> 8))
#define REDS2(x)    (((x) & 0xFFFF) + ((x) >> 16))

/*
 * If, upon entry, the values of q[] are all in the -N..N range (where
 * N >= 98302) then the new values of q[] are in the -2N..2N range.
 *
 * Since alpha_tab[v] <= 256, maximum allowed range is for N = 8388608.
 */

void FFT_LOOP_16_8(int *q, int rb)
{
	#ifndef COMPILER_UNROLLING
    #pragma unroll 1
    #endif
	for(int i = 0; i < 16; i += 4)
	{
		#ifndef COMPILER_UNROLLING
		#ifdef __Tahiti__
		#pragma unroll 1
		#else
		#pragma unroll
		#endif
		#endif
		for(int x = 0; x < 4; ++x)
		{
			const int m = q[(rb) + i + x];
			const int t = REDS2(mul24(q[(rb) + i + x + 16], alpha_tab[(i << 3) + (x << 3)]));
			q[(rb) + i + x] = m + t;
			q[(rb) + i + x + 16] = m - t;
		}
	}
}

void FFT_LOOP_32_4(int *q, int rb)
{
	#ifndef COMPILER_UNROLLING
    #pragma unroll 1
    #endif
    for(int i = 0; i < 32; i += 4)
    {
		#ifndef COMPILER_UNROLLING
		#pragma unroll 1
		#endif
		for(int x = 0; x < 4; ++x)
		{
			const int m = q[(rb) + i + x];
			const int t = REDS2(mul24(q[(rb) + i + x + 32], alpha_tab[(i << 2) + (x << 2)]));
			q[(rb) + i + x] = m + t;
			q[(rb) + i + x + 32] = m - t;
		}
	}
}

void FFT_LOOP_64_2(int *q, int rb)
{
	#ifndef COMPILER_UNROLLING
    #pragma unroll 1
    #endif
    for(int i = 0; i < 64; i += 4)
    {
		#ifndef COMPILER_UNROLLING
		#pragma unroll 1
		#endif
		for(int x = 0; x < 4; ++x)
		{
			const int m = q[(rb) + i + x];
			const int t = REDS2(mul24(q[(rb) + i + x + 64], alpha_tab[(i << 1) + (x << 1)]));
			q[(rb) + i + x] = m + t;
			q[(rb) + i + x + 64] = m - t;
		}
	}
}

void FFT_LOOP_128_1(int *q)
{
	//#ifndef COMPILER_UNROLLING
    #pragma unroll //32
    //#endif
    for(int i = 0; i < 128; i += 4)
    {
		//#ifndef COMPILER_UNROLLING
		#pragma unroll
		//#endif
		for(int x = 0; x < 4; ++x)
		{
			const int m = q[i + x];
			const int t = REDS2(mul24(q[i + x + 128], alpha_tab[i + x]));
			q[i + x] = m + t;
			q[i + x + 128] = m - t;
		}
	}
}

/*
 * Output ranges:
 *   d0:   min=    0   max= 1020
 *   d1:   min=  -67   max= 4587
 *   d2:   min=-4335   max= 4335
 *   d3:   min=-4147   max=  507
 *   d4:   min= -510   max=  510
 *   d5:   min= -252   max= 4402
 *   d6:   min=-4335   max= 4335
 *   d7:   min=-4332   max=  322
 */
//#define FFT8(xb, xs, d)   do { \
    s32 x0 = x[(xb)]; \
    s32 x1 = x[(xb) + (xs)]; \
    s32 x2 = x[(xb) + 2 * (xs)]; \
    s32 x3 = x[(xb) + 3 * (xs)]; \
    s32 a0 = x0 + x2; \
    s32 a1 = x0 + (x2 << 4); \
    s32 a2 = x0 - x2; \
    s32 a3 = x0 - (x2 << 4); \
    s32 b0 = x1 + x3; \
    s32 b1 = REDS1((x1 << 2) + (x3 << 6)); \
    s32 b2 = (x1 << 4) - (x3 << 4); \
    s32 b3 = REDS1((x1 << 6) + (x3 << 2)); \
    d ## 0 = a0 + b0; \
    d ## 1 = a1 + b1; \
    d ## 2 = a2 + b2; \
    d ## 3 = a3 + b3; \
    d ## 4 = a0 - b0; \
    d ## 5 = a1 - b1; \
    d ## 6 = a2 - b2; \
    d ## 7 = a3 - b3; \
  } while (0)
  
#define FFT8(xb, xs, d)	do { \
	int x0 = x[(xb)]; \
	int x1 = x[(xb) + (xs)]; \
    int x2 = x[(xb) + 2 * (xs)]; \
    int x3 = x[(xb) + 3 * (xs)]; \
	d ## 0 = (x0 + x2) + (x1 + x3); \
	d ## 4 = (x0 + x2) - (x1 + x3); \
	d ## 1 = (x0 + (x2 << 4)) + REDS1((x1 << 2) + (x3 << 6)); \
	d ## 5 = (x0 + (x2 << 4)) - REDS1((x1 << 2) + (x3 << 6)); \
	d ## 2 = (x0 - x2) + ((x1 << 4) - (x3 << 4)); \
	d ## 6 = (x0 - x2) - ((x1 << 4) - (x3 << 4)); \
	d ## 3 = (x0 - (x2 << 4)) + REDS1((x1 << 6) + (x3 << 2)); \
	d ## 7 = (x0 - (x2 << 4)) - REDS1((x1 << 6) + (x3 << 2)); \
} while(0)

//#define FFT8_A(xb, xs, d)	do { \
	const int x0 = x[(xb)]; \
	const int x1 = x[(xb) + (xs)]; \
    const int x2 = x[(xb) + 2 * (xs)]; \
    const int x3 = x[(xb) + 3 * (xs)]; \
	d[0] = (x0 + x2) + (x1 + x3); \
	d[4] = (x0 + x2) - (x1 + x3); \
	d[1] = (x0 + (x2 << 4)) + REDS1((x1 << 2) + (x3 << 6)); \
	d[5] = (x0 + (x2 << 4)) - REDS1((x1 << 2) + (x3 << 6)); \
	d[2] = (x0 - x2) + ((x1 << 4) - (x3 << 4)); \
	d[6] = (x0 - x2) - ((x1 << 4) - (x3 << 4)); \
	d[3] = (x0 - (x2 << 4)) + REDS1((x1 << 6) + (x3 << 2)); \
	d[7] = (x0 - (x2 << 4)) - REDS1((x1 << 6) + (x3 << 2)); \
} while(0)

#define FFT8_A(xb, xs, d)	do { \
	const int x0 = x[(xb)]; \
	const int x1 = x[(xb) + (xs)]; \
	d[0] = x0 + x1; \
	d[4] = x0 - x1; \
	d[1] = x0 + REDS1(x1 << 2); \
	d[5] = x0 - REDS1(x1 << 2); \
	d[2] = x0 + (x1 << 4); \
	d[6] = x0 - (x1 << 4); \
	d[3] = x0 + REDS1(x1 << 6); \
	d[7] = x0 - REDS1(x1 << 6); \
} while(0)

//#define FFT16(q, x, xb, xs, rb)	do { \
	int d1[8], d2[8]; \
	FFT8_A(xb, (xs) << 1, d1); \
	FFT8_A((xb) + (xs), (xs) << 1, d2); \
	for(int i = 0; i < 8; ++i) \
		q[(rb) + i] = d1[i] + (d2[i] << i); \
	\
	for(int i = 0; i < 8; ++i) \
		q[(rb) + i + 8] = d1[i] - (d2[i] << i); \
} while(0)

/*
void FFT16(int *restrict q, const uchar *restrict x, int xb, int xs, int rb)
{
	//int d1[8], d2[8];
	//FFT8_A(xb, (xs) << 1, d1);
	//FFT8_A((xb) + (xs), (xs) << 1, d2);
	
	const int x0 = x[xb], x1 = x[xb + (xs << 1)], x2 = x[xb + (xs << 2)], x3 = x[xb + (xs * 5)];
	const int d1[8] = { (x0 + x2) + (x1 + x3), (x0 + (x2 << 4)) + REDS1((x1 << 2) + (x3 << 6)), (x0 - x2) + ((x1 << 4) - (x3 << 4)), (x0 - (x2 << 4)) + REDS1((x1 << 6) + (x3 << 2)), \
						(x0 + x2) - (x1 + x3), (x0 + (x2 << 4)) - REDS1((x1 << 2) + (x3 << 6)), (x0 - x2) - ((x1 << 4) - (x3 << 4)), (x0 - (x2 << 4)) - REDS1((x1 << 6) + (x3 << 2)) };
	const int x20 = x[xb + xs], x21 = x[xb + (xs * 3)], x22 = x[xb + (xs * 5)], x23 = x[xb + (xs * 6)];
	const int d2[8] = { (x20 + x22) + (x21 + x23), (x20 + (x22 << 4)) + REDS1((x21 << 2) + (x23 << 6)), (x20 - x22) + ((x21 << 4) - (x23 << 4)), (x20 - (x22 << 4)) + REDS1((x21 << 6) + (x23 << 2)), \
						(x20 + x22) - (x21 + x23), (x20 + (x22 << 4)) - REDS1((x21 << 2) + (x23 << 6)), (x20 - x22) - ((x21 << 4) - (x23 << 4)), (x20 - (x22 << 4)) - REDS1((x21 << 6) + (x23 << 2)) };
	
	#pragma unroll
	for(int i = 0; i < 8; ++i)
	{
		q[rb + i] = d1[i] + (d2[i] << i);
		q[rb + i + 8] = d1[i] - (d2[i] << i);
	}
}*/

void FFT16(int *restrict q, const uchar *restrict x, int xb, int rb)
{
	/*int d1[8], d2[8];
	FFT8_A(xb, (xs) << 1, d1);
	FFT8_A((xb) + (xs), (xs) << 1, d2);*/
	
	const int x0 = x[xb], x1 = x[xb + 32];
	const int d1[8] = { x0 + x1, x0 + REDS1(x1 << 2), x0 + (x1 << 4), x0 + REDS1(x1 << 6), \
						x0 - x1, x0 - REDS1(x1 << 2), x0 - (x1 << 4), x0 - REDS1(x1 << 6) };
	const int x20 = x[xb + 16], x21 = x[xb + 48];
	const int d2[8] = { x20 + x21, x20 + REDS1(x21 << 2), x20 + (x21 << 4), x20 + REDS1(x21 << 6), \
						x20 - x21, x20 - REDS1(x21 << 2), x20 - (x21 << 4), x20 - REDS1(x21 << 6) };
	
	#ifndef COMPILER_UNROLLING
	#pragma unroll
	#endif
	for(int i = 0; i < 8; ++i)
	{
		q[rb + i] = d1[i] + (d2[i] << i);
		q[rb + i + 8] = d1[i] - (d2[i] << i);
	}
}

/*
 * When k=16, we have alpha=2. Multiplication by alpha^i is then reduced
 * to some shifting.
 *
 * Output: within -591471..591723
 */
//#define FFT16(xb, xs, rb)   do { \
    s32 d1_0, d1_1, d1_2, d1_3, d1_4, d1_5, d1_6, d1_7; \
    s32 d2_0, d2_1, d2_2, d2_3, d2_4, d2_5, d2_6, d2_7; \
    FFT8(xb, (xs) << 1, d1_); \
    FFT8((xb) + (xs), (xs) << 1, d2_); \
    q[(rb) +  0] = d1_0 + d2_0; \
    q[(rb) +  1] = d1_1 + (d2_1 << 1); \
    q[(rb) +  2] = d1_2 + (d2_2 << 2); \
    q[(rb) +  3] = d1_3 + (d2_3 << 3); \
    q[(rb) +  4] = d1_4 + (d2_4 << 4); \
    q[(rb) +  5] = d1_5 + (d2_5 << 5); \
    q[(rb) +  6] = d1_6 + (d2_6 << 6); \
    q[(rb) +  7] = d1_7 + (d2_7 << 7); \
    q[(rb) +  8] = d1_0 - d2_0; \
    q[(rb) +  9] = d1_1 - (d2_1 << 1); \
    q[(rb) + 10] = d1_2 - (d2_2 << 2); \
    q[(rb) + 11] = d1_3 - (d2_3 << 3); \
    q[(rb) + 12] = d1_4 - (d2_4 << 4); \
    q[(rb) + 13] = d1_5 - (d2_5 << 5); \
    q[(rb) + 14] = d1_6 - (d2_6 << 6); \
    q[(rb) + 15] = d1_7 - (d2_7 << 7); \
  } while (0)

/*
 * Output range: |q| <= 1183446
 */
//#define FFT32(xb, xs, rb, id)   do { \
    FFT16(q, x, xb, (xs) << 1, rb); \
    FFT16(q, x, (xb) + (xs), (xs) << 1, (rb) + 16); \
    FFT_LOOP_16_8(q, rb); \
  } while (0)

void FFT32(int *restrict q, const uchar *restrict x, int xb, int xs, int rb)
{
	FFT16(q, x, xb, rb);
	FFT16(q, x, xb + xs, rb + 16);
	FFT_LOOP_16_8(q, rb);
}

/*
 * Output range: |q| <= 2366892
 */
//#define FFT64(xb, xs, rb)   do { \
  FFT32(xb, (xs) << 1, (rb), label_a); \
  FFT32((xb) + (xs), (xs) << 1, (rb) + 32, label_b); \
  FFT_LOOP_32_4(q, rb); \
  } while (0)

#define FFT64(xb, rb)   do { \
  FFT32(q, x, xb, 8, (rb)); \
  FFT32(q, x, (xb) + 4, 8, (rb) + 32); \
  FFT_LOOP_32_4(q, rb); \
  } while (0)

/*
 * Output range: |q| <= 9467568
 */
#define FFT256   do { \
    FFT64(0, 0); \
    FFT64(2, 64); \
    FFT_LOOP_64_2(q, 0); \
    FFT64(1, 128); \
    FFT64(3, 192); \
    FFT_LOOP_64_2(q, 128); \
    FFT_LOOP_128_1(q); \
  } while (0)

// xb == amount added to it
// xs == stripe

/*
 * beta^(255*i) mod 257
 */
__constant static const ushort yoff_b_n[] = {
    1, 163,  98,  40,  95,  65,  58, 202,  30,   7, 113, 172,
   23, 151, 198, 149, 129, 210,  49,  20, 176, 161,  29, 101,
   15, 132, 185,  86, 140, 204,  99, 203, 193, 105, 153,  10,
   88, 209, 143, 179, 136,  66, 221,  43,  70, 102, 178, 230,
  225, 181, 205,   5,  44, 233, 200, 218,  68,  33, 239, 150,
   35,  51,  89, 115, 241, 219, 231, 131,  22, 245, 100, 109,
   34, 145, 248,  75, 146, 154, 173, 186, 249, 238, 244, 194,
   11, 251,  50, 183,  17, 201, 124, 166,  73,  77, 215,  93,
  253, 119, 122,  97, 134, 254,  25, 220, 137, 229,  62,  83,
  165, 167, 236, 175, 255, 188,  61, 177,  67, 127, 141, 110,
  197, 243,  31, 170, 211, 212, 118, 216, 256,  94, 159, 217,
  162, 192, 199,  55, 227, 250, 144,  85, 234, 106,  59, 108,
  128,  47, 208, 237,  81,  96, 228, 156, 242, 125,  72, 171,
  117,  53, 158,  54,  64, 152, 104, 247, 169,  48, 114,  78,
  121, 191,  36, 214, 187, 155,  79,  27,  32,  76,  52, 252,
  213,  24,  57,  39, 189, 224,  18, 107, 222, 206, 168, 142,
   16,  38,  26, 126, 235,  12, 157, 148, 223, 112,   9, 182,
  111, 103,  84,  71,   8,  19,  13,  63, 246,   6, 207,  74,
  240,  56, 133,  91, 184, 180,  42, 164,   4, 138, 135, 160,
  123,   3, 232,  37, 120,  28, 195, 174,  92,  90,  21,  82,
    2,  69, 196,  80, 190, 130, 116, 147,  60,  14, 226,  87,
   46,  45, 139,  41
};

#define INNER(l, h, mm) (upsample(((ushort)mul24((h), (mm))), ((ushort)mul24((l), (mm)))))

#define W_BIG(sb, o1, o2, mm) \
	INNER(q[16 * (sb) + 2 * 0 + o1], q[16 * (sb) + 2 * 0 + o2], mm), \
	INNER(q[16 * (sb) + 2 * 1 + o1], q[16 * (sb) + 2 * 1 + o2], mm), \
	INNER(q[16 * (sb) + 2 * 2 + o1], q[16 * (sb) + 2 * 2 + o2], mm), \
	INNER(q[16 * (sb) + 2 * 3 + o1], q[16 * (sb) + 2 * 3 + o2], mm), \
	INNER(q[16 * (sb) + 2 * 4 + o1], q[16 * (sb) + 2 * 4 + o2], mm), \
	INNER(q[16 * (sb) + 2 * 5 + o1], q[16 * (sb) + 2 * 5 + o2], mm), \
	INNER(q[16 * (sb) + 2 * 6 + o1], q[16 * (sb) + 2 * 6 + o2], mm), \
	INNER(q[16 * (sb) + 2 * 7 + o1], q[16 * (sb) + 2 * 7 + o2], mm)

/*
static const __constant uchar SIMD_WB_ARG1[4][8] =
{
	{ 4, 6, 0, 2, 7, 5, 3, 1 },
	{ 15, 11, 12, 8, 9, 13, 10, 14 },
	{ 17, 18, 23, 20, 22, 21, 16, 19 },
	{ 30, 24, 25, 31, 27, 29, 28, 26 }
};

static const __constant short SIMD_WB_ARG2[4] = { 0, 0, -256, -383 };

static const __constant short SIMD_WB_ARG3[4] = { 1, 1, -128, -255 };

static const __constant short SIMD_WB_ARG4[4] = { 185, 185, 233, 233 };

static const __constant uint8 SIMD_PP8_V[7] =
{
	(uint8)(1, 0, 3, 2, 5, 4, 7, 6),
	(uint8)(6, 7, 4, 5, 2, 3, 0, 1),
	(uint8)(2, 3, 0, 1, 6, 7, 4, 5),
	(uint8)(3, 2, 1, 0, 7, 6, 5, 4),
	(uint8)(5, 4, 7, 6, 1, 0, 3, 2),
	(uint8)(7, 6, 5, 4, 3, 2, 1, 0),
	(uint8)(4, 5, 6, 7, 0, 1, 2, 3)
};
*/

#define IF(x, y, z)    	bitselect((z), (y), (x))
#define MAJ(x, y, z)	bitselect((x), (y), ((z) ^ (x)))

#define STEP_BIG_IF(A, B, C, D, w, r, s, pp8b)	do { \
	const uint8 tA = rotate(A, r); \
	const uint8 tt = D + w + IF(A, B, C); \
	A = rotate(tt, s) + shuffle(tA, pp8b); \
	\
	D = C; C = B; B = tA; \
} while(0)

#define STEP_BIG_MAJ(A, B, C, D, w, r, s, pp8b)	do { \
	const uint8 tA = rotate(A, r); \
	const uint8 tt = D + w + MAJ(A, B, C); \
	A = rotate(tt, s) + shuffle(tA, pp8b); \
	\
	D = C; C = B; B = tA; \
} while(0)

/*void STEP_BIG_IF(uint8 *restrict A, uint8 *restrict B, uint8 *restrict C, uint8 *restrict D, const uint8 w, const uint r, const uint s, const uint8 pp8b)
{
	const uint8 tA = rotate(*A, r);
	*A = rotate((*D + w + IF(*A, *B, *C)), s) + shuffle(tA, pp8b);
	*D = *C; *C = *B; *B = tA;
}

void STEP_BIG_MAJ(uint8 *restrict A, uint8 *restrict B, uint8 *restrict C, uint8 *restrict D, const uint8 w, const uint r, const uint s, const uint8 pp8b)
{
	const uint8 tA = rotate(*A, r);
	*A = rotate((*D + w + MAJ(*A, *B, *C)), s) + shuffle(tA, pp8b);
	*D = *C; *C = *B; *B = tA;
}*/

void SIMD_Expand(int *restrict q, const uchar *restrict x)
{
	FFT256;
  
	/*FFT64(0, 0);
	FFT64(2, 64);
	FFT_LOOP_64_2(q, 0);
	FFT64(1, 128);
	FFT64(3, 192);
	FFT_LOOP_64_2(q, 128);
	FFT_LOOP_128_1(q);*/
	
	/*
	//#pragma unroll 4
	for (int i = 0; i < 128; i ++)
	{
		const int tq = REDS1(REDS1(q[i] + yoff_b_n[i]));
		//q[i] = (tq <= 128 ? tq : tq - 257);
		q[i] = select(tq - 257, tq, tq <= 128);
	}
	
	//#pragma unroll 8
	for (int i = 128; i < 256; i ++)
	{	
		const int tq = REDS1(REDS1(q[i] + yoff_b_n[i]));
		//q[i] = (tq <= 128 ? tq : tq - 257);
		q[i] = select(tq - 257, tq, tq <= 128);
	}
	*/
	
	for(int i = 0; i < 256; ++i)
	{
		const int tq = REDS1(REDS1(q[i] + yoff_b_n[i]));
		q[i] = select(tq - 257, tq, tq <= 128);
	}
}

static const __constant uint8 perms[7] = 
{
	(uint8)(1, 0, 3, 2, 5, 4, 7, 6),
	(uint8)(6, 7, 4, 5, 2, 3, 0, 1),
	(uint8)(2, 3, 0, 1, 6, 7, 4, 5),
	(uint8)(3, 2, 1, 0, 7, 6, 5, 4),
	(uint8)(5, 4, 7, 6, 1, 0, 3, 2),
	(uint8)(7, 6, 5, 4, 3, 2, 1, 0),
	(uint8)(4, 5, 6, 7, 0, 1, 2, 3),
};

static const __constant uint rotconsts[4][4] =
{
	{ 3U, 23U, 17U, 27U },
	{ 28U, 19U, 22U, 7U },
	{ 29U, 9U, 15U, 5U },
	{ 4U, 13U, 10U, 25U }
};

static const __constant uchar SIMD_WB_ARG1[4][8] =
{
	{ 4, 6, 0, 2, 7, 5, 3, 1 },
	{ 15, 11, 12, 8, 9, 13, 10, 14 },
	{ 17, 18, 23, 20, 22, 21, 16, 19 },
	{ 30, 24, 25, 31, 27, 29, 28, 26 }
};

static const __constant short SIMD_WB_ARG2[4] = { 0, 0, -256, -383 };

static const __constant short SIMD_WB_ARG3[4] = { 1, 1, -128, -255 };

static const __constant short SIMD_WB_ARG4[4] = { 185, 185, 233, 233 };

void FOUR_ROUNDS_BIG_1(const int *restrict q, uint8 *restrict A, uint8 *restrict B, uint8 *restrict C, uint8 *restrict D)
{
	#if defined(__Tahiti__) || defined(__Pitcairn__)
	
	STEP_BIG_IF(*A, *B, *C, *D, (uint8)(W_BIG(4, 0, 1, 185)), 3U, 23U, (uint8)(1, 0, 3, 2, 5, 4, 7, 6));
    STEP_BIG_IF(*A, *B, *C, *D, (uint8)(W_BIG(6, 0, 1, 185)), 23U, 17U, (uint8)(6, 7, 4, 5, 2, 3, 0, 1));
    STEP_BIG_IF(*A, *B, *C, *D, (uint8)(W_BIG(0, 0, 1, 185)), 17U, 27U, (uint8)(2, 3, 0, 1, 6, 7, 4, 5));
    STEP_BIG_IF(*A, *B, *C, *D, (uint8)(W_BIG(2, 0, 1, 185)), 27U, 3U, (uint8)(3, 2, 1, 0, 7, 6, 5, 4));
    STEP_BIG_MAJ(*A, *B, *C, *D, (uint8)(W_BIG(7, 0, 1, 185)), 3U, 23U, (uint8)(5, 4, 7, 6, 1, 0, 3, 2));
    STEP_BIG_MAJ(*A, *B, *C, *D, (uint8)(W_BIG(5, 0, 1, 185)), 23U, 17U, (uint8)(7, 6, 5, 4, 3, 2, 1, 0));
    STEP_BIG_MAJ(*A, *B, *C, *D, (uint8)(W_BIG(3, 0, 1, 185)), 17U, 27U, (uint8)(4, 5, 6, 7, 0, 1, 2, 3));
    STEP_BIG_MAJ(*A, *B, *C, *D, (uint8)(W_BIG(1, 0, 1, 185)), 27U, 3U, (uint8)(1, 0, 3, 2, 5, 4, 7, 6));
    
    STEP_BIG_IF(*A, *B, *C, *D, (uint8)(W_BIG(15, 0, 1, 185)), 28U, 19U, (uint8)(6, 7, 4, 5, 2, 3, 0, 1));
	STEP_BIG_IF(*A, *B, *C, *D, (uint8)(W_BIG(11, 0, 1, 185)), 19U, 22U, (uint8)(2, 3, 0, 1, 6, 7, 4, 5));
	STEP_BIG_IF(*A, *B, *C, *D, (uint8)(W_BIG(12, 0, 1, 185)), 22U, 7U, (uint8)(3, 2, 1, 0, 7, 6, 5, 4));
	STEP_BIG_IF(*A, *B, *C, *D, (uint8)(W_BIG(8, 0, 1, 185)), 7U, 28U, (uint8)(5, 4, 7, 6, 1, 0, 3, 2));
	STEP_BIG_MAJ(*A, *B, *C, *D, (uint8)(W_BIG(9, 0, 1, 185)), 28U, 19U, (uint8)(7, 6, 5, 4, 3, 2, 1, 0));
	STEP_BIG_MAJ(*A, *B, *C, *D, (uint8)(W_BIG(13, 0, 1, 185)), 19U, 22U, (uint8)(4, 5, 6, 7, 0, 1, 2, 3));
	STEP_BIG_MAJ(*A, *B, *C, *D, (uint8)(W_BIG(10, 0, 1, 185)), 22U, 7U, (uint8)(1, 0, 3, 2, 5, 4, 7, 6));
	STEP_BIG_MAJ(*A, *B, *C, *D, (uint8)(W_BIG(14, 0, 1, 185)), 7U, 28U, (uint8)(6, 7, 4, 5, 2, 3, 0, 1));
	
	STEP_BIG_IF(*A, *B, *C, *D, (uint8)(W_BIG(17, -256, -128, 233)), 29U, 9U, (uint8)(2, 3, 0, 1, 6, 7, 4, 5));
	STEP_BIG_IF(*A, *B, *C, *D, (uint8)(W_BIG(18, -256, -128, 233)), 9U, 15U, (uint8)(3, 2, 1, 0, 7, 6, 5, 4));
	STEP_BIG_IF(*A, *B, *C, *D, (uint8)(W_BIG(23, -256, -128, 233)), 15U, 5U, (uint8)(5, 4, 7, 6, 1, 0, 3, 2));
	STEP_BIG_IF(*A, *B, *C, *D, (uint8)(W_BIG(20, -256, -128, 233)), 5U, 29U, (uint8)(7, 6, 5, 4, 3, 2, 1, 0));
	STEP_BIG_MAJ(*A, *B, *C, *D, (uint8)(W_BIG(22, -256, -128, 233)), 29U, 9U, (uint8)(4, 5, 6, 7, 0, 1, 2, 3));
	STEP_BIG_MAJ(*A, *B, *C, *D, (uint8)(W_BIG(21, -256, -128, 233)), 9U, 15U, (uint8)(1, 0, 3, 2, 5, 4, 7, 6));
	STEP_BIG_MAJ(*A, *B, *C, *D, (uint8)(W_BIG(16, -256, -128, 233)), 15U, 5U, (uint8)(6, 7, 4, 5, 2, 3, 0, 1));
	STEP_BIG_MAJ(*A, *B, *C, *D, (uint8)(W_BIG(19, -256, -128, 233)), 5U, 29U, (uint8)(2, 3, 0, 1, 6, 7, 4, 5));
	
	STEP_BIG_IF(*A, *B, *C, *D, (uint8)(W_BIG(30, -383, -255, 233)), 4U, 13U, (uint8)(3, 2, 1, 0, 7, 6, 5, 4));
	STEP_BIG_IF(*A, *B, *C, *D, (uint8)(W_BIG(24, -383, -255, 233)), 13U, 10U, (uint8)(5, 4, 7, 6, 1, 0, 3, 2));
	STEP_BIG_IF(*A, *B, *C, *D, (uint8)(W_BIG(25, -383, -255, 233)), 10U, 25U, (uint8)(7, 6, 5, 4, 3, 2, 1, 0));
	STEP_BIG_IF(*A, *B, *C, *D, (uint8)(W_BIG(31, -383, -255, 233)), 25U, 4U, (uint8)(4, 5, 6, 7, 0, 1, 2, 3));
	STEP_BIG_MAJ(*A, *B, *C, *D, (uint8)(W_BIG(27, -383, -255, 233)), 4U, 13U, (uint8)(1, 0, 3, 2, 5, 4, 7, 6));
	STEP_BIG_MAJ(*A, *B, *C, *D, (uint8)(W_BIG(29, -383, -255, 233)), 13U, 10U, (uint8)(6, 7, 4, 5, 2, 3, 0, 1));
	STEP_BIG_MAJ(*A, *B, *C, *D, (uint8)(W_BIG(28, -383, -255, 233)), 10U, 25U, (uint8)(2, 3, 0, 1, 6, 7, 4, 5));
	STEP_BIG_MAJ(*A, *B, *C, *D, (uint8)(W_BIG(26, -383, -255, 233)), 25U, 4U, (uint8)(3, 2, 1, 0, 7, 6, 5, 4));
	
	#else
	
	#pragma unroll 1
	for(int i = 0; i < 4; ++i)
	{
		/*#pragma unroll 1
		for(int x = 0; x < 4; ++x)
		{
			STEP_BIG_IF(*A, *B, *C, *D, (uint8)(W_BIG(SIMD_WB_ARG1[i][x], SIMD_WB_ARG2[i], SIMD_WB_ARG3[i], SIMD_WB_ARG4[i])), rotconsts[i][x], rotconsts[i][(x + 1) & 3], perms[(i + x) % 7]);
		}
		
		#pragma unroll 1
		for(int x = 0; x < 4; ++x)
		{
			STEP_BIG_MAJ(*A, *B, *C, *D, (uint8)(W_BIG(SIMD_WB_ARG1[i][x + 4], SIMD_WB_ARG2[i], SIMD_WB_ARG3[i], SIMD_WB_ARG4[i])), rotconsts[i][x], rotconsts[i][(x + 1) & 3], perms[((x == 3) ? i % 7 : (i + x + 4) % 7)]);
		}*/
		
		STEP_BIG_IF(*A, *B, *C, *D, (uint8)(W_BIG(SIMD_WB_ARG1[i][0], SIMD_WB_ARG2[i], SIMD_WB_ARG3[i], SIMD_WB_ARG4[i])), rotconsts[i][0], rotconsts[i][1], perms[(i + 0) % 7]);
		STEP_BIG_IF(*A, *B, *C, *D, (uint8)(W_BIG(SIMD_WB_ARG1[i][1], SIMD_WB_ARG2[i], SIMD_WB_ARG3[i], SIMD_WB_ARG4[i])), rotconsts[i][1], rotconsts[i][2], perms[(i + 1) % 7]);
		STEP_BIG_IF(*A, *B, *C, *D, (uint8)(W_BIG(SIMD_WB_ARG1[i][2], SIMD_WB_ARG2[i], SIMD_WB_ARG3[i], SIMD_WB_ARG4[i])), rotconsts[i][2], rotconsts[i][3], perms[(i + 2) % 7]);
		STEP_BIG_IF(*A, *B, *C, *D, (uint8)(W_BIG(SIMD_WB_ARG1[i][3], SIMD_WB_ARG2[i], SIMD_WB_ARG3[i], SIMD_WB_ARG4[i])), rotconsts[i][3], rotconsts[i][0], perms[(i + 3) % 7]);
		STEP_BIG_MAJ(*A, *B, *C, *D, (uint8)(W_BIG(SIMD_WB_ARG1[i][4], SIMD_WB_ARG2[i], SIMD_WB_ARG3[i], SIMD_WB_ARG4[i])), rotconsts[i][0], rotconsts[i][1], perms[(i + 4) % 7]);
		STEP_BIG_MAJ(*A, *B, *C, *D, (uint8)(W_BIG(SIMD_WB_ARG1[i][5], SIMD_WB_ARG2[i], SIMD_WB_ARG3[i], SIMD_WB_ARG4[i])), rotconsts[i][1], rotconsts[i][2], perms[(i + 5) % 7]);
		STEP_BIG_MAJ(*A, *B, *C, *D, (uint8)(W_BIG(SIMD_WB_ARG1[i][6], SIMD_WB_ARG2[i], SIMD_WB_ARG3[i], SIMD_WB_ARG4[i])), rotconsts[i][2], rotconsts[i][3], perms[(i + 6) % 7]);
		STEP_BIG_MAJ(*A, *B, *C, *D, (uint8)(W_BIG(SIMD_WB_ARG1[i][7], SIMD_WB_ARG2[i], SIMD_WB_ARG3[i], SIMD_WB_ARG4[i])) , rotconsts[i][3], rotconsts[i][0], perms[(i + 0) % 7]);
	}
	
	#endif
}

static const __constant uint8 precomp[4][8] = 
{
	{
		(uint8)(0x531B1720U, 0xAC2CDE09U, 0x0B902D87U, 0x2369B1F4U, 0x2931AA01U, 0x02E4B082U, 0xC914C914U, 0xC1DAE1A6U),
		(uint8)(0xF18C2B5CU, 0x08AC306BU, 0x27BFC914U, 0xCEDC548DU, 0xC630C4BEU, 0xF18C4335U, 0xF0D3427CU, 0xBE3DA380U),
		(uint8)(0x143C02E4U, 0xA948C630U, 0xA4F2DE09U, 0xA71D2085U, 0xA439BD84U, 0x109FCD6AU, 0xEEA8EF61U, 0xA5AB1CE8U),
		(uint8)(0x0B90D4A4U, 0x3D6D039DU, 0x25944D53U, 0xBAA0E034U, 0x5BC71E5AU, 0xB1F4F2FEU, 0x12CADE09U, 0x548D41C3U),
		(uint8)(0x3CB4F80DU, 0x36ECEBC4U, 0xA66443EEU, 0x43351ABDU, 0xC7A20C49U, 0xEB0BB366U, 0xF5293F98U, 0x49B6DE09U),
		(uint8)(0x531B29EAU, 0x02E402E4U, 0xDB25C405U, 0x53D4E543U, 0x0AD71720U, 0xE1A61A04U, 0xB87534C1U, 0x3EDF43EEU),
		(uint8)(0x213E50F0U, 0x39173EDFU, 0xA9485B0EU, 0xEEA82EF9U, 0x14F55771U, 0xFAF15546U, 0x3D6DD9B3U, 0xAB73B92EU),
		(uint8)(0x582A48FDU, 0xEEA81892U, 0x4F7EAA01U, 0xAF10A88FU, 0x11581720U, 0x34C124DBU, 0xD1C0AB73U, 0x1E5AF0D3U)
	},
	{
		(uint8)(0xC34C07F3U, 0xC914143CU, 0x599CBC12U, 0xBCCBE543U, 0x385EF3B7U, 0x14F54C9AU, 0x0AD7C068U, 0xB64A21F7U),
		(uint8)(0xDEC2AF10U, 0xC6E9C121U, 0x56B8A4F2U, 0x1158D107U, 0xEB0BA88FU, 0x050FAABAU, 0xC293264DU, 0x548D46D2U),
		(uint8)(0xACE5E8E0U, 0x53D421F7U, 0xF470D279U, 0xDC974E0CU, 0xD6CF55FFU, 0xFD1C4F7EU, 0x36EC36ECU, 0x3E261E5AU),
		(uint8)(0xEBC4FD1CU, 0x56B839D0U, 0x5B0E21F7U, 0x58E3DF7BU, 0x5BC7427CU, 0xEF613296U, 0x1158109FU, 0x5A55E318U),
		(uint8)(0xA7D6B703U, 0x1158E76EU, 0xB08255FFU, 0x50F05771U, 0xEEA8E8E0U, 0xCB3FDB25U, 0x2E40548DU, 0xE1A60F2DU),
		(uint8)(0xACE5D616U, 0xFD1CFD1CU, 0x24DB3BFBU, 0xAC2C1ABDU, 0xF529E8E0U, 0x1E5AE5FCU, 0x478BCB3FU, 0xC121BC12U),
		(uint8)(0xF4702B5CU, 0xC293FC63U, 0xDA6CB2ADU, 0x45601FCCU, 0xA439E1A6U, 0x4E0C0D02U, 0xED3621F7U, 0xAB73BE3DU),
		(uint8)(0x0E74D4A4U, 0xF754CF95U, 0xD84136ECU, 0x3124AB73U, 0x39D03B42U, 0x0E74BCCBU, 0x0F2DBD84U, 0x41C35C80U),
	},
	{
		(uint8)(0xA4135BEDU, 0xE10E1EF2U, 0x6C4F93B1U, 0x6E2191DFU, 0xE2E01D20U, 0xD1952E6BU, 0x6A7D9583U, 0x131DECE3U),
		(uint8)(0x369CC964U, 0xFB73048DU, 0x9E9D6163U, 0x280CD7F4U, 0xD9C6263AU, 0x1062EF9EU, 0x2AC7D539U, 0xAD2D52D3U),
		(uint8)(0x0A03F5FDU, 0x197CE684U, 0xAA72558EU, 0xDE5321ADU, 0xF0870F79U, 0x607A9F86U, 0xAFE85018U, 0x2AC7D539U),
		(uint8)(0xE2E01D20U, 0x2AC7D539U, 0xC6A93957U, 0x624C9DB4U, 0x6C4F93B1U, 0x641E9BE2U, 0x452CBAD4U, 0x263AD9C6U),
		(uint8)(0xC964369CU, 0xC3053CFBU, 0x452CBAD4U, 0x95836A7DU, 0x4AA2B55EU, 0xAB5B54A5U, 0xAC4453BCU, 0x74808B80U),
		(uint8)(0xCB3634CAU, 0xFC5C03A4U, 0x4B8BB475U, 0x21ADDE53U, 0xE2E01D20U, 0xDF3C20C4U, 0xBD8F4271U, 0xAA72558EU),
		(uint8)(0xFC5C03A4U, 0x48D0B730U, 0x2AC7D539U, 0xD70B28F5U, 0x53BCAC44U, 0x3FB6C04AU, 0x14EFEB11U, 0xDB982468U),
		(uint8)(0x9A1065F0U, 0xB0D14F2FU, 0x8D5272AEU, 0xC4D73B29U, 0x91DF6E21U, 0x949A6B66U, 0x303DCFC3U, 0x5932A6CEU),
	},
	{
		(uint8)(0x1234EDCCU, 0xF5140AECU, 0xCDF1320FU, 0x3DE4C21CU, 0x48D0B730U, 0x1234EDCCU, 0x131DECE3U, 0x52D3AD2DU),
		(uint8)(0xE684197CU, 0x6D3892C8U, 0x72AE8D52U, 0x6FF3900DU, 0x73978C69U, 0xEB1114EFU, 0x15D8EA28U, 0x71C58E3BU),
		(uint8)(0x90F66F0AU, 0x15D8EA28U, 0x9BE2641EU, 0x65F09A10U, 0xEA2815D8U, 0xBD8F4271U, 0x3A40C5C0U, 0xD9C6263AU),
		(uint8)(0xB38C4C74U, 0xBAD4452CU, 0x70DC8F24U, 0xAB5B54A5U, 0x46FEB902U, 0x1A65E59BU, 0x0DA7F259U, 0xA32A5CD6U),
		(uint8)(0xD62229DEU, 0xB81947E7U, 0x6D3892C8U, 0x15D8EA28U, 0xE59B1A65U, 0x065FF9A1U, 0xB2A34D5DU, 0x6A7D9583U),
		(uint8)(0x975568ABU, 0xFC5C03A4U, 0x2E6BD195U, 0x966C6994U, 0xF2590DA7U, 0x263AD9C6U, 0x5A1BA5E5U, 0xB0D14F2FU),
		(uint8)(0x975568ABU, 0x6994966CU, 0xF1700E90U, 0xD3672C99U, 0xCC1F33E1U, 0xFC5C03A4U, 0x452CBAD4U, 0x4E46B1BAU),
		(uint8)(0xF1700E90U, 0xB2A34D5DU, 0xD0AC2F54U, 0x5760A8A0U, 0x8C697397U, 0x624C9DB4U, 0xE85617AAU, 0x95836A7DU),
	}
};

void FOUR_ROUNDS_BIG_2(uint8 *restrict A, uint8 *restrict B, uint8 *restrict C, uint8 *restrict D)
{
	#pragma unroll
	for(int i = 0; i < 4; ++i)
	{
		#pragma unroll
		for(int x = 0; x < 4; ++x)
		{
			STEP_BIG_IF(*A, *B, *C, *D, precomp[i][x], rotconsts[i][x], rotconsts[i][(x + 1) & 3], perms[(i + x) % 7]);
		}
		
		#pragma unroll
		for(int x = 0; x < 4; ++x)
		{
			STEP_BIG_MAJ(*A, *B, *C, *D, precomp[i][x + 4], rotconsts[i][x], rotconsts[i][(x + 1) & 3], perms[((x == 3) ? i % 7 : (i + x + 4) % 7)]);
		}
	}
}

static const __constant uint8 precomp2[4] =
{
	(uint8)(0x0BA16B95, 0x72F999AD, 0x9FECC2AE, 0xBA3264FC, 0x5E894929, 0x8E9F30E5, 0x2F1DAA37, 0xF0F2C558),
	(uint8)(0xAC506643, 0xA90635A5, 0xE25B878B, 0xAAB7878F, 0x88817F7A, 0x0A02892B, 0x559A7550, 0x598F657E),
	(uint8)(0x7EEF60A1, 0x6B70E3E8, 0x9C1714D1, 0xB958E2A8, 0xAB02675E, 0xED1C014F, 0xCD8D65BB, 0xFDB7A257),
	(uint8)(0x09254899, 0xD699C7BC, 0x9019B6DC, 0x2B9022E4, 0x8FA14956, 0x21BF9BD3, 0xB94D0943, 0x6FFDDC22)
};

void SIMD_Compress(const int *restrict q, uint8 *A, uint8 *B, uint8 *C, uint8 *D)
{
	FOUR_ROUNDS_BIG_1(q, A, B, C, D);
	
	/*STEP_BIG_IF(A, B, &C, &D, (uint8)(0x0BA16B95, 0x72F999AD, 0x9FECC2AE, 0xBA3264FC, 0x5E894929, 0x8E9F30E5, 0x2F1DAA37, 0xF0F2C558), 4, 13, (uint8)(5, 4, 7, 6, 1, 0, 3, 2));
	
	STEP_BIG_IF(A, B, &C, &D, (uint8)(0xAC506643, 0xA90635A5, 0xE25B878B, 0xAAB7878F, 0x88817F7A, 0x0A02892B, 0x559A7550, 0x598F657E), 13, 10, (uint8)(7, 6, 5, 4, 3, 2, 1, 0));
	
	STEP_BIG_IF(A, B, &C, &D, (uint8)(0x7EEF60A1, 0x6B70E3E8, 0x9C1714D1, 0xB958E2A8, 0xAB02675E, 0xED1C014F, 0xCD8D65BB, 0xFDB7A257), 10, 25, (uint8)(4, 5, 6, 7, 0, 1, 2, 3));
	
	STEP_BIG_IF(A, B, &C, &D, (uint8)(0x09254899, 0xD699C7BC, 0x9019B6DC, 0x2B9022E4, 0x8FA14956, 0x21BF9BD3, 0xB94D0943, 0x6FFDDC22), 25, 4, (uint8)(1, 0, 3, 2, 5, 4, 7, 6));*/
	
	#pragma unroll
	for(int i = 0; i < 4; ++i)
	{
		//STEP_BIG_IF(A, B, &C, &D, precomp2[i], rotconsts[3][i], rotconsts[3][(i + 1) & 3], perms[(i + 4) % 7]);
		STEP_BIG_IF(*A, *B, *C, *D, precomp2[i], rotconsts[3][i], rotconsts[3][(i + 1) & 3], perms[(i + 4) % 7]);
	}
	
	uint8 A2 = *A, B2 = *B, C2 = *C, D2 = *D;
	
	(*A).s0 ^= 0x200;
	
	FOUR_ROUNDS_BIG_2(A, B, C, D);
	
	/*STEP_BIG_IF(A, B, &C, &D, A2, 4, 13, (uint8)(5, 4, 7, 6, 1, 0, 3, 2));
	
	STEP_BIG_IF(A, B, &C, &D, B2, 13, 10, (uint8)(7, 6, 5, 4, 3, 2, 1, 0));
	
	STEP_BIG_IF(A, B, &C, &D, C2, 10, 25, (uint8)(4, 5, 6, 7, 0, 1, 2, 3));
	
	STEP_BIG_IF(A, B, &C, &D, D2, 25, 4, (uint8)(1, 0, 3, 2, 5, 4, 7, 6));*/
	
	STEP_BIG_IF(*A, *B, *C, *D, A2, 4, 13, (uint8)(5, 4, 7, 6, 1, 0, 3, 2));
	
	STEP_BIG_IF(*A, *B, *C, *D, B2, 13, 10, (uint8)(7, 6, 5, 4, 3, 2, 1, 0));
	
	STEP_BIG_IF(*A, *B, *C, *D, C2, 10, 25, (uint8)(4, 5, 6, 7, 0, 1, 2, 3));
	
	STEP_BIG_IF(*A, *B, *C, *D, D2, 25, 4, (uint8)(1, 0, 3, 2, 5, 4, 7, 6));
}

#endif
