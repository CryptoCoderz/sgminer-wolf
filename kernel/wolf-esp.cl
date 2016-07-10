/*
 * Espers' HMQ1725 kernel implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2014  phm
 * Copyright (c) 2016  pallas, CryptoCoderz Team
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
 * @author   phm <phm@inbox.com>
 * @author   pallas @ bitcointalk
 */

#ifndef HMQ1725_CL
#define HMQ1725_CL

#if __ENDIAN_LITTLE__
#define SPH_LITTLE_ENDIAN 1
#else
#define SPH_BIG_ENDIAN 1
#endif

#define SPH_UPTR sph_u64

typedef unsigned int sph_u32;
typedef int sph_s32;
#ifndef __OPENCL_VERSION__
typedef unsigned long long sph_u64;
typedef long long sph_s64;
#else
typedef unsigned long sph_u64;
typedef long sph_s64;
#endif

#define SPH_64 1
#define SPH_64_TRUE 1

#define SPH_C32(x)    ((sph_u32)(x ## U))
#define SPH_T32(x)    ((x) & SPH_C32(0xFFFFFFFF))
#define SPH_ROTL32(x, n)   SPH_T32(((x) << (n)) | ((x) >> (32 - (n))))
#define SPH_ROTR32(x, n)   SPH_ROTL32(x, (32 - (n)))

#define SPH_C64(x)    ((sph_u64)(x ## UL))
#define SPH_T64(x)    ((x) & SPH_C64(0xFFFFFFFFFFFFFFFF))
#define SPH_ROTL64(x, n)   SPH_T64(((x) << (n)) | ((x) >> (64 - (n))))
#define SPH_ROTR64(x, n)   SPH_ROTL64(x, (64 - (n)))

#define SPH_ECHO_64 1
#define SPH_KECCAK_64 1
#define SPH_JH_64 1
#define SPH_SIMD_NOCOPY 0
#define SPH_KECCAK_NOCOPY 0
#define SPH_COMPACT_BLAKE_64 0
#define SPH_LUFFA_PARALLEL 0
#define SPH_SMALL_FOOTPRINT_GROESTL 0
#define SPH_GROESTL_BIG_ENDIAN 0

#define SPH_CUBEHASH_UNROLL 0
#define SPH_KECCAK_UNROLL   0

// Added for HMQ1725
#ifndef SPH_HAMSI_EXPAND_BIG
  #define SPH_HAMSI_EXPAND_BIG 1
#endif

#pragma OPENCL EXTENSION cl_amd_media_ops : enable

#define VSWAP8(x)	(((x) >> 56) | (((x) >> 40) & 0x000000000000FF00UL) | (((x) >> 24) & 0x0000000000FF0000UL) \
          | (((x) >>  8) & 0x00000000FF000000UL) | (((x) <<  8) & 0x000000FF00000000UL) \
          | (((x) << 24) & 0x0000FF0000000000UL) | (((x) << 40) & 0x00FF000000000000UL) | (((x) << 56) & 0xFF00000000000000UL))

#define VSWAP4(x)	(((x) >> 24) | (((x) << 8) & 0x00FF0000) | (((x) >> 8) & 0x0000FF00) | (((x) << 24)))

ulong FAST_ROTL64_LO(const uint2 x, const uint y) { return(as_ulong(amd_bitalign(x, x.s10, 32 - y))); }
ulong FAST_ROTL64_HI(const uint2 x, const uint y) { return(as_ulong(amd_bitalign(x.s10, x, 32 - (y - 32)))); }

#define WOLF_JH_64BIT 1

#include "wolf-aes.cl"
#include "wolf-blake.cl"
#include "wolf-bmw.cl"
#include "wolf-groestl.cl"
#include "wolf-jh.cl"
#include "wolf-skein.cl"
// Added for HMQ1725
#include "wolf-luffa.cl"
#include "wolf-cubehash.cl"
#include "wolf-shavite.cl"
#include "wolf-simd.cl"
#include "wolf-echo.cl"
#include "wolf-hamsi.cl"
#include "wolf-fugue.cl"
#include "wolf-shabal.cl"
#include "wolf-whirlpool.cl"
#include "sha2big.cl"
#include "haval.cl"

#define SWAP4(x) as_uint(as_uchar4(x).wzyx)
#define SWAP8(x) as_ulong(as_uchar8(x).s76543210)

#if SPH_BIG_ENDIAN
  #define DEC64E(x) (x)
  #define DEC64BE(x) (*(const __global sph_u64 *) (x));
  // Added for HMQ172
  #define DEC32LE(x) SWAP4(*(const __global sph_u32 *) (x));
  #define DEC64LE(x) SWAP8(*(const __global sph_u64 *) (x));
#else
  #define DEC64E(x) SWAP8(x)
  #define DEC64BE(x) SWAP8(*(const __global sph_u64 *) (x));
  // Added for HMQ172
  #define DEC32LE(x) (*(const __global sph_u32 *) (x));
  #define DEC64LE(x) (*(const __global sph_u64 *) (x));
#endif

// Added for HMQ1725
#define SHL(x, n) ((x) << (n))
#define SHR(x, n) ((x) >> (n))

typedef union {
   unsigned char h1[64];
   uint h4[16];
   ulong h8[8];
} hash_t;


__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search(__global unsigned char* block, __global hash_t *hashes)
{   
    const uint idx = get_global_id(0) - get_global_offset(0);
	hashes += idx;
	
	// bmw
	ulong msg[16] = { 0 };
	
	#pragma unroll
	for(int i = 0; i < 19; ++i) ((uint *)msg)[i] = ((__global uint *)block)[i];
	
	((uint *)msg)[19] = get_global_id(0);
	
	msg[10] = 0x80UL;
	msg[15] = 0x280UL;
	
	#pragma unroll
	for(int i = 0; i < 2; ++i)
	{
		ulong h[16];
		for(int x = 0; x < 16; ++x) h[x] = ((i) ? BMW512_FINAL[x] : BMW512_IV[x]);
		BMW_Compression(msg, h);
	}
	
	#pragma unroll
	for(int i = 0; i < 8; ++i) hashes->h8[i] = msg[i + 8];
	
	mem_fence(CLK_GLOBAL_MEM_FENCE);
}


__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search1(__global hash_t *hashes, __global uint *Branch1Nonces, __global uint *Branch2Nonces, const ulong FoundIdx)
{
	ulong8 n, h, st;
	__local ulong LT0[256];
	
	hashes += get_global_id(0) - get_global_offset(0);
	
	for(int i = get_local_id(0); i < 256; i += get_local_size(0)) LT0[i] = WPL_T0_G[i];
	
	st = n = vload8(0, hashes->h8);
	h = 0;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	n = WhirlpoolRound(n, h, LT0);
	
	st ^= n;
	n = (ulong8)(0x80UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0x2000000000000UL) ^ st;
	h = st;
	
	n = WhirlpoolRound(n, h, LT0);
	
	st ^= n;
	st.s0 ^= 0x80UL;
	st.s7 ^= 0x2000000000000UL;
	
	vstore8(st, 0, hashes->h8);
	
	if(hashes->h4[0] & 24) Branch1Nonces[atomic_inc(Branch1Nonces + FoundIdx)] = get_global_id(0) - get_global_offset(0);
	else Branch2Nonces[atomic_inc(Branch2Nonces + FoundIdx)] = get_global_id(0) - get_global_offset(0);
	
	barrier(CLK_GLOBAL_MEM_FENCE);
}


	//bool dec = ((hash.h4[0] & 24) != 0);
__kernel void search2(__global hash_t *hashes, __global uint *BranchNonces)
{
	const uint idx = BranchNonces[get_global_id(0) - get_global_offset(0)];
	
	__local ulong T0[256], T1[256], T2[256], T3[256];
	__global hash_t *hash = hashes + idx;
	
	for(int i = get_local_id(0); i < 256; i += WORKSIZE)
	{
		const ulong tmp = T0_G[i];
		T0[i] = tmp;
		T1[i] = rotate(tmp, 8UL);
		T2[i] = rotate(tmp, 16UL);
		T3[i] = rotate(tmp, 24UL);
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	ulong M[16] = { 0 }, G[16];
	
	#pragma unroll
	for(int i = 0; i < 8; ++i) M[i] = hash->h8[i];
	
	M[8] = 0x80UL;
	M[15] = 0x0100000000000000UL;
	
	#pragma unroll
	for(int i = 0; i < 16; ++i) G[i] = M[i];
	
	G[15] ^= 0x0002000000000000UL;
	
	//#pragma unroll 2
	for(int i = 0; i < 14; ++i)
	{
		ulong H[16], H2[16];
		
		#pragma unroll
		for(int x = 0; x < 16; ++x)
		{
			H[x] = G[x] ^ PC64(x << 4, i);
			H2[x] = M[x] ^ QC64(x << 4, i);
		}
		
		GROESTL_RBTT(G[0], H, 0, 1, 2, 3, 4, 5, 6, 11);
		GROESTL_RBTT(G[1], H, 1, 2, 3, 4, 5, 6, 7, 12);
		GROESTL_RBTT(G[2], H, 2, 3, 4, 5, 6, 7, 8, 13);
		GROESTL_RBTT(G[3], H, 3, 4, 5, 6, 7, 8, 9, 14);
		GROESTL_RBTT(G[4], H, 4, 5, 6, 7, 8, 9, 10, 15);
		GROESTL_RBTT(G[5], H, 5, 6, 7, 8, 9, 10, 11, 0);
		GROESTL_RBTT(G[6], H, 6, 7, 8, 9, 10, 11, 12, 1);
		GROESTL_RBTT(G[7], H, 7, 8, 9, 10, 11, 12, 13, 2);
		GROESTL_RBTT(G[8], H, 8, 9, 10, 11, 12, 13, 14, 3);
		GROESTL_RBTT(G[9], H, 9, 10, 11, 12, 13, 14, 15, 4);
		GROESTL_RBTT(G[10], H, 10, 11, 12, 13, 14, 15, 0, 5);
		GROESTL_RBTT(G[11], H, 11, 12, 13, 14, 15, 0, 1, 6);
		GROESTL_RBTT(G[12], H, 12, 13, 14, 15, 0, 1, 2, 7);
		GROESTL_RBTT(G[13], H, 13, 14, 15, 0, 1, 2, 3, 8);
		GROESTL_RBTT(G[14], H, 14, 15, 0, 1, 2, 3, 4, 9);
		GROESTL_RBTT(G[15], H, 15, 0, 1, 2, 3, 4, 5, 10);
		
		GROESTL_RBTT(M[0], H2, 1, 3, 5, 11, 0, 2, 4, 6);
		GROESTL_RBTT(M[1], H2, 2, 4, 6, 12, 1, 3, 5, 7);
		GROESTL_RBTT(M[2], H2, 3, 5, 7, 13, 2, 4, 6, 8);
		GROESTL_RBTT(M[3], H2, 4, 6, 8, 14, 3, 5, 7, 9);
		GROESTL_RBTT(M[4], H2, 5, 7, 9, 15, 4, 6, 8, 10);
		GROESTL_RBTT(M[5], H2, 6, 8, 10, 0, 5, 7, 9, 11);
		GROESTL_RBTT(M[6], H2, 7, 9, 11, 1, 6, 8, 10, 12);
		GROESTL_RBTT(M[7], H2, 8, 10, 12, 2, 7, 9, 11, 13);
		GROESTL_RBTT(M[8], H2, 9, 11, 13, 3, 8, 10, 12, 14);
		GROESTL_RBTT(M[9], H2, 10, 12, 14, 4, 9, 11, 13, 15);
		GROESTL_RBTT(M[10], H2, 11, 13, 15, 5, 10, 12, 14, 0);
		GROESTL_RBTT(M[11], H2, 12, 14, 0, 6, 11, 13, 15, 1);
		GROESTL_RBTT(M[12], H2, 13, 15, 1, 7, 12, 14, 0, 2);
		GROESTL_RBTT(M[13], H2, 14, 0, 2, 8, 13, 15, 1, 3);
		GROESTL_RBTT(M[14], H2, 15, 1, 3, 9, 14, 0, 2, 4);
		GROESTL_RBTT(M[15], H2, 0, 2, 4, 10, 15, 1, 3, 5);
	}
	
	#pragma unroll
	for(int i = 0; i < 16; ++i) G[i] ^= M[i];
			
	G[15] ^= 0x0002000000000000UL;
	
	((ulong8 *)M)[0] = ((ulong8 *)G)[1];
	
	//#pragma unroll 2
	for(int i = 0; i < 14; ++i)
	{
		ulong H[16];
		
		#pragma unroll
		for(int x = 0; x < 16; ++x)
			H[x] = G[x] ^ PC64(x << 4, i); //G[x] ^ as_uint2(PC64((x << 4), i)); //(uint2)(G[x].s0 ^ ((x << 4) | i), G[x].s1);
			
		GROESTL_RBTT(G[0], H, 0, 1, 2, 3, 4, 5, 6, 11);
		GROESTL_RBTT(G[1], H, 1, 2, 3, 4, 5, 6, 7, 12);
		GROESTL_RBTT(G[2], H, 2, 3, 4, 5, 6, 7, 8, 13);
		GROESTL_RBTT(G[3], H, 3, 4, 5, 6, 7, 8, 9, 14);
		GROESTL_RBTT(G[4], H, 4, 5, 6, 7, 8, 9, 10, 15);
		GROESTL_RBTT(G[5], H, 5, 6, 7, 8, 9, 10, 11, 0);
		GROESTL_RBTT(G[6], H, 6, 7, 8, 9, 10, 11, 12, 1);
		GROESTL_RBTT(G[7], H, 7, 8, 9, 10, 11, 12, 13, 2);
		GROESTL_RBTT(G[8], H, 8, 9, 10, 11, 12, 13, 14, 3);
		GROESTL_RBTT(G[9], H, 9, 10, 11, 12, 13, 14, 15, 4);
		GROESTL_RBTT(G[10], H, 10, 11, 12, 13, 14, 15, 0, 5);
		GROESTL_RBTT(G[11], H, 11, 12, 13, 14, 15, 0, 1, 6);
		GROESTL_RBTT(G[12], H, 12, 13, 14, 15, 0, 1, 2, 7);
		GROESTL_RBTT(G[13], H, 13, 14, 15, 0, 1, 2, 3, 8);
		GROESTL_RBTT(G[14], H, 14, 15, 0, 1, 2, 3, 4, 9);
		GROESTL_RBTT(G[15], H, 15, 0, 1, 2, 3, 4, 5, 10);
	}
	
	vstore8((((ulong8 *)M)[0] ^ ((ulong8 *)G)[1]), 0, hash->h8);
	mem_fence(CLK_GLOBAL_MEM_FENCE);
}

__kernel void search3(__global hash_t *hashes, __global uint *BranchNonces)
{
	const uint idx = BranchNonces[get_global_id(0) - get_global_offset(0)];
	
	const ulong8 m = vload8(0, hashes[idx].h8);
	
	const ulong8 h = (ulong8)(	0x4903ADFF749C51CEUL, 0x0D95DE399746DF03UL, 0x8FD1934127C79BCEUL, 0x9A255629FF352CB1UL,
								0x5DB62599DF6CA7B0UL, 0xEABE394CA9D5C3F4UL, 0x991112C71A75B523UL, 0xAE18A40B660FCC33UL);
	
	const ulong t[3] = { 0x40UL, 0xF000000000000000UL, 0xF000000000000040UL }, t2[3] = { 0x08UL, 0xFF00000000000000UL, 0xFF00000000000008UL };
		
	ulong8 p = Skein512Block(m, h, 0xCAB2076D98173EC4UL, t);
	
	const ulong8 h2 = m ^ p;
	p = (ulong8)(0);
	ulong h8 = h2.s0 ^ h2.s1 ^ h2.s2 ^ h2.s3 ^ h2.s4 ^ h2.s5 ^ h2.s6 ^ h2.s7 ^ 0x1BD11BDAA9FC1A22UL;
	
	p = Skein512Block(p, h2, h8, t2);
	//p = VSWAP8(p);
	
	vstore8(p, 0, hashes[idx].h8);
	
	mem_fence(CLK_GLOBAL_MEM_FENCE);
}


__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search4(__global hash_t *hashes)
{
	const uint idx = get_global_id(0) - get_global_offset(0);
	__global hash_t *hash = hashes + idx;

	JH_CHUNK_TYPE evnhi = (JH_CHUNK_TYPE)(JH_BASE_TYPE_CAST(0x17AA003E964BD16FUL), JH_BASE_TYPE_CAST(0x1E806F53C1A01D89UL), JH_BASE_TYPE_CAST(0x694AE34105E66901UL), JH_BASE_TYPE_CAST(0x56F8B19DECF657CFUL));
	JH_CHUNK_TYPE evnlo = (JH_CHUNK_TYPE)(JH_BASE_TYPE_CAST(0x43D5157A052E6A63UL), JH_BASE_TYPE_CAST(0x806D2BEA6B05A92AUL), JH_BASE_TYPE_CAST(0x5AE66F2E8E8AB546UL), JH_BASE_TYPE_CAST(0x56B116577C8806A7UL));
	JH_CHUNK_TYPE oddhi = (JH_CHUNK_TYPE)(JH_BASE_TYPE_CAST(0x0BEF970C8D5E228AUL), JH_BASE_TYPE_CAST(0xA6BA7520DBCC8E58UL), JH_BASE_TYPE_CAST(0x243C84C1D0A74710UL), JH_BASE_TYPE_CAST(0xFB1785E6DFFCC2E3UL));
	JH_CHUNK_TYPE oddlo = (JH_CHUNK_TYPE)(JH_BASE_TYPE_CAST(0x61C3B3F2591234E9UL), JH_BASE_TYPE_CAST(0xF73BF8BA763A0FA9UL), JH_BASE_TYPE_CAST(0x99C15A2DB1716E3BUL), JH_BASE_TYPE_CAST(0x4BDD8CCC78465A54UL));
	
	#ifdef WOLF_JH_64BIT
	
	evnhi.s0 ^= JH_BASE_TYPE_CAST(hash->h8[0]);
	evnlo.s0 ^= JH_BASE_TYPE_CAST(hash->h8[1]);
	oddhi.s0 ^= JH_BASE_TYPE_CAST(hash->h8[2]);
	oddlo.s0 ^= JH_BASE_TYPE_CAST(hash->h8[3]);
	evnhi.s1 ^= JH_BASE_TYPE_CAST(hash->h8[4]);
	evnlo.s1 ^= JH_BASE_TYPE_CAST(hash->h8[5]);
	oddhi.s1 ^= JH_BASE_TYPE_CAST(hash->h8[6]);
	oddlo.s1 ^= JH_BASE_TYPE_CAST(hash->h8[7]);
	
	#else
	
	evnhi.lo.lo ^= JH_BASE_TYPE_CAST(hash->h8[0]);
	evnlo.lo.lo ^= JH_BASE_TYPE_CAST(hash->h8[1]);
	oddhi.lo.lo ^= JH_BASE_TYPE_CAST(hash->h8[2]);
	oddlo.lo.lo ^= JH_BASE_TYPE_CAST(hash->h8[3]);
	evnhi.lo.hi ^= JH_BASE_TYPE_CAST(hash->h8[4]);
	evnlo.lo.hi ^= JH_BASE_TYPE_CAST(hash->h8[5]);
	oddhi.lo.hi ^= JH_BASE_TYPE_CAST(hash->h8[6]);
	oddlo.lo.hi ^= JH_BASE_TYPE_CAST(hash->h8[7]);
	
	#endif
	
	for(bool flag = false;; flag++)
	{
		#pragma unroll
		for(int r = 0; r < 6; ++r)
		{
			evnhi = Sb(evnhi, Ceven_hi((r * 7) + 0));
			evnlo = Sb(evnlo, Ceven_lo((r * 7) + 0));
			oddhi = Sb(oddhi, Codd_hi((r * 7) + 0));
			oddlo = Sb(oddlo, Codd_lo((r * 7) + 0));
						
			Lb(&evnhi, &oddhi);
			Lb(&evnlo, &oddlo);
			
			JH_RND(&oddhi, &oddlo, 0);
			
			evnhi = Sb(evnhi, Ceven_hi((r * 7) + 1));
			evnlo = Sb(evnlo, Ceven_lo((r * 7) + 1));
			oddhi = Sb(oddhi, Codd_hi((r * 7) + 1));
			oddlo = Sb(oddlo, Codd_lo((r * 7) + 1));
			Lb(&evnhi, &oddhi);
			Lb(&evnlo, &oddlo);
			
			JH_RND(&oddhi, &oddlo, 1);
			
			evnhi = Sb(evnhi, Ceven_hi((r * 7) + 2));
			evnlo = Sb(evnlo, Ceven_lo((r * 7) + 2));
			oddhi = Sb(oddhi, Codd_hi((r * 7) + 2));
			oddlo = Sb(oddlo, Codd_lo((r * 7) + 2));
			Lb(&evnhi, &oddhi);
			Lb(&evnlo, &oddlo);
			
			JH_RND(&oddhi, &oddlo, 2);
			
			evnhi = Sb(evnhi, Ceven_hi((r * 7) + 3));
			evnlo = Sb(evnlo, Ceven_lo((r * 7) + 3));
			oddhi = Sb(oddhi, Codd_hi((r * 7) + 3));
			oddlo = Sb(oddlo, Codd_lo((r * 7) + 3));
			Lb(&evnhi, &oddhi);
			Lb(&evnlo, &oddlo);
			
			JH_RND(&oddhi, &oddlo, 3);
			
			evnhi = Sb(evnhi, Ceven_hi((r * 7) + 4));
			evnlo = Sb(evnlo, Ceven_lo((r * 7) + 4));
			oddhi = Sb(oddhi, Codd_hi((r * 7) + 4));
			oddlo = Sb(oddlo, Codd_lo((r * 7) + 4));
			Lb(&evnhi, &oddhi);
			Lb(&evnlo, &oddlo);
			
			JH_RND(&oddhi, &oddlo, 4);
			
			evnhi = Sb(evnhi, Ceven_hi((r * 7) + 5));
			evnlo = Sb(evnlo, Ceven_lo((r * 7) + 5));
			oddhi = Sb(oddhi, Codd_hi((r * 7) + 5));
			oddlo = Sb(oddlo, Codd_lo((r * 7) + 5));
			Lb(&evnhi, &oddhi);
			Lb(&evnlo, &oddlo);
			
			JH_RND(&oddhi, &oddlo, 5);
			
			evnhi = Sb(evnhi, Ceven_hi((r * 7) + 6));
			evnlo = Sb(evnlo, Ceven_lo((r * 7) + 6));
			oddhi = Sb(oddhi, Codd_hi((r * 7) + 6));
			oddlo = Sb(oddlo, Codd_lo((r * 7) + 6));
			Lb(&evnhi, &oddhi);
			Lb(&evnlo, &oddlo);
			
			JH_RND(&oddhi, &oddlo, 6);
		}
				
		if(flag) break;
		
		#ifdef WOLF_JH_64BIT
		
		evnhi.s2 ^= JH_BASE_TYPE_CAST(hash->h8[0]);
		evnlo.s2 ^= JH_BASE_TYPE_CAST(hash->h8[1]);
		oddhi.s2 ^= JH_BASE_TYPE_CAST(hash->h8[2]);
		oddlo.s2 ^= JH_BASE_TYPE_CAST(hash->h8[3]);
		evnhi.s3 ^= JH_BASE_TYPE_CAST(hash->h8[4]);
		evnlo.s3 ^= JH_BASE_TYPE_CAST(hash->h8[5]);
		oddhi.s3 ^= JH_BASE_TYPE_CAST(hash->h8[6]);
		oddlo.s3 ^= JH_BASE_TYPE_CAST(hash->h8[7]);
		
		evnhi.s0 ^= JH_BASE_TYPE_CAST(0x80UL);
		oddlo.s1 ^= JH_BASE_TYPE_CAST(0x0002000000000000UL);
		
		#else
			
		evnhi.hi.lo ^= JH_BASE_TYPE_CAST(hash->h8[0]);
		evnlo.hi.lo ^= JH_BASE_TYPE_CAST(hash->h8[1]);
		oddhi.hi.lo ^= JH_BASE_TYPE_CAST(hash->h8[2]);
		oddlo.hi.lo ^= JH_BASE_TYPE_CAST(hash->h8[3]);
		evnhi.hi.hi ^= JH_BASE_TYPE_CAST(hash->h8[4]);
		evnlo.hi.hi ^= JH_BASE_TYPE_CAST(hash->h8[5]);
		oddhi.hi.hi ^= JH_BASE_TYPE_CAST(hash->h8[6]);
		oddlo.hi.hi ^= JH_BASE_TYPE_CAST(hash->h8[7]);
		
		evnhi.lo.lo ^= JH_BASE_TYPE_CAST(0x80UL);
		oddlo.lo.hi ^= JH_BASE_TYPE_CAST(0x0002000000000000UL);
		
		#endif
	}
	
	#ifdef WOLF_JH_64BIT
	
	evnhi.s2 ^= JH_BASE_TYPE_CAST(0x80UL);
	oddlo.s3 ^= JH_BASE_TYPE_CAST(0x2000000000000UL);
	
	hash->h8[0] = as_ulong(evnhi.s2);
	hash->h8[1] = as_ulong(evnlo.s2);
	hash->h8[2] = as_ulong(oddhi.s2);
	hash->h8[3] = as_ulong(oddlo.s2);
	hash->h8[4] = as_ulong(evnhi.s3);
	hash->h8[5] = as_ulong(evnlo.s3);
	hash->h8[6] = as_ulong(oddhi.s3);
	hash->h8[7] = as_ulong(oddlo.s3);
	
	#else
	
	evnhi.hi.lo ^= JH_BASE_TYPE_CAST(0x80UL);
	oddlo.hi.hi ^= JH_BASE_TYPE_CAST(0x2000000000000UL);
	
	hash->h8[0] = as_ulong(evnhi.hi.lo);
	hash->h8[1] = as_ulong(evnlo.hi.lo);
	hash->h8[2] = as_ulong(oddhi.hi.lo);
	hash->h8[3] = as_ulong(oddlo.hi.lo);
	hash->h8[4] = as_ulong(evnhi.hi.hi);
	hash->h8[5] = as_ulong(evnlo.hi.hi);
	hash->h8[6] = as_ulong(oddhi.hi.hi);
	hash->h8[7] = as_ulong(oddlo.hi.hi);
	
	#endif
	
	mem_fence(CLK_GLOBAL_MEM_FENCE);
}

static const __constant ulong keccakf_rndc[24] = 
{
    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808aUL,
    0x8000000080008000UL, 0x000000000000808bUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL, 0x000000000000008aUL,
    0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000aUL,
    0x000000008000808bUL, 0x800000000000008bUL, 0x8000000000008089UL,
    0x8000000000008003UL, 0x8000000000008002UL, 0x8000000000000080UL, 
    0x000000000000800aUL, 0x800000008000000aUL, 0x8000000080008081UL,
    0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL
};

#define ROTL64(x, y)	rotate(x, y)

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search5(__global hash_t *hashes, __global uint *Branch3Nonces, __global uint *Branch4Nonces, const ulong FoundIdx)
{
	const uint idx = get_global_id(0) - get_global_offset(0);
		
	ulong st0, st1, st2, st3, st4, st5, st6, st7, st8, st9, st10, st11, st12, st13;
	ulong st14, st15, st16, st17, st18, st19, st20, st21, st22, st23, st24;
	
	st0 = hashes[idx].h8[0];
	st1 = hashes[idx].h8[1];
	st2 = hashes[idx].h8[2];
	st3 = hashes[idx].h8[3];
	st4 = hashes[idx].h8[4];
	st5 = hashes[idx].h8[5];
	st6 = hashes[idx].h8[6];
	st7 = hashes[idx].h8[7];
	st8 = as_ulong((uint2)(1, 0x80000000U));
	st9 = st10 = st11 = st12 = st13 = st14 = st15 = st16 = 0;
	st17 = st18 = st19 = st20 = st21 = st22 = st23 = st24 = 0;
	
	for(int i = 0; i < 24; ++i)
	{
		ulong u[5], tmp1, tmp2;
		u[0] = st0 ^ st5 ^ st10 ^ st15 ^ st20 ^ ROTL64(st2 ^ st7 ^ st12 ^ st17 ^ st22, 1UL);
		u[1] = st1 ^ st6 ^ st11 ^ st16 ^ st21 ^ ROTL64(st3 ^ st8 ^ st13 ^ st18 ^ st23, 1UL);
		u[2] = st2 ^ st7 ^ st12 ^ st17 ^ st22 ^ ROTL64(st4 ^ st9 ^ st14 ^ st19 ^ st24, 1UL);
		u[3] = st3 ^ st8 ^ st13 ^ st18 ^ st23 ^ ROTL64(st0 ^ st5 ^ st10 ^ st15 ^ st20, 1UL);
		u[4] = st4 ^ st9 ^ st14 ^ st19 ^ st24 ^ ROTL64(st1 ^ st6 ^ st11 ^ st16 ^ st21, 1UL);
		tmp1 = st1 ^ u[0];
		
		st0 ^= u[4];
		
		st1 = rotate(st6 ^ u[0], 44UL);
		st6 = rotate(st9 ^ u[3], 20UL);
		st9 = rotate(st22 ^ u[1], 61UL);
		st22 = rotate(st14 ^ u[3], 39UL);
		st14 = rotate(st20 ^ u[4], 18UL);
		st20 = rotate(st2 ^ u[1], 62UL);
		st2 = rotate(st12 ^ u[1], 43UL);
		st12 = rotate(st13 ^ u[2], 25UL);
		st13 = rotate(st19 ^ u[3],  8UL);
		st19 = rotate(st23 ^ u[2], 56UL);
		st23 = rotate(st15 ^ u[4], 41UL);
		st15 = rotate(st4 ^ u[3], 27UL);
		st4 = rotate(st24 ^ u[3], 14UL);
		st24 = rotate(st21 ^ u[0],  2UL);
		st21 = rotate(st8 ^ u[2], 55UL);
		st8 = rotate(st16 ^ u[0], 45UL);
		st16 = rotate(st5 ^ u[4], 36UL);
		st5 = rotate(st3 ^ u[2], 28UL);
		st3 = rotate(st18 ^ u[2], 21UL);
		st18 = rotate(st17 ^ u[1], 15UL);
		st17 = rotate(st11 ^ u[0], 10UL);
		st11 = rotate(st7 ^ u[1],  6UL);
		st7 = rotate(st10 ^ u[4],  3UL);
		st10 = rotate(tmp1, 1UL);
		
		tmp1 = st0; tmp2 = st1; st0 = bitselect(st0 ^ st2, st0, st1); st1 = bitselect(st1 ^ st3, st1, st2); st2 = bitselect(st2 ^ st4, st2, st3); st3 = bitselect(st3 ^ tmp1, st3, st4); st4 = bitselect(st4 ^ tmp2, st4, tmp1);
		tmp1 = st5; tmp2 = st6; st5 = bitselect(st5 ^ st7, st5, st6); st6 = bitselect(st6 ^ st8, st6, st7); st7 = bitselect(st7 ^ st9, st7, st8); st8 = bitselect(st8 ^ tmp1, st8, st9); st9 = bitselect(st9 ^ tmp2, st9, tmp1);
		tmp1 = st10; tmp2 = st11; st10 = bitselect(st10 ^ st12, st10, st11); st11 = bitselect(st11 ^ st13, st11, st12); st12 = bitselect(st12 ^ st14, st12, st13); st13 = bitselect(st13 ^ tmp1, st13, st14); st14 = bitselect(st14 ^ tmp2, st14, tmp1);
		tmp1 = st15; tmp2 = st16; st15 = bitselect(st15 ^ st17, st15, st16); st16 = bitselect(st16 ^ st18, st16, st17); st17 = bitselect(st17 ^ st19, st17, st18); st18 = bitselect(st18 ^ tmp1, st18, st19); st19 = bitselect(st19 ^ tmp2, st19, tmp1);
		tmp1 = st20; tmp2 = st21; st20 = bitselect(st20 ^ st22, st20, st21); st21 = bitselect(st21 ^ st23, st21, st22); st22 = bitselect(st22 ^ st24, st22, st23); st23 = bitselect(st23 ^ tmp1, st23, st24); st24 = bitselect(st24 ^ tmp2, st24, tmp1);
		st0 ^= keccakf_rndc[i];
	}
	
	hashes[idx].h8[0] = st0;
	hashes[idx].h8[1] = st1;
	hashes[idx].h8[2] = st2;
	hashes[idx].h8[3] = st3;
	hashes[idx].h8[4] = st4;
	hashes[idx].h8[5] = st5;
	hashes[idx].h8[6] = st6;
	hashes[idx].h8[7] = st7;

	if(hashes[idx].h4[0] & 24) Branch3Nonces[atomic_inc(Branch3Nonces + FoundIdx)] = get_global_id(0) - get_global_offset(0);
	else Branch4Nonces[atomic_inc(Branch4Nonces + FoundIdx)] = get_global_id(0) - get_global_offset(0);
	
	mem_fence(CLK_GLOBAL_MEM_FENCE);
}


  //dec = ((hash.h4[0] & 24) != 0);
  
__kernel void search6(__global hash_t *hashes, __global uint *BranchNonces)
{
	ulong16 V;
	ulong M[16];
	
	hashes += BranchNonces[get_global_id(0) - get_global_offset(0)];
	
	((ulong8 *)M)[0] = VSWAP8(vload8(0, hashes->h8));
	((ulong8 *)M)[1] = (ulong8)(0x8000000000000000UL, 0UL, 0UL, 0UL, 0UL, 1UL, 0UL, 0x200UL);
	
	V.lo = vload8(0, BLAKE512_IV);
	V.hi = vload8(0, blake_cb);
	
	V.scd ^= (ulong2)(0x200UL, 0x200UL);
	
	for(bool flag = false; ; flag = true)
	{
		BLAKE_RND(0);
		BLAKE_RND(1);
		BLAKE_RND(2);
		BLAKE_RND(3);
		BLAKE_RND(4);
		BLAKE_RND(5);
		if(flag) break;
		BLAKE_RND(6);
		BLAKE_RND(7);
		BLAKE_RND(8);
		BLAKE_RND(9);
	}
	
	vstore8(VSWAP8(((__constant ulong8 *)BLAKE512_IV)[0] ^ V.lo ^ V.hi), 0, hashes->h8);
	
	mem_fence(CLK_GLOBAL_MEM_FENCE);
}

__kernel void search7(__global hash_t *hashes, __global uint *BranchNonces)
{
	ulong msg[16] = { 0 };
	hashes += BranchNonces[get_global_id(0) - get_global_offset(0)];	

	#pragma unroll
	for(int i = 0; i < 8; ++i) msg[i] = hashes->h8[i];

	msg[8] = 0x80UL;
	msg[15] = 512UL;
	
	#pragma unroll
	for(int i = 0; i < 2; ++i)
	{
		ulong h[16];
		for(int x = 0; x < 16; ++x) h[x] = ((i) ? BMW512_FINAL[x] : BMW512_IV[x]);
		BMW_Compression(msg, h);
	}
	
	#pragma unroll
	for(int i = 0; i < 8; ++i) hashes->h8[i] = msg[i + 8];
	
	mem_fence(CLK_GLOBAL_MEM_FENCE);
}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search8(__global hash_t *hashes)
{
	hashes += get_global_id(0) - get_global_offset(0);
	
	uint8 V[5] =
	{
		(uint8)(0x6D251E69U, 0x44B051E0U, 0x4EAA6FB4U, 0xDBF78465U, 0x6E292011U, 0x90152DF4U, 0xEE058139U, 0xDEF610BBU),
		(uint8)(0xC3B44B95U, 0xD9D2F256U, 0x70EEE9A0U, 0xDE099FA3U, 0x5D9B0557U, 0x8FC944B3U, 0xCF1CCF0EU, 0x746CD581U),
		(uint8)(0xF7EFC89DU, 0x5DBA5781U, 0x04016CE5U, 0xAD659C05U, 0x0306194FU, 0x666D1836U, 0x24AA230AU, 0x8B264AE7U),
		(uint8)(0x858075D5U, 0x36D79CCEU, 0xE571F7D7U, 0x204B1F67U, 0x35870C6AU, 0x57E9E923U, 0x14BCB808U, 0x7CDE72CEU),
		(uint8)(0x6C68E9BEU, 0x5EC41E22U, 0xC825B7C7U, 0xAFFB4363U, 0xF5DF3999U, 0x0FC688F1U, 0xB07224CCU, 0x03E86CEAU)
	};
	
	#pragma unroll
	for(int i = 0; i < 8; ++i) hashes->h8[i] = SWAP8(hashes->h8[i]);
	
	#pragma unroll
    for(int i = 0; i < 6; ++i)
    {
		uint8 M;
		switch(i)
		{
			case 0:
			case 1:
				M = shuffle(vload8(i, hashes->h4), (uint8)(1, 0, 3, 2, 5, 4, 7, 6));
				break;
			case 2:
			case 3:
				M = (uint8)(0);
				M.s0 = (i == 2) ? 0x80000000 : 0;
				break;
			case 4:
			case 5:
				vstore8(shuffle(V[0] ^ V[1] ^ V[2] ^ V[3] ^ V[4], (uint8)(1, 0, 3, 2, 5, 4, 7, 6)), i & 1, hashes->h4);
		}
		if(i == 5) break;
		
		MessageInj(V, M);
		LuffaPerm(V);
    }
	
	#pragma unroll
	for(int i = 0; i < 8; ++i) hashes->h8[i] = SWAP8(hashes->h8[i]);
	
	mem_fence(CLK_GLOBAL_MEM_FENCE);
}


__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search9(__global hash_t *hashes, __global uint *Branch5Nonces, __global uint *Branch6Nonces, const ulong FoundIdx)
{
	const uint idx = get_global_id(0) - get_global_offset(0);
	
	// cubehash.h1
	
	uint state[32] = { 	0x2AEA2A61U, 0x50F494D4U, 0x2D538B8BU, 0x4167D83EU, 0x3FEE2313U, 0xC701CF8CU,
						0xCC39968EU, 0x50AC5695U, 0x4D42C787U, 0xA647A8B3U, 0x97CF0BEFU, 0x825B4537U,
						0xEEF864D2U, 0xF22090C4U, 0xD0E5CD33U, 0xA23911AEU, 0xFCD398D9U, 0x148FE485U,
						0x1B017BEFU, 0xB6444532U, 0x6A536159U, 0x2FF5781CU, 0x91FA7934U, 0x0DBADEA9U,
						0xD65C8A2BU, 0xA5A70E75U, 0xB1C62456U, 0xBC796576U, 0x1921C8F7U, 0xE7989AF1U, 
						0x7795D246U, 0xD43E3B44U };
	
	((ulong4 *)state)[0] ^= vload4(0, hashes[idx].h8);
	ulong4 xor = vload4(1, hashes[idx].h8);
	
	#pragma unroll 2
	for(int i = 0; i < 14; ++i)
	{
		#pragma unroll 4
		for(int x = 0; x < 8; ++x)
		{
			CubeHashEvenRound(state);
			CubeHashOddRound(state);
		}
		
		if(i == 12)
		{
			vstore8(((ulong8 *)state)[0], 0, hashes[idx].h8);
			break;
		}
		if(!i) ((ulong4 *)state)[0] ^= xor;
		state[0] ^= (i == 1) ? 0x80 : 0;
		state[31] ^= (i == 2) ? 1 : 0;
	}
	
	mem_fence(CLK_GLOBAL_MEM_FENCE);
  
	if(hashes[idx].h4[0] & 24) Branch5Nonces[atomic_inc(Branch5Nonces + FoundIdx)] = get_global_id(0) - get_global_offset(0);
	else Branch6Nonces[atomic_inc(Branch6Nonces + FoundIdx)] = get_global_id(0) - get_global_offset(0);
}


  //dec = ((hash.h4[0] & 24) != 0);
__kernel void search10(__global hash_t *hashes, __global uint *BranchNonces)
{
	const uint idx = BranchNonces[get_global_id(0) - get_global_offset(0)];
		
	ulong st0, st1, st2, st3, st4, st5, st6, st7, st8, st9, st10, st11, st12, st13;
	ulong st14, st15, st16, st17, st18, st19, st20, st21, st22, st23, st24;
	
	st0 = hashes[idx].h8[0];
	st1 = hashes[idx].h8[1];
	st2 = hashes[idx].h8[2];
	st3 = hashes[idx].h8[3];
	st4 = hashes[idx].h8[4];
	st5 = hashes[idx].h8[5];
	st6 = hashes[idx].h8[6];
	st7 = hashes[idx].h8[7];
	st8 = as_ulong((uint2)(1, 0x80000000U));
	st9 = st10 = st11 = st12 = st13 = st14 = st15 = st16 = 0;
	st17 = st18 = st19 = st20 = st21 = st22 = st23 = st24 = 0;
	
	for(int i = 0; i < 24; ++i)
	{
		ulong u[5], tmp1, tmp2;
		u[0] = st0 ^ st5 ^ st10 ^ st15 ^ st20 ^ ROTL64(st2 ^ st7 ^ st12 ^ st17 ^ st22, 1UL);
		u[1] = st1 ^ st6 ^ st11 ^ st16 ^ st21 ^ ROTL64(st3 ^ st8 ^ st13 ^ st18 ^ st23, 1UL);
		u[2] = st2 ^ st7 ^ st12 ^ st17 ^ st22 ^ ROTL64(st4 ^ st9 ^ st14 ^ st19 ^ st24, 1UL);
		u[3] = st3 ^ st8 ^ st13 ^ st18 ^ st23 ^ ROTL64(st0 ^ st5 ^ st10 ^ st15 ^ st20, 1UL);
		u[4] = st4 ^ st9 ^ st14 ^ st19 ^ st24 ^ ROTL64(st1 ^ st6 ^ st11 ^ st16 ^ st21, 1UL);
		tmp1 = st1 ^ u[0];
		
		st0 ^= u[4];
		
		st1 = rotate(st6 ^ u[0], 44UL);
		st6 = rotate(st9 ^ u[3], 20UL);
		st9 = rotate(st22 ^ u[1], 61UL);
		st22 = rotate(st14 ^ u[3], 39UL);
		st14 = rotate(st20 ^ u[4], 18UL);
		st20 = rotate(st2 ^ u[1], 62UL);
		st2 = rotate(st12 ^ u[1], 43UL);
		st12 = rotate(st13 ^ u[2], 25UL);
		st13 = rotate(st19 ^ u[3],  8UL);
		st19 = rotate(st23 ^ u[2], 56UL);
		st23 = rotate(st15 ^ u[4], 41UL);
		st15 = rotate(st4 ^ u[3], 27UL);
		st4 = rotate(st24 ^ u[3], 14UL);
		st24 = rotate(st21 ^ u[0],  2UL);
		st21 = rotate(st8 ^ u[2], 55UL);
		st8 = rotate(st16 ^ u[0], 45UL);
		st16 = rotate(st5 ^ u[4], 36UL);
		st5 = rotate(st3 ^ u[2], 28UL);
		st3 = rotate(st18 ^ u[2], 21UL);
		st18 = rotate(st17 ^ u[1], 15UL);
		st17 = rotate(st11 ^ u[0], 10UL);
		st11 = rotate(st7 ^ u[1],  6UL);
		st7 = rotate(st10 ^ u[4],  3UL);
		st10 = rotate(tmp1, 1UL);
		
		tmp1 = st0; tmp2 = st1; st0 = bitselect(st0 ^ st2, st0, st1); st1 = bitselect(st1 ^ st3, st1, st2); st2 = bitselect(st2 ^ st4, st2, st3); st3 = bitselect(st3 ^ tmp1, st3, st4); st4 = bitselect(st4 ^ tmp2, st4, tmp1);
		tmp1 = st5; tmp2 = st6; st5 = bitselect(st5 ^ st7, st5, st6); st6 = bitselect(st6 ^ st8, st6, st7); st7 = bitselect(st7 ^ st9, st7, st8); st8 = bitselect(st8 ^ tmp1, st8, st9); st9 = bitselect(st9 ^ tmp2, st9, tmp1);
		tmp1 = st10; tmp2 = st11; st10 = bitselect(st10 ^ st12, st10, st11); st11 = bitselect(st11 ^ st13, st11, st12); st12 = bitselect(st12 ^ st14, st12, st13); st13 = bitselect(st13 ^ tmp1, st13, st14); st14 = bitselect(st14 ^ tmp2, st14, tmp1);
		tmp1 = st15; tmp2 = st16; st15 = bitselect(st15 ^ st17, st15, st16); st16 = bitselect(st16 ^ st18, st16, st17); st17 = bitselect(st17 ^ st19, st17, st18); st18 = bitselect(st18 ^ tmp1, st18, st19); st19 = bitselect(st19 ^ tmp2, st19, tmp1);
		tmp1 = st20; tmp2 = st21; st20 = bitselect(st20 ^ st22, st20, st21); st21 = bitselect(st21 ^ st23, st21, st22); st22 = bitselect(st22 ^ st24, st22, st23); st23 = bitselect(st23 ^ tmp1, st23, st24); st24 = bitselect(st24 ^ tmp2, st24, tmp1);
		st0 ^= keccakf_rndc[i];
	}
	
	hashes[idx].h8[0] = st0;
	hashes[idx].h8[1] = st1;
	hashes[idx].h8[2] = st2;
	hashes[idx].h8[3] = st3;
	hashes[idx].h8[4] = st4;
	hashes[idx].h8[5] = st5;
	hashes[idx].h8[6] = st6;
	hashes[idx].h8[7] = st7;
	
	mem_fence(CLK_GLOBAL_MEM_FENCE);
}

__kernel void search11(__global hash_t *hashes, __global uint *BranchNonces)
{
	const uint idx = BranchNonces[get_global_id(0) - get_global_offset(0)];
	__global hash_t *hash = hashes + idx;

	JH_CHUNK_TYPE evnhi = (JH_CHUNK_TYPE)(JH_BASE_TYPE_CAST(0x17AA003E964BD16FUL), JH_BASE_TYPE_CAST(0x1E806F53C1A01D89UL), JH_BASE_TYPE_CAST(0x694AE34105E66901UL), JH_BASE_TYPE_CAST(0x56F8B19DECF657CFUL));
	JH_CHUNK_TYPE evnlo = (JH_CHUNK_TYPE)(JH_BASE_TYPE_CAST(0x43D5157A052E6A63UL), JH_BASE_TYPE_CAST(0x806D2BEA6B05A92AUL), JH_BASE_TYPE_CAST(0x5AE66F2E8E8AB546UL), JH_BASE_TYPE_CAST(0x56B116577C8806A7UL));
	JH_CHUNK_TYPE oddhi = (JH_CHUNK_TYPE)(JH_BASE_TYPE_CAST(0x0BEF970C8D5E228AUL), JH_BASE_TYPE_CAST(0xA6BA7520DBCC8E58UL), JH_BASE_TYPE_CAST(0x243C84C1D0A74710UL), JH_BASE_TYPE_CAST(0xFB1785E6DFFCC2E3UL));
	JH_CHUNK_TYPE oddlo = (JH_CHUNK_TYPE)(JH_BASE_TYPE_CAST(0x61C3B3F2591234E9UL), JH_BASE_TYPE_CAST(0xF73BF8BA763A0FA9UL), JH_BASE_TYPE_CAST(0x99C15A2DB1716E3BUL), JH_BASE_TYPE_CAST(0x4BDD8CCC78465A54UL));
	
	#ifdef WOLF_JH_64BIT
	
	evnhi.s0 ^= JH_BASE_TYPE_CAST(hash->h8[0]);
	evnlo.s0 ^= JH_BASE_TYPE_CAST(hash->h8[1]);
	oddhi.s0 ^= JH_BASE_TYPE_CAST(hash->h8[2]);
	oddlo.s0 ^= JH_BASE_TYPE_CAST(hash->h8[3]);
	evnhi.s1 ^= JH_BASE_TYPE_CAST(hash->h8[4]);
	evnlo.s1 ^= JH_BASE_TYPE_CAST(hash->h8[5]);
	oddhi.s1 ^= JH_BASE_TYPE_CAST(hash->h8[6]);
	oddlo.s1 ^= JH_BASE_TYPE_CAST(hash->h8[7]);
	
	#else
	
	evnhi.lo.lo ^= JH_BASE_TYPE_CAST(hash->h8[0]);
	evnlo.lo.lo ^= JH_BASE_TYPE_CAST(hash->h8[1]);
	oddhi.lo.lo ^= JH_BASE_TYPE_CAST(hash->h8[2]);
	oddlo.lo.lo ^= JH_BASE_TYPE_CAST(hash->h8[3]);
	evnhi.lo.hi ^= JH_BASE_TYPE_CAST(hash->h8[4]);
	evnlo.lo.hi ^= JH_BASE_TYPE_CAST(hash->h8[5]);
	oddhi.lo.hi ^= JH_BASE_TYPE_CAST(hash->h8[6]);
	oddlo.lo.hi ^= JH_BASE_TYPE_CAST(hash->h8[7]);
	
	#endif
	
	for(bool flag = false;; flag++)
	{
		#pragma unroll
		for(int r = 0; r < 6; ++r)
		{
			evnhi = Sb(evnhi, Ceven_hi((r * 7) + 0));
			evnlo = Sb(evnlo, Ceven_lo((r * 7) + 0));
			oddhi = Sb(oddhi, Codd_hi((r * 7) + 0));
			oddlo = Sb(oddlo, Codd_lo((r * 7) + 0));
						
			Lb(&evnhi, &oddhi);
			Lb(&evnlo, &oddlo);
			
			JH_RND(&oddhi, &oddlo, 0);
			
			evnhi = Sb(evnhi, Ceven_hi((r * 7) + 1));
			evnlo = Sb(evnlo, Ceven_lo((r * 7) + 1));
			oddhi = Sb(oddhi, Codd_hi((r * 7) + 1));
			oddlo = Sb(oddlo, Codd_lo((r * 7) + 1));
			Lb(&evnhi, &oddhi);
			Lb(&evnlo, &oddlo);
			
			JH_RND(&oddhi, &oddlo, 1);
			
			evnhi = Sb(evnhi, Ceven_hi((r * 7) + 2));
			evnlo = Sb(evnlo, Ceven_lo((r * 7) + 2));
			oddhi = Sb(oddhi, Codd_hi((r * 7) + 2));
			oddlo = Sb(oddlo, Codd_lo((r * 7) + 2));
			Lb(&evnhi, &oddhi);
			Lb(&evnlo, &oddlo);
			
			JH_RND(&oddhi, &oddlo, 2);
			
			evnhi = Sb(evnhi, Ceven_hi((r * 7) + 3));
			evnlo = Sb(evnlo, Ceven_lo((r * 7) + 3));
			oddhi = Sb(oddhi, Codd_hi((r * 7) + 3));
			oddlo = Sb(oddlo, Codd_lo((r * 7) + 3));
			Lb(&evnhi, &oddhi);
			Lb(&evnlo, &oddlo);
			
			JH_RND(&oddhi, &oddlo, 3);
			
			evnhi = Sb(evnhi, Ceven_hi((r * 7) + 4));
			evnlo = Sb(evnlo, Ceven_lo((r * 7) + 4));
			oddhi = Sb(oddhi, Codd_hi((r * 7) + 4));
			oddlo = Sb(oddlo, Codd_lo((r * 7) + 4));
			Lb(&evnhi, &oddhi);
			Lb(&evnlo, &oddlo);
			
			JH_RND(&oddhi, &oddlo, 4);
			
			evnhi = Sb(evnhi, Ceven_hi((r * 7) + 5));
			evnlo = Sb(evnlo, Ceven_lo((r * 7) + 5));
			oddhi = Sb(oddhi, Codd_hi((r * 7) + 5));
			oddlo = Sb(oddlo, Codd_lo((r * 7) + 5));
			Lb(&evnhi, &oddhi);
			Lb(&evnlo, &oddlo);
			
			JH_RND(&oddhi, &oddlo, 5);
			
			evnhi = Sb(evnhi, Ceven_hi((r * 7) + 6));
			evnlo = Sb(evnlo, Ceven_lo((r * 7) + 6));
			oddhi = Sb(oddhi, Codd_hi((r * 7) + 6));
			oddlo = Sb(oddlo, Codd_lo((r * 7) + 6));
			Lb(&evnhi, &oddhi);
			Lb(&evnlo, &oddlo);
			
			JH_RND(&oddhi, &oddlo, 6);
		}
				
		if(flag) break;
		
		#ifdef WOLF_JH_64BIT
		
		evnhi.s2 ^= JH_BASE_TYPE_CAST(hash->h8[0]);
		evnlo.s2 ^= JH_BASE_TYPE_CAST(hash->h8[1]);
		oddhi.s2 ^= JH_BASE_TYPE_CAST(hash->h8[2]);
		oddlo.s2 ^= JH_BASE_TYPE_CAST(hash->h8[3]);
		evnhi.s3 ^= JH_BASE_TYPE_CAST(hash->h8[4]);
		evnlo.s3 ^= JH_BASE_TYPE_CAST(hash->h8[5]);
		oddhi.s3 ^= JH_BASE_TYPE_CAST(hash->h8[6]);
		oddlo.s3 ^= JH_BASE_TYPE_CAST(hash->h8[7]);
		
		evnhi.s0 ^= JH_BASE_TYPE_CAST(0x80UL);
		oddlo.s1 ^= JH_BASE_TYPE_CAST(0x0002000000000000UL);
		
		#else
			
		evnhi.hi.lo ^= JH_BASE_TYPE_CAST(hash->h8[0]);
		evnlo.hi.lo ^= JH_BASE_TYPE_CAST(hash->h8[1]);
		oddhi.hi.lo ^= JH_BASE_TYPE_CAST(hash->h8[2]);
		oddlo.hi.lo ^= JH_BASE_TYPE_CAST(hash->h8[3]);
		evnhi.hi.hi ^= JH_BASE_TYPE_CAST(hash->h8[4]);
		evnlo.hi.hi ^= JH_BASE_TYPE_CAST(hash->h8[5]);
		oddhi.hi.hi ^= JH_BASE_TYPE_CAST(hash->h8[6]);
		oddlo.hi.hi ^= JH_BASE_TYPE_CAST(hash->h8[7]);
		
		evnhi.lo.lo ^= JH_BASE_TYPE_CAST(0x80UL);
		oddlo.lo.hi ^= JH_BASE_TYPE_CAST(0x0002000000000000UL);
		
		#endif
	}
	
	#ifdef WOLF_JH_64BIT
	
	evnhi.s2 ^= JH_BASE_TYPE_CAST(0x80UL);
	oddlo.s3 ^= JH_BASE_TYPE_CAST(0x2000000000000UL);
	
	hash->h8[0] = as_ulong(evnhi.s2);
	hash->h8[1] = as_ulong(evnlo.s2);
	hash->h8[2] = as_ulong(oddhi.s2);
	hash->h8[3] = as_ulong(oddlo.s2);
	hash->h8[4] = as_ulong(evnhi.s3);
	hash->h8[5] = as_ulong(evnlo.s3);
	hash->h8[6] = as_ulong(oddhi.s3);
	hash->h8[7] = as_ulong(oddlo.s3);
	
	#else
	
	evnhi.hi.lo ^= JH_BASE_TYPE_CAST(0x80UL);
	oddlo.hi.hi ^= JH_BASE_TYPE_CAST(0x2000000000000UL);
	
	hash->h8[0] = as_ulong(evnhi.hi.lo);
	hash->h8[1] = as_ulong(evnlo.hi.lo);
	hash->h8[2] = as_ulong(oddhi.hi.lo);
	hash->h8[3] = as_ulong(oddlo.hi.lo);
	hash->h8[4] = as_ulong(evnhi.hi.hi);
	hash->h8[5] = as_ulong(evnlo.hi.hi);
	hash->h8[6] = as_ulong(oddhi.hi.hi);
	hash->h8[7] = as_ulong(oddlo.hi.hi);
	
	#endif
	
	mem_fence(CLK_GLOBAL_MEM_FENCE);
}


__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search12(__global hash_t *hashes)
{
	__local uint AES0[256], AES1[256], AES2[256], AES3[256];
	hashes += get_global_id(0) - get_global_offset(0);
	
	const int step = get_local_size(0);
	
	for(int i = get_local_id(0); i < 256; i += step)
	{
		const uint tmp = AES0_C[i];
		AES0[i] = tmp;
		AES1[i] = rotate(tmp, 8U);
		AES2[i] = rotate(tmp, 16U);
		AES3[i] = rotate(tmp, 24U);
	}
	
	const uint4 h[4] = {(uint4)(0x72FCCDD8, 0x79CA4727, 0x128A077B, 0x40D55AEC), (uint4)(0xD1901A06, 0x430AE307, 0xB29F5CD1, 0xDF07FBFC), \
						(uint4)(0x8E45D73D, 0x681AB538, 0xBDE86578, 0xDD577E47), (uint4)(0xE275EADE, 0x502D9FCD, 0xB9357178, 0x022A4B9A) };
	
	uint4 rk[8] = { (uint4)(0) }, p[4] = { h[0], h[1], h[2], h[3] };
	
	((uint16 *)rk)[0] = vload16(0, hashes->h4);
	rk[4].s0 = 0x80;
	rk[6].s3 = 0x2000000;
	rk[7].s3 = 0x2000000;
	mem_fence(CLK_LOCAL_MEM_FENCE);
	
	#pragma unroll 1
	for(int r = 0; r < 3; ++r)
	{
		if(r == 0)
		{
			p[0] = Shavite_AES_4Round(AES0, AES1, AES2, AES3, p[1] ^ rk[0], &(rk[1]), p[0]);
			p[2] = Shavite_AES_4Round(AES0, AES1, AES2, AES3, p[3] ^ rk[4], &(rk[5]), p[2]);
		}
		#pragma unroll 1
		for(int y = 0; y < 2; ++y)
		{
			rk[0] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[0], rk[7]);
			rk[0].s03 ^= ((!y && !r) ? (uint2)(0x200, 0xFFFFFFFF) : (uint2)(0));
			uint4 x = rk[0] ^ (y != 1 ? p[0] : p[2]);
			rk[1] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[1], rk[0]);
			rk[1].s3 ^= (!y && r == 1 ? 0xFFFFFDFFU : 0);	// ~(0x200)
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[1]);
			rk[2] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[2], rk[1]);
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[2]);
			rk[3] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[3], rk[2]);
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[3]);
			if(y != 1) p[3] = AES_Round(AES0, AES1, AES2, AES3, x, p[3]);
			else p[1] = AES_Round(AES0, AES1, AES2, AES3, x, p[1]);
			
			rk[4] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[4], rk[3]);
			x = rk[4] ^ (y != 1 ? p[2] : p[0]);
			rk[5] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[5], rk[4]);
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[5]);
			rk[6] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[6], rk[5]);
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[6]);
			rk[7] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[7], rk[6]);
			rk[7].s23 ^= ((!y && r == 2) ? (uint2)(0x200, 0xFFFFFFFF) : (uint2)(0));
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[7]);
			if(y != 1) p[1] = AES_Round(AES0, AES1, AES2, AES3, x, p[1]);
			else p[3] = AES_Round(AES0, AES1, AES2, AES3, x, p[3]);
						
			rk[0] ^= shuffle2(rk[6], rk[7], (uint4)(1, 2, 3, 4));
			x = rk[0] ^ (!y ? p[3] : p[1]);
			rk[1] ^= shuffle2(rk[7], rk[0], (uint4)(1, 2, 3, 4));
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[1]);
			rk[2] ^= shuffle2(rk[0], rk[1], (uint4)(1, 2, 3, 4));
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[2]);
			rk[3] ^= shuffle2(rk[1], rk[2], (uint4)(1, 2, 3, 4));
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[3]);
			if(!y) p[2] = AES_Round(AES0, AES1, AES2, AES3, x, p[2]);
			else p[0] = AES_Round(AES0, AES1, AES2, AES3, x, p[0]);
					
			rk[4] ^= shuffle2(rk[2], rk[3], (uint4)(1, 2, 3, 4));
			x = rk[4] ^ (!y ? p[1] : p[3]);
			rk[5] ^= shuffle2(rk[3], rk[4], (uint4)(1, 2, 3, 4));
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[5]);
			rk[6] ^= shuffle2(rk[4], rk[5], (uint4)(1, 2, 3, 4));
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[6]);
			rk[7] ^= shuffle2(rk[5], rk[6], (uint4)(1, 2, 3, 4));
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[7]);
			if(!y) p[0] = AES_Round(AES0, AES1, AES2, AES3, x, p[0]);
			else p[2] = AES_Round(AES0, AES1, AES2, AES3, x, p[2]);
		}
		if(r == 2)
		{
			rk[0] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[0], rk[7]);
			uint4 x = rk[0] ^ p[0];
			rk[1] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[1], rk[0]);
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[1]);
			rk[2] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[2], rk[1]);
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[2]);
			rk[3] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[3], rk[2]);
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[3]);
			p[3] = AES_Round(AES0, AES1, AES2, AES3, x, p[3]);
			
			rk[4] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[4], rk[3]);
			x = rk[4] ^ p[2];
			rk[5] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[5], rk[4]);
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[5]);
			rk[6] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[6], rk[5]);
			rk[6].s13 ^= (uint2)(0x200, 0xFFFFFFFF);
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[6]);
			rk[7] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[7], rk[6]);
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[7]);
			p[1] = AES_Round(AES0, AES1, AES2, AES3, x, p[1]);
		}
	}
	
	// h[0] ^ p[2], h[1] ^ p[3], h[2] ^ p[0], h[3] ^ p[1]
	for(int i = 0; i < 4; ++i) vstore4(h[i] ^ p[(i + 2) & 3], i, hashes->h4);
	
	mem_fence(CLK_GLOBAL_MEM_FENCE);
}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search13(__global hash_t *hashes, __global uint *Branch7Nonces, __global uint *Branch8Nonces, const ulong FoundIdx)
{
	int q[256] = { 0 };
	unsigned char x[128] = { 0 };
	
	hashes += get_global_id(0) - get_global_offset(0);
	
	((uint16 *)x)[0] = vload16(0, hashes->h4);
	
	SIMD_Expand(q, x);
	
	uint8 State[4] = { 	(uint8)(0x0BA16B95, 0x72F999AD, 0x9FECC2AE, 0xBA3264FC, 0x5E894929, 0x8E9F30E5, 0x2F1DAA37, 0xF0F2C558),
						(uint8)(0xAC506643, 0xA90635A5, 0xE25B878B, 0xAAB7878F, 0x88817F7A, 0x0A02892B, 0x559A7550, 0x598F657E),
						(uint8)(0x7EEF60A1, 0x6B70E3E8, 0x9C1714D1, 0xB958E2A8, 0xAB02675E, 0xED1C014F, 0xCD8D65BB, 0xFDB7A257),
						(uint8)(0x09254899, 0xD699C7BC, 0x9019B6DC, 0x2B9022E4, 0x8FA14956, 0x21BF9BD3, 0xB94D0943, 0x6FFDDC22) };
	
	State[0] ^= ((uint8 *)x)[0];
	State[1] ^= ((uint8 *)x)[1];
	
	SIMD_Compress(q, State, State + 1, State + 2, State + 3);
	
	vstore8(State[0], 0, hashes->h4);
	vstore8(State[1], 1, hashes->h4);
	
	if(hashes->h4[0] & 24) Branch7Nonces[atomic_inc(Branch7Nonces + FoundIdx)] = get_global_id(0) - get_global_offset(0);
	else Branch8Nonces[atomic_inc(Branch8Nonces + FoundIdx)] = get_global_id(0) - get_global_offset(0);
	
	mem_fence(CLK_GLOBAL_MEM_FENCE);
}


  //dec = ((hash.h4[0] & 24) != 0);
__kernel void search14(__global hash_t *hashes, __global uint *BranchNonces)
{
	ulong8 n, h, st;
	__local ulong LT0[256];
	
	hashes += BranchNonces[get_global_id(0) - get_global_offset(0)];
	
	for(int i = get_local_id(0); i < 256; i += get_local_size(0)) LT0[i] = WPL_T0_G[i];
	
	st = n = vload8(0, hashes->h8);
	h = 0;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	n = WhirlpoolRound(n, h, LT0);
	
	st ^= n;
	n = (ulong8)(0x80UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0x2000000000000UL) ^ st;
	h = st;
	
	n = WhirlpoolRound(n, h, LT0);
	
	st ^= n;
	st.s0 ^= 0x80UL;
	st.s7 ^= 0x2000000000000UL;
	
	vstore8(st, 0, hashes->h8);
	
	barrier(CLK_GLOBAL_MEM_FENCE);
}

 __kernel void search15(__global hash_t *hashes, __global uint *BranchNonces)
{
	hashes += BranchNonces[get_global_id(0) - get_global_offset(0)];
	
   // Haval
  sph_u32 s0 = SPH_C32(0x243F6A88);
  sph_u32 s1 = SPH_C32(0x85A308D3);
  sph_u32 s2 = SPH_C32(0x13198A2E);
  sph_u32 s3 = SPH_C32(0x03707344);
  sph_u32 s4 = SPH_C32(0xA4093822);
  sph_u32 s5 = SPH_C32(0x299F31D0);
  sph_u32 s6 = SPH_C32(0x082EFA98);
  sph_u32 s7 = SPH_C32(0xEC4E6C89);

  sph_u32 X_var[32];

  for (int i = 0; i < 16; i++)
    X_var[i] = hashes->h4[i];

  X_var[16] = 0x00000001U;
  X_var[17] = 0x00000000U;
  X_var[18] = 0x00000000U;
  X_var[19] = 0x00000000U;
  X_var[20] = 0x00000000U;
  X_var[21] = 0x00000000U;
  X_var[22] = 0x00000000U;
  X_var[23] = 0x00000000U;
  X_var[24] = 0x00000000U;
  X_var[25] = 0x00000000U;
  X_var[26] = 0x00000000U;
  X_var[27] = 0x00000000U;
  X_var[28] = 0x00000000U;
  X_var[29] = 0x40290000U;
  X_var[30] = 0x00000200U;
  X_var[31] = 0x00000000U;

#define A(x) X_var[x]
  CORE5(A);

  hashes->h4[0] = s0;
  hashes->h4[1] = s1;
  hashes->h4[2] = s2;
  hashes->h4[3] = s3;
  hashes->h4[4] = s4;
  hashes->h4[5] = s5;
  hashes->h4[6] = s6;
  hashes->h4[7] = s7;
  hashes->h8[4] = 0;
  hashes->h8[5] = 0;
  hashes->h8[6] = 0;
  hashes->h8[7] = 0;
}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search16(__global hash_t *hashes)
{
	const uint hidx = get_global_id(0) - get_global_offset(0);
	__local uint AES0[256];
	
	const uint step = get_local_size(0);
	
	AES0[get_local_id(0)] = AES0_C[get_local_id(0)];
	AES0[get_local_id(0) + 64] = AES0_C[get_local_id(0) + 64];
	AES0[get_local_id(0) + 128] = AES0_C[get_local_id(0) + 128];
	AES0[get_local_id(0) + 192] = AES0_C[get_local_id(0) + 192];
	
	uint4 W[16];
	
	#pragma unroll
	for(int i = 0; i < 8; ++i) W[i] = (uint4)(512, 0, 0, 0);
	
	((uint16 *)W)[2] = vload16(0, hashes[hidx].h4);
	
	W[12] = (uint4)(0x80, 0, 0, 0);
	W[13] = (uint4)(0, 0, 0, 0);
	W[14] = (uint4)(0, 0, 0, 0x02000000);
	W[15] = (uint4)(512, 0, 0, 0);
	
	mem_fence(CLK_LOCAL_MEM_FENCE);
	
	#pragma unroll 1
	for(uchar i = 0; i < 10; ++i)
	{
		BigSubBytesSmall(AES0, W, i);	
		BigShiftRows(W);
		BigMixColumns(W);
	}
	
	#pragma unroll
	for(int i = 0; i < 4; ++i) vstore4(vload4(i, hashes[hidx].h4) ^ W[i] ^ W[i + 8] ^ (uint4)(512, 0, 0, 0), i, hashes[hidx].h4);
}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search17(__global hash_t *hashes, __global uint *Branch9Nonces, __global uint *Branch10Nonces, const ulong FoundIdx)
{
    ulong16 V;
	ulong M[16];
	uint idx = get_global_id(0) - get_global_offset(0);
	
	hashes += idx;
	
	((ulong8 *)M)[0] = VSWAP8(vload8(0, hashes->h8));
	((ulong8 *)M)[1] = (ulong8)(0x8000000000000000UL, 0UL, 0UL, 0UL, 0UL, 1UL, 0UL, 0x200UL);
	
	V.lo = vload8(0, BLAKE512_IV);
	V.hi = vload8(0, blake_cb);
	
	V.scd ^= (ulong2)(0x200UL, 0x200UL);
	
	for(bool flag = false; ; flag = true)
	{
		BLAKE_RND(0);
		BLAKE_RND(1);
		BLAKE_RND(2);
		BLAKE_RND(3);
		BLAKE_RND(4);
		BLAKE_RND(5);
		if(flag) break;
		BLAKE_RND(6);
		BLAKE_RND(7);
		BLAKE_RND(8);
		BLAKE_RND(9);
	}
	
	vstore8(VSWAP8(((__constant ulong8 *)BLAKE512_IV)[0] ^ V.lo ^ V.hi), 0, hashes->h8);
	
	if(hashes->h4[0] & 24) Branch9Nonces[atomic_inc(Branch9Nonces + FoundIdx)] = idx;
	else Branch10Nonces[atomic_inc(Branch10Nonces + FoundIdx)] = idx;
	
	mem_fence(CLK_GLOBAL_MEM_FENCE);
}


  //dec = ((hash.h4[0] & 24) != 0);
__kernel void search18(__global hash_t *hashes, __global uint *BranchNonces)
{
	__local uint AES0[256], AES1[256], AES2[256], AES3[256];
	hashes += BranchNonces[get_global_id(0) - get_global_offset(0)];
	
	const int step = get_local_size(0);
	
	for(int i = get_local_id(0); i < 256; i += step)
	{
		const uint tmp = AES0_C[i];
		AES0[i] = tmp;
		AES1[i] = rotate(tmp, 8U);
		AES2[i] = rotate(tmp, 16U);
		AES3[i] = rotate(tmp, 24U);
	}
	
	const uint4 h[4] = {(uint4)(0x72FCCDD8, 0x79CA4727, 0x128A077B, 0x40D55AEC), (uint4)(0xD1901A06, 0x430AE307, 0xB29F5CD1, 0xDF07FBFC), \
						(uint4)(0x8E45D73D, 0x681AB538, 0xBDE86578, 0xDD577E47), (uint4)(0xE275EADE, 0x502D9FCD, 0xB9357178, 0x022A4B9A) };
	
	uint4 rk[8] = { (uint4)(0) }, p[4] = { h[0], h[1], h[2], h[3] };
	
	((uint16 *)rk)[0] = vload16(0, hashes->h4);
	rk[4].s0 = 0x80;
	rk[6].s3 = 0x2000000;
	rk[7].s3 = 0x2000000;
	mem_fence(CLK_LOCAL_MEM_FENCE);
	
	#pragma unroll 1
	for(int r = 0; r < 3; ++r)
	{
		if(r == 0)
		{
			p[0] = Shavite_AES_4Round(AES0, AES1, AES2, AES3, p[1] ^ rk[0], &(rk[1]), p[0]);
			p[2] = Shavite_AES_4Round(AES0, AES1, AES2, AES3, p[3] ^ rk[4], &(rk[5]), p[2]);
		}
		#pragma unroll 1
		for(int y = 0; y < 2; ++y)
		{
			rk[0] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[0], rk[7]);
			rk[0].s03 ^= ((!y && !r) ? (uint2)(0x200, 0xFFFFFFFF) : (uint2)(0));
			uint4 x = rk[0] ^ (y != 1 ? p[0] : p[2]);
			rk[1] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[1], rk[0]);
			rk[1].s3 ^= (!y && r == 1 ? 0xFFFFFDFFU : 0);	// ~(0x200)
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[1]);
			rk[2] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[2], rk[1]);
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[2]);
			rk[3] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[3], rk[2]);
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[3]);
			if(y != 1) p[3] = AES_Round(AES0, AES1, AES2, AES3, x, p[3]);
			else p[1] = AES_Round(AES0, AES1, AES2, AES3, x, p[1]);
			
			rk[4] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[4], rk[3]);
			x = rk[4] ^ (y != 1 ? p[2] : p[0]);
			rk[5] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[5], rk[4]);
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[5]);
			rk[6] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[6], rk[5]);
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[6]);
			rk[7] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[7], rk[6]);
			rk[7].s23 ^= ((!y && r == 2) ? (uint2)(0x200, 0xFFFFFFFF) : (uint2)(0));
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[7]);
			if(y != 1) p[1] = AES_Round(AES0, AES1, AES2, AES3, x, p[1]);
			else p[3] = AES_Round(AES0, AES1, AES2, AES3, x, p[3]);
						
			rk[0] ^= shuffle2(rk[6], rk[7], (uint4)(1, 2, 3, 4));
			x = rk[0] ^ (!y ? p[3] : p[1]);
			rk[1] ^= shuffle2(rk[7], rk[0], (uint4)(1, 2, 3, 4));
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[1]);
			rk[2] ^= shuffle2(rk[0], rk[1], (uint4)(1, 2, 3, 4));
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[2]);
			rk[3] ^= shuffle2(rk[1], rk[2], (uint4)(1, 2, 3, 4));
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[3]);
			if(!y) p[2] = AES_Round(AES0, AES1, AES2, AES3, x, p[2]);
			else p[0] = AES_Round(AES0, AES1, AES2, AES3, x, p[0]);
					
			rk[4] ^= shuffle2(rk[2], rk[3], (uint4)(1, 2, 3, 4));
			x = rk[4] ^ (!y ? p[1] : p[3]);
			rk[5] ^= shuffle2(rk[3], rk[4], (uint4)(1, 2, 3, 4));
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[5]);
			rk[6] ^= shuffle2(rk[4], rk[5], (uint4)(1, 2, 3, 4));
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[6]);
			rk[7] ^= shuffle2(rk[5], rk[6], (uint4)(1, 2, 3, 4));
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[7]);
			if(!y) p[0] = AES_Round(AES0, AES1, AES2, AES3, x, p[0]);
			else p[2] = AES_Round(AES0, AES1, AES2, AES3, x, p[2]);
		}
		if(r == 2)
		{
			rk[0] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[0], rk[7]);
			uint4 x = rk[0] ^ p[0];
			rk[1] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[1], rk[0]);
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[1]);
			rk[2] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[2], rk[1]);
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[2]);
			rk[3] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[3], rk[2]);
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[3]);
			p[3] = AES_Round(AES0, AES1, AES2, AES3, x, p[3]);
			
			rk[4] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[4], rk[3]);
			x = rk[4] ^ p[2];
			rk[5] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[5], rk[4]);
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[5]);
			rk[6] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[6], rk[5]);
			rk[6].s13 ^= (uint2)(0x200, 0xFFFFFFFF);
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[6]);
			rk[7] = Shavite_Key_Expand(AES0, AES1, AES2, AES3, rk[7], rk[6]);
			x = AES_Round(AES0, AES1, AES2, AES3, x, rk[7]);
			p[1] = AES_Round(AES0, AES1, AES2, AES3, x, p[1]);
		}
	}
	
	// h[0] ^ p[2], h[1] ^ p[3], h[2] ^ p[0], h[3] ^ p[1]
	for(int i = 0; i < 4; ++i) vstore4(h[i] ^ p[(i + 2) & 3], i, hashes->h4);
	
	mem_fence(CLK_GLOBAL_MEM_FENCE);
}

 __kernel void search19(__global hash_t *hashes, __global uint *BranchNonces)
{
	hashes += BranchNonces[get_global_id(0) - get_global_offset(0)];
	
	uint8 V[5] =
	{
		(uint8)(0x6D251E69U, 0x44B051E0U, 0x4EAA6FB4U, 0xDBF78465U, 0x6E292011U, 0x90152DF4U, 0xEE058139U, 0xDEF610BBU),
		(uint8)(0xC3B44B95U, 0xD9D2F256U, 0x70EEE9A0U, 0xDE099FA3U, 0x5D9B0557U, 0x8FC944B3U, 0xCF1CCF0EU, 0x746CD581U),
		(uint8)(0xF7EFC89DU, 0x5DBA5781U, 0x04016CE5U, 0xAD659C05U, 0x0306194FU, 0x666D1836U, 0x24AA230AU, 0x8B264AE7U),
		(uint8)(0x858075D5U, 0x36D79CCEU, 0xE571F7D7U, 0x204B1F67U, 0x35870C6AU, 0x57E9E923U, 0x14BCB808U, 0x7CDE72CEU),
		(uint8)(0x6C68E9BEU, 0x5EC41E22U, 0xC825B7C7U, 0xAFFB4363U, 0xF5DF3999U, 0x0FC688F1U, 0xB07224CCU, 0x03E86CEAU)
	};
	
	#pragma unroll
	for(int i = 0; i < 8; ++i) hashes->h8[i] = SWAP8(hashes->h8[i]);
	
	#pragma unroll
    for(int i = 0; i < 6; ++i)
    {
		uint8 M;
		switch(i)
		{
			case 0:
			case 1:
				M = shuffle(vload8(i, hashes->h4), (uint8)(1, 0, 3, 2, 5, 4, 7, 6));
				break;
			case 2:
			case 3:
				M = (uint8)(0);
				M.s0 = (i == 2) ? 0x80000000 : 0;
				break;
			case 4:
			case 5:
				vstore8(shuffle(V[0] ^ V[1] ^ V[2] ^ V[3] ^ V[4], (uint8)(1, 0, 3, 2, 5, 4, 7, 6)), i & 1, hashes->h4);
		}
		if(i == 5) break;
		
		MessageInj(V, M);
		LuffaPerm(V);
    }
	
	#pragma unroll
	for(int i = 0; i < 8; ++i) hashes->h8[i] = SWAP8(hashes->h8[i]);
	
	mem_fence(CLK_GLOBAL_MEM_FENCE);
}


__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search20(__global hash_t *hashes)
{
	__local uint T512_L[1024];
	__constant const uint *T512_C = &T512[0][0];

	for(int i = get_local_id(0), step = get_local_size(0); i < 1024; i += step) T512_L[i] = T512_C[i];
	
	mem_fence(CLK_LOCAL_MEM_FENCE);
	
	uint16 h = ((__constant uint16 *)HAMSI_IV512)[0];
	
	hashes += get_global_id(0) - get_global_offset(0);
		
	#pragma unroll 1
	for(int i = 0; i < 10; ++i)
	{
		const __local uint16 *tp = (__local uint16 *)T512_L;
		const uint rnds = (i == 9) ? 12 : 6;
		uint16 m = 0;
		ulong tmp;
		
		if(i < 8) tmp = hashes->h8[i];
		else tmp = ((i == 8) ? 0x80 : 0x2000000000000UL);
				
		#pragma unroll
		for(ulong y = 1; y > 0; y <<= 1, ++tp) if(tmp & y) m ^= *tp;
		
		h = HamsiRound(m, h, (i == 9) ? alpha_f : alpha_n, rnds);
	}
	
	vstore16(VSWAP4(h), 0, hashes->h4);
	
	mem_fence(CLK_GLOBAL_MEM_FENCE);
}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search21(__global hash_t *hashes, __global uint *Branch11Nonces, __global uint *Branch12Nonces, const ulong FoundIdx)
{
	const uint idx = get_global_id(0) - get_global_offset(0);
	
	// mixtab
	__local uint mixtab0[256], mixtab1[256], mixtab2[256], mixtab3[256];
	
	// Let the compiler unroll this one
	for(int i = get_local_id(0), step = get_local_size(0); i < 256; i += step)
	{
		mixtab0[i] = mixtab0_c[i];
		mixtab1[i] = rotate(mixtab0_c[i], 24U);
		mixtab2[i] = rotate(mixtab0_c[i], 16U);
		mixtab3[i] = rotate(mixtab0_c[i], 8U);
	}

	// fugue
	
	uint4 S[9] = { 0 };
	
	S[5] = vload4(0, IV512);
	S[6] = vload4(1, IV512);
	S[7] = vload4(2, IV512);
	S[8] = vload4(3, IV512);
	
	uint input[18];
	((uint16 *)input)[0] = VSWAP4(vload16(0, hashes[idx].h4));
	input[16] = 0;
	input[17] = 0x200;
	
	mem_fence(CLK_LOCAL_MEM_FENCE);
	
	#pragma unroll
	for(int i = 0; i < 19; ++i)
	{
		if(i < 18) TIX4(input[i], S[0].s0, S[0].s1, S[1].s0, S[1].s3, S[2].s0, S[5].s2, S[6].s0, S[6].s3, S[7].s2);
				
		for(int x = 0; x < 8; ++x)
		{
			ROR3;
			CMIX36(S[0].s0, S[0].s1, S[0].s2, S[1].s0, S[1].s1, S[1].s2, S[4].s2, S[4].s3, S[5].s0);
			SMIX(mixtab0, mixtab1, mixtab2, mixtab3, &S[0]);
			
			ROR3;
			CMIX36(S[0].s0, S[0].s1, S[0].s2, S[1].s0, S[1].s1, S[1].s2, S[4].s2, S[4].s3, S[5].s0);
			SMIX(mixtab0, mixtab1, mixtab2, mixtab3, &S[0]);
			
			ROR3;
			CMIX36(S[0].s0, S[0].s1, S[0].s2, S[1].s0, S[1].s1, S[1].s2, S[4].s2, S[4].s3, S[5].s0);
			SMIX(mixtab0, mixtab1, mixtab2, mixtab3, &S[0]);
			
			ROR3;
			CMIX36(S[0].s0, S[0].s1, S[0].s2, S[1].s0, S[1].s1, S[1].s2, S[4].s2, S[4].s3, S[5].s0);
			SMIX(mixtab0, mixtab1, mixtab2, mixtab3, &S[0]);
			
			if(i != 18) break;
		}
	}
	
	#pragma unroll 4
	for(int i = 0; i < 12; ++i)
	{
		S[1].s0 ^= S[0].s0;
		S[2].s1 ^= S[0].s0;
		S[4].s2 ^= S[0].s0;
		S[6].s3 ^= S[0].s0;
		
		ROR9;
		SMIX(mixtab0, mixtab1, mixtab2, mixtab3, &S[0]);
		
		S[1].s0 ^= S[0].s0;
		S[2].s2 ^= S[0].s0;
		S[4].s2 ^= S[0].s0;
		S[6].s3 ^= S[0].s0;
		
		ROR9;
		SMIX(mixtab0, mixtab1, mixtab2, mixtab3, &S[0]);
		
		S[1].s0 ^= S[0].s0;
		S[2].s2 ^= S[0].s0;
		S[4].s3 ^= S[0].s0;
		S[6].s3 ^= S[0].s0;
		
		ROR9;
		SMIX(mixtab0, mixtab1, mixtab2, mixtab3, &S[0]);
		
		S[1].s0 ^= S[0].s0;
		S[2].s2 ^= S[0].s0;
		S[4].s3 ^= S[0].s0;
		S[7].s0 ^= S[0].s0;
		
		ROR8;
		SMIX(mixtab0, mixtab1, mixtab2, mixtab3, &S[0]);
	}
	
	S[1].s0 ^= S[0].s0;
	S[2].s1 ^= S[0].s0;
	S[4].s2 ^= S[0].s0;
	S[6].s3 ^= S[0].s0;
	
	ROR9;
	SMIX(mixtab0, mixtab1, mixtab2, mixtab3, &S[0]);
	
	S[1].s0 ^= S[0].s0;
	S[2].s2 ^= S[0].s0;
	S[4].s2 ^= S[0].s0;
	S[6].s3 ^= S[0].s0;
	
	ROR9;
	SMIX(mixtab0, mixtab1, mixtab2, mixtab3, &S[0]);
	
	S[1].s0 ^= S[0].s0;
	S[2].s2 ^= S[0].s0;
	S[4].s3 ^= S[0].s0;
	S[6].s3 ^= S[0].s0;
	
	ROR9;
	SMIX(mixtab0, mixtab1, mixtab2, mixtab3, &S[0]);
	
	S[1].s0 ^= S[0].s0;
	S[2].s2 ^= S[0].s0;
	S[4].s3 ^= S[0].s0;
	S[7].s0 ^= S[0].s0;
	
	ROR8;
	SMIX(mixtab0, mixtab1, mixtab2, mixtab3, &S[0]);
	
	S[1].s0 ^= S[0].s0;
	S[2].s1 ^= S[0].s0;
	S[4].s2 ^= S[0].s0;
	S[6].s3 ^= S[0].s0;
	
	hashes[idx].h4[0] = SWAP4(S[0].s1);
	hashes[idx].h4[1] = SWAP4(S[0].s2);
	hashes[idx].h4[2] = SWAP4(S[0].s3);
	hashes[idx].h4[3] = SWAP4(S[1].s0);
	hashes[idx].h4[4] = SWAP4(S[2].s1);
	hashes[idx].h4[5] = SWAP4(S[2].s2);
	hashes[idx].h4[6] = SWAP4(S[2].s3);
	hashes[idx].h4[7] = SWAP4(S[3].s0);
	hashes[idx].h4[8] = SWAP4(S[4].s2);
	hashes[idx].h4[9] = SWAP4(S[4].s3);
	hashes[idx].h4[10] = SWAP4(S[5].s0);
	hashes[idx].h4[11] = SWAP4(S[5].s1);
	hashes[idx].h4[12] = SWAP4(S[6].s3);
	hashes[idx].h4[13] = SWAP4(S[7].s0);
	hashes[idx].h4[14] = SWAP4(S[7].s1);
	hashes[idx].h4[15] = SWAP4(S[7].s2);

	if(hashes[idx].h4[0] & 24) Branch11Nonces[atomic_inc(Branch11Nonces + FoundIdx)] = get_global_id(0) - get_global_offset(0);
	else Branch12Nonces[atomic_inc(Branch12Nonces + FoundIdx)] = get_global_id(0) - get_global_offset(0);

	mem_fence(CLK_GLOBAL_MEM_FENCE);
}


  //dec = ((hash.h4[0] & 24) != 0);
__kernel void search22(__global hash_t *hashes, __global uint *BranchNonces)
{
	const uint hidx = BranchNonces[get_global_id(0) - get_global_offset(0)];
	__local uint AES0[256];
	
	const uint step = get_local_size(0);
	
	AES0[get_local_id(0)] = AES0_C[get_local_id(0)];
	AES0[get_local_id(0) + 64] = AES0_C[get_local_id(0) + 64];
	AES0[get_local_id(0) + 128] = AES0_C[get_local_id(0) + 128];
	AES0[get_local_id(0) + 192] = AES0_C[get_local_id(0) + 192];
	
	// echo
	uint4 W[16];
	
	#pragma unroll
	for(int i = 0; i < 8; ++i) W[i] = (uint4)(512, 0, 0, 0);
	
	((uint16 *)W)[2] = vload16(0, hashes[hidx].h4);
	
	W[12] = (uint4)(0x80, 0, 0, 0);
	W[13] = (uint4)(0, 0, 0, 0);
	W[14] = (uint4)(0, 0, 0, 0x02000000);
	W[15] = (uint4)(512, 0, 0, 0);
	
	mem_fence(CLK_LOCAL_MEM_FENCE);
	
	#pragma unroll 1
	for(uchar i = 0; i < 10; ++i)
	{
		BigSubBytesSmall(AES0, W, i);	
		BigShiftRows(W);
		BigMixColumns(W);
	}
	
	#pragma unroll
	for(int i = 0; i < 4; ++i) vstore4(vload4(i, hashes[hidx].h4) ^ W[i] ^ W[i + 8] ^ (uint4)(512, 0, 0, 0), i, hashes[hidx].h4);
}

 __kernel void search23(__global hash_t *hashes, __global uint *BranchNonces)
{
	int q[256] = { 0 };
	unsigned char x[128] = { 0 };
	
	hashes += BranchNonces[get_global_id(0) - get_global_offset(0)];
	
	((uint16 *)x)[0] = vload16(0, hashes->h4);
	
	SIMD_Expand(q, x);
	
	uint8 State[4] = { 	(uint8)(0x0BA16B95, 0x72F999AD, 0x9FECC2AE, 0xBA3264FC, 0x5E894929, 0x8E9F30E5, 0x2F1DAA37, 0xF0F2C558),
						(uint8)(0xAC506643, 0xA90635A5, 0xE25B878B, 0xAAB7878F, 0x88817F7A, 0x0A02892B, 0x559A7550, 0x598F657E),
						(uint8)(0x7EEF60A1, 0x6B70E3E8, 0x9C1714D1, 0xB958E2A8, 0xAB02675E, 0xED1C014F, 0xCD8D65BB, 0xFDB7A257),
						(uint8)(0x09254899, 0xD699C7BC, 0x9019B6DC, 0x2B9022E4, 0x8FA14956, 0x21BF9BD3, 0xB94D0943, 0x6FFDDC22) };
	
	State[0] ^= ((uint8 *)x)[0];
	State[1] ^= ((uint8 *)x)[1];
	
	SIMD_Compress(q, State, State + 1, State + 2, State + 3);
	
	vstore8(State[0], 0, hashes->h4);
	vstore8(State[1], 1, hashes->h4);
	
	mem_fence(CLK_GLOBAL_MEM_FENCE);
}


__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search24(__global hash_t *hashes)
{
	const uint idx = get_global_id(0) - get_global_offset(0);

	// shabal
	uint16 A, B, C, M;
	uint Wlow = 1;
	
	A.s0 = A_init_512[0];
	A.s1 = A_init_512[1];
	A.s2 = A_init_512[2];
	A.s3 = A_init_512[3];
	A.s4 = A_init_512[4];
	A.s5 = A_init_512[5];
	A.s6 = A_init_512[6];
	A.s7 = A_init_512[7];
	A.s8 = A_init_512[8];
	A.s9 = A_init_512[9];
	A.sa = A_init_512[10];
	A.sb = A_init_512[11];
	
	B = vload16(0, B_init_512);
	C = vload16(0, C_init_512);
	M = vload16(0, hashes[idx].h4);
	
	// INPUT_BLOCK_ADD
	B += M;
	
	// XOR_W
	//do { A.s0 ^= Wlow; } while(0);
	A.s0 ^= Wlow;
	
	// APPLY_P
	B = rotate(B, 17U);
	SHABAL_PERM_V;
	
	uint16 tmpC1, tmpC2, tmpC3;
	
	tmpC1 = shuffle2(C, (uint16)0, (uint16)(11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 17, 17, 17, 17));
	tmpC2 = shuffle2(C, (uint16)0, (uint16)(15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 17, 17, 17));
	tmpC3 = shuffle2(C, (uint16)0, (uint16)(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 17, 17, 17));
	
	A += tmpC1 + tmpC2 + tmpC3;
		
	// INPUT_BLOCK_SUB
	C -= M;
	
	++Wlow;
	M = 0;
	M.s0 = 0x80;
	
	#pragma unroll 2
	for(int i = 0; i < 4; ++i)
	{
		SWAP_BC_V;
		
		// INPUT_BLOCK_ADD
		if(i == 0) B.s0 += M.s0;
		
		// XOR_W;
		A.s0 ^= Wlow;
		
		// APPLY_P
		B = rotate(B, 17U);
		SHABAL_PERM_V;
		
		if(i == 3) break;
		
		tmpC1 = shuffle2(C, (uint16)0, (uint16)(11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 17, 17, 17, 17));
		tmpC2 = shuffle2(C, (uint16)0, (uint16)(15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 17, 17, 17));
		tmpC3 = shuffle2(C, (uint16)0, (uint16)(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 17, 17, 17));
	
		A += tmpC1 + tmpC2 + tmpC3;
	}
	
	vstore16(B, 0, hashes[idx].h4);

	mem_fence(CLK_GLOBAL_MEM_FENCE);
}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search25(__global hash_t *hashes, __global uint *Branch13Nonces, __global uint *Branch14Nonces, const ulong FoundIdx)
{
	ulong8 n, h, st;
	__local ulong LT0[256];
	
	hashes += get_global_id(0) - get_global_offset(0);
	
	for(int i = get_local_id(0); i < 256; i += get_local_size(0)) LT0[i] = WPL_T0_G[i];
	
	st = n = vload8(0, hashes->h8);
	h = 0;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	n = WhirlpoolRound(n, h, LT0);
	
	st ^= n;
	n = (ulong8)(0x80UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0x2000000000000UL) ^ st;
	h = st;
	
	n = WhirlpoolRound(n, h, LT0);
	
	st ^= n;
	st.s0 ^= 0x80UL;
	st.s7 ^= 0x2000000000000UL;
	
	vstore8(st, 0, hashes->h8);
	
	if(hashes->h4[0] & 24) Branch13Nonces[atomic_inc(Branch13Nonces + FoundIdx)] = get_global_id(0) - get_global_offset(0);
	else Branch14Nonces[atomic_inc(Branch14Nonces + FoundIdx)] = get_global_id(0) - get_global_offset(0);
	
	barrier(CLK_GLOBAL_MEM_FENCE);
}


  //dec = ((hash.h4[0] & 24) != 0);
 __kernel void search26(__global hash_t *hashes, __global uint *BranchNonces)
{
	const uint idx = BranchNonces[get_global_id(0) - get_global_offset(0)];
	
	// mixtab
	__local uint mixtab0[256], mixtab1[256], mixtab2[256], mixtab3[256];
	
	// Let the compiler unroll this one
	for(int i = get_local_id(0), step = get_local_size(0); i < 256; i += step)
	{
		mixtab0[i] = mixtab0_c[i];
		mixtab1[i] = rotate(mixtab0_c[i], 24U);
		mixtab2[i] = rotate(mixtab0_c[i], 16U);
		mixtab3[i] = rotate(mixtab0_c[i], 8U);
	}

	// fugue
	
	uint4 S[9] = { 0 };
	
	S[5] = vload4(0, IV512);
	S[6] = vload4(1, IV512);
	S[7] = vload4(2, IV512);
	S[8] = vload4(3, IV512);
	
	uint input[18];
	((uint16 *)input)[0] = VSWAP4(vload16(0, hashes[idx].h4));
	input[16] = 0;
	input[17] = 0x200;
	
	mem_fence(CLK_LOCAL_MEM_FENCE);
	
	#pragma unroll
	for(int i = 0; i < 19; ++i)
	{
		if(i < 18) TIX4(input[i], S[0].s0, S[0].s1, S[1].s0, S[1].s3, S[2].s0, S[5].s2, S[6].s0, S[6].s3, S[7].s2);
				
		for(int x = 0; x < 8; ++x)
		{
			ROR3;
			CMIX36(S[0].s0, S[0].s1, S[0].s2, S[1].s0, S[1].s1, S[1].s2, S[4].s2, S[4].s3, S[5].s0);
			SMIX(mixtab0, mixtab1, mixtab2, mixtab3, &S[0]);
			
			ROR3;
			CMIX36(S[0].s0, S[0].s1, S[0].s2, S[1].s0, S[1].s1, S[1].s2, S[4].s2, S[4].s3, S[5].s0);
			SMIX(mixtab0, mixtab1, mixtab2, mixtab3, &S[0]);
			
			ROR3;
			CMIX36(S[0].s0, S[0].s1, S[0].s2, S[1].s0, S[1].s1, S[1].s2, S[4].s2, S[4].s3, S[5].s0);
			SMIX(mixtab0, mixtab1, mixtab2, mixtab3, &S[0]);
			
			ROR3;
			CMIX36(S[0].s0, S[0].s1, S[0].s2, S[1].s0, S[1].s1, S[1].s2, S[4].s2, S[4].s3, S[5].s0);
			SMIX(mixtab0, mixtab1, mixtab2, mixtab3, &S[0]);
			
			if(i != 18) break;
		}
	}
	
	#pragma unroll 4
	for(int i = 0; i < 12; ++i)
	{
		S[1].s0 ^= S[0].s0;
		S[2].s1 ^= S[0].s0;
		S[4].s2 ^= S[0].s0;
		S[6].s3 ^= S[0].s0;
		
		ROR9;
		SMIX(mixtab0, mixtab1, mixtab2, mixtab3, &S[0]);
		
		S[1].s0 ^= S[0].s0;
		S[2].s2 ^= S[0].s0;
		S[4].s2 ^= S[0].s0;
		S[6].s3 ^= S[0].s0;
		
		ROR9;
		SMIX(mixtab0, mixtab1, mixtab2, mixtab3, &S[0]);
		
		S[1].s0 ^= S[0].s0;
		S[2].s2 ^= S[0].s0;
		S[4].s3 ^= S[0].s0;
		S[6].s3 ^= S[0].s0;
		
		ROR9;
		SMIX(mixtab0, mixtab1, mixtab2, mixtab3, &S[0]);
		
		S[1].s0 ^= S[0].s0;
		S[2].s2 ^= S[0].s0;
		S[4].s3 ^= S[0].s0;
		S[7].s0 ^= S[0].s0;
		
		ROR8;
		SMIX(mixtab0, mixtab1, mixtab2, mixtab3, &S[0]);
	}
	
	S[1].s0 ^= S[0].s0;
	S[2].s1 ^= S[0].s0;
	S[4].s2 ^= S[0].s0;
	S[6].s3 ^= S[0].s0;
	
	ROR9;
	SMIX(mixtab0, mixtab1, mixtab2, mixtab3, &S[0]);
	
	S[1].s0 ^= S[0].s0;
	S[2].s2 ^= S[0].s0;
	S[4].s2 ^= S[0].s0;
	S[6].s3 ^= S[0].s0;
	
	ROR9;
	SMIX(mixtab0, mixtab1, mixtab2, mixtab3, &S[0]);
	
	S[1].s0 ^= S[0].s0;
	S[2].s2 ^= S[0].s0;
	S[4].s3 ^= S[0].s0;
	S[6].s3 ^= S[0].s0;
	
	ROR9;
	SMIX(mixtab0, mixtab1, mixtab2, mixtab3, &S[0]);
	
	S[1].s0 ^= S[0].s0;
	S[2].s2 ^= S[0].s0;
	S[4].s3 ^= S[0].s0;
	S[7].s0 ^= S[0].s0;
	
	ROR8;
	SMIX(mixtab0, mixtab1, mixtab2, mixtab3, &S[0]);
	
	S[1].s0 ^= S[0].s0;
	S[2].s1 ^= S[0].s0;
	S[4].s2 ^= S[0].s0;
	S[6].s3 ^= S[0].s0;
	
	hashes[idx].h4[0] = SWAP4(S[0].s1);
	hashes[idx].h4[1] = SWAP4(S[0].s2);
	hashes[idx].h4[2] = SWAP4(S[0].s3);
	hashes[idx].h4[3] = SWAP4(S[1].s0);
	hashes[idx].h4[4] = SWAP4(S[2].s1);
	hashes[idx].h4[5] = SWAP4(S[2].s2);
	hashes[idx].h4[6] = SWAP4(S[2].s3);
	hashes[idx].h4[7] = SWAP4(S[3].s0);
	hashes[idx].h4[8] = SWAP4(S[4].s2);
	hashes[idx].h4[9] = SWAP4(S[4].s3);
	hashes[idx].h4[10] = SWAP4(S[5].s0);
	hashes[idx].h4[11] = SWAP4(S[5].s1);
	hashes[idx].h4[12] = SWAP4(S[6].s3);
	hashes[idx].h4[13] = SWAP4(S[7].s0);
	hashes[idx].h4[14] = SWAP4(S[7].s1);
	hashes[idx].h4[15] = SWAP4(S[7].s2);

	mem_fence(CLK_GLOBAL_MEM_FENCE);
}

static const __constant ulong SHA512_INIT[8] =
{
	0x6A09E667F3BCC908UL, 0xBB67AE8584CAA73BUL,
	0x3C6EF372FE94F82BUL, 0xA54FF53A5F1D36F1UL,
	0x510E527FADE682D1UL, 0x9B05688C2B3E6C1FUL,
	0x1F83D9ABFB41BD6BUL, 0x5BE0CD19137E2179UL
};

#define SHA3_STEP(A, B, C, D, E, F, G, H, idx0, idx1)	do { \
	ulong tmp = H + BSG5_1(E) + CH(E, F, G) + K512[idx0] + W[idx1]; \
	D += tmp; \
	H = tmp + BSG5_0(A) + MAJ(A, B, C); \
} while(0)

#define BSG5_0(x)      (FAST_ROTL64_HI(as_uint2(x), 36) ^ FAST_ROTL64_LO(as_uint2(x), 30) ^ FAST_ROTL64_LO(as_uint2(x), 25))
#define BSG5_1(x)      (FAST_ROTL64_HI(as_uint2(x), 50) ^ FAST_ROTL64_HI(as_uint2(x), 46) ^ FAST_ROTL64_LO(as_uint2(x), 23))
#define SSG5_0(x)      (FAST_ROTL64_HI(as_uint2(x), 63) ^ as_ulong(as_uchar8(x).s12345670) ^ SPH_T64((x) >> 7))
#define SSG5_1(x)      (FAST_ROTL64_HI(as_uint2(x), 45) ^ FAST_ROTL64_LO(as_uint2(x), 3) ^ SPH_T64((x) >> 6))

 __kernel void search27(__global hash_t *hashes, __global uint *BranchNonces)
{
	hashes += BranchNonces[get_global_id(0) - get_global_offset(0)];
	ulong W[16] = { 0UL };
	
	for(int i = 0; i < 8; ++i) W[i] = SWAP8(hashes->h8[i]);
	
	W[8] = 0x8000000000000000UL;
	W[15] = 0x0000000000000200UL;
	
	ulong8 State = vload8(0, SHA512_INIT);
	
	/*
	#pragma unroll 16
	for(int i = 16; i < 80; i++)
		W[i] = SPH_T64(SSG5_1(W[i - 2]) + W[i - 7] + SSG5_0(W[i - 15]) + W[i - 16]);
	
	#pragma unroll 8
	for(int i = 0; i < 80; ++i)
	{
		SHA3_STEP(State.s0, State.s1, State.s2, State.s3, State.s4, State.s5, State.s6, State.s7, i);
		State = shuffle(State, (ulong8)(7, 0, 1, 2, 3, 4, 5, 6));
	}
	*/
	
	#pragma unroll
	for(int i = 0; i < 80; i += 16)
	{
		ulong WBak[32];
		
		#pragma unroll
		for(int x = 0; x < 16; ++x)
		{
			SHA3_STEP(State.s0, State.s1, State.s2, State.s3, State.s4, State.s5, State.s6, State.s7, i + x, x);
			State = shuffle(State, (ulong8)(7, 0, 1, 2, 3, 4, 5, 6));
		}
		
		((ulong16 *)WBak)[0] = ((ulong16 *)W)[0];
		
		#pragma unroll
		for(int x = 16; x < 32; ++x)
			W[x - 16] = WBak[x] = SSG5_1(WBak[x - 2]) + WBak[x - 7] + SSG5_0(WBak[x - 15]) + WBak[x - 16];
		
		//((ulong16 *)W)[0] = ((ulong16 *)WBak)[1];
	}
	
	State += ((__constant ulong8 *)SHA512_INIT)[0];
	vstore8(VSWAP8(State), 0, hashes->h8);		
}


__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search28(__global hash_t *hashes)
{
	const uint idx = get_global_id(0) - get_global_offset(0);
	
	__local ulong T0[256], T1[256], T2[256], T3[256];
	__global hash_t *hash = hashes + idx;
	
	for(int i = get_local_id(0); i < 256; i += WORKSIZE)
	{
		const ulong tmp = T0_G[i];
		T0[i] = tmp;
		T1[i] = rotate(tmp, 8UL);
		T2[i] = rotate(tmp, 16UL);
		T3[i] = rotate(tmp, 24UL);
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	ulong M[16] = { 0 }, G[16];
	
	#pragma unroll
	for(int i = 0; i < 8; ++i) M[i] = hash->h8[i];
	
	M[8] = 0x80UL;
	M[15] = 0x0100000000000000UL;
	
	#pragma unroll
	for(int i = 0; i < 16; ++i) G[i] = M[i];
	
	G[15] ^= 0x0002000000000000UL;
	
	//#pragma unroll 2
	for(int i = 0; i < 14; ++i)
	{
		ulong H[16], H2[16];
		
		#pragma unroll
		for(int x = 0; x < 16; ++x)
		{
			H[x] = G[x] ^ PC64(x << 4, i);
			H2[x] = M[x] ^ QC64(x << 4, i);
		}
		
		GROESTL_RBTT(G[0], H, 0, 1, 2, 3, 4, 5, 6, 11);
		GROESTL_RBTT(G[1], H, 1, 2, 3, 4, 5, 6, 7, 12);
		GROESTL_RBTT(G[2], H, 2, 3, 4, 5, 6, 7, 8, 13);
		GROESTL_RBTT(G[3], H, 3, 4, 5, 6, 7, 8, 9, 14);
		GROESTL_RBTT(G[4], H, 4, 5, 6, 7, 8, 9, 10, 15);
		GROESTL_RBTT(G[5], H, 5, 6, 7, 8, 9, 10, 11, 0);
		GROESTL_RBTT(G[6], H, 6, 7, 8, 9, 10, 11, 12, 1);
		GROESTL_RBTT(G[7], H, 7, 8, 9, 10, 11, 12, 13, 2);
		GROESTL_RBTT(G[8], H, 8, 9, 10, 11, 12, 13, 14, 3);
		GROESTL_RBTT(G[9], H, 9, 10, 11, 12, 13, 14, 15, 4);
		GROESTL_RBTT(G[10], H, 10, 11, 12, 13, 14, 15, 0, 5);
		GROESTL_RBTT(G[11], H, 11, 12, 13, 14, 15, 0, 1, 6);
		GROESTL_RBTT(G[12], H, 12, 13, 14, 15, 0, 1, 2, 7);
		GROESTL_RBTT(G[13], H, 13, 14, 15, 0, 1, 2, 3, 8);
		GROESTL_RBTT(G[14], H, 14, 15, 0, 1, 2, 3, 4, 9);
		GROESTL_RBTT(G[15], H, 15, 0, 1, 2, 3, 4, 5, 10);
		
		GROESTL_RBTT(M[0], H2, 1, 3, 5, 11, 0, 2, 4, 6);
		GROESTL_RBTT(M[1], H2, 2, 4, 6, 12, 1, 3, 5, 7);
		GROESTL_RBTT(M[2], H2, 3, 5, 7, 13, 2, 4, 6, 8);
		GROESTL_RBTT(M[3], H2, 4, 6, 8, 14, 3, 5, 7, 9);
		GROESTL_RBTT(M[4], H2, 5, 7, 9, 15, 4, 6, 8, 10);
		GROESTL_RBTT(M[5], H2, 6, 8, 10, 0, 5, 7, 9, 11);
		GROESTL_RBTT(M[6], H2, 7, 9, 11, 1, 6, 8, 10, 12);
		GROESTL_RBTT(M[7], H2, 8, 10, 12, 2, 7, 9, 11, 13);
		GROESTL_RBTT(M[8], H2, 9, 11, 13, 3, 8, 10, 12, 14);
		GROESTL_RBTT(M[9], H2, 10, 12, 14, 4, 9, 11, 13, 15);
		GROESTL_RBTT(M[10], H2, 11, 13, 15, 5, 10, 12, 14, 0);
		GROESTL_RBTT(M[11], H2, 12, 14, 0, 6, 11, 13, 15, 1);
		GROESTL_RBTT(M[12], H2, 13, 15, 1, 7, 12, 14, 0, 2);
		GROESTL_RBTT(M[13], H2, 14, 0, 2, 8, 13, 15, 1, 3);
		GROESTL_RBTT(M[14], H2, 15, 1, 3, 9, 14, 0, 2, 4);
		GROESTL_RBTT(M[15], H2, 0, 2, 4, 10, 15, 1, 3, 5);
	}
	
	#pragma unroll
	for(int i = 0; i < 16; ++i) G[i] ^= M[i];
			
	G[15] ^= 0x0002000000000000UL;
	
	((ulong8 *)M)[0] = ((ulong8 *)G)[1];
	
	//#pragma unroll 2
	for(int i = 0; i < 14; ++i)
	{
		ulong H[16];
		
		#pragma unroll
		for(int x = 0; x < 16; ++x)
			H[x] = G[x] ^ PC64(x << 4, i); //G[x] ^ as_uint2(PC64((x << 4), i)); //(uint2)(G[x].s0 ^ ((x << 4) | i), G[x].s1);
			
		GROESTL_RBTT(G[0], H, 0, 1, 2, 3, 4, 5, 6, 11);
		GROESTL_RBTT(G[1], H, 1, 2, 3, 4, 5, 6, 7, 12);
		GROESTL_RBTT(G[2], H, 2, 3, 4, 5, 6, 7, 8, 13);
		GROESTL_RBTT(G[3], H, 3, 4, 5, 6, 7, 8, 9, 14);
		GROESTL_RBTT(G[4], H, 4, 5, 6, 7, 8, 9, 10, 15);
		GROESTL_RBTT(G[5], H, 5, 6, 7, 8, 9, 10, 11, 0);
		GROESTL_RBTT(G[6], H, 6, 7, 8, 9, 10, 11, 12, 1);
		GROESTL_RBTT(G[7], H, 7, 8, 9, 10, 11, 12, 13, 2);
		GROESTL_RBTT(G[8], H, 8, 9, 10, 11, 12, 13, 14, 3);
		GROESTL_RBTT(G[9], H, 9, 10, 11, 12, 13, 14, 15, 4);
		GROESTL_RBTT(G[10], H, 10, 11, 12, 13, 14, 15, 0, 5);
		GROESTL_RBTT(G[11], H, 11, 12, 13, 14, 15, 0, 1, 6);
		GROESTL_RBTT(G[12], H, 12, 13, 14, 15, 0, 1, 2, 7);
		GROESTL_RBTT(G[13], H, 13, 14, 15, 0, 1, 2, 3, 8);
		GROESTL_RBTT(G[14], H, 14, 15, 0, 1, 2, 3, 4, 9);
		GROESTL_RBTT(G[15], H, 15, 0, 1, 2, 3, 4, 5, 10);
	}
	
	vstore8((((ulong8 *)M)[0] ^ ((ulong8 *)G)[1]), 0, hash->h8);
	mem_fence(CLK_GLOBAL_MEM_FENCE);
}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search29(__global hash_t *hashes, __global uint *Branch15Nonces, __global uint *Branch16Nonces, const ulong FoundIdx)
{
	hashes += get_global_id(0) - get_global_offset(0);
	ulong W[16] = { 0UL };
	
	for(int i = 0; i < 8; ++i) W[i] = SWAP8(hashes->h8[i]);
	
	W[8] = 0x8000000000000000UL;
	W[15] = 0x0000000000000200UL;
	
	ulong8 State = vload8(0, SHA512_INIT);
	
	/*
	#pragma unroll 16
	for(int i = 16; i < 80; i++)
		W[i] = SPH_T64(SSG5_1(W[i - 2]) + W[i - 7] + SSG5_0(W[i - 15]) + W[i - 16]);
	
	#pragma unroll 8
	for(int i = 0; i < 80; ++i)
	{
		SHA3_STEP(State.s0, State.s1, State.s2, State.s3, State.s4, State.s5, State.s6, State.s7, i);
		State = shuffle(State, (ulong8)(7, 0, 1, 2, 3, 4, 5, 6));
	}
	*/
	
	#pragma unroll
	for(int i = 0; i < 80; i += 16)
	{
		ulong WBak[32];
		
		#pragma unroll
		for(int x = 0; x < 16; ++x)
		{
			SHA3_STEP(State.s0, State.s1, State.s2, State.s3, State.s4, State.s5, State.s6, State.s7, i + x, x);
			State = shuffle(State, (ulong8)(7, 0, 1, 2, 3, 4, 5, 6));
		}
		
		((ulong16 *)WBak)[0] = ((ulong16 *)W)[0];
		
		#pragma unroll
		for(int x = 16; x < 32; ++x)
			W[x - 16] = WBak[x] = SSG5_1(WBak[x - 2]) + WBak[x - 7] + SSG5_0(WBak[x - 15]) + WBak[x - 16];
		
		//((ulong16 *)W)[0] = ((ulong16 *)WBak)[1];
	}
	
	State += ((__constant ulong8 *)SHA512_INIT)[0];
	vstore8(VSWAP8(State), 0, hashes->h8);	
  
	if(hashes->h4[0] & 24) Branch15Nonces[atomic_inc(Branch15Nonces + FoundIdx)] = get_global_id(0) - get_global_offset(0);
	else Branch16Nonces[atomic_inc(Branch16Nonces + FoundIdx)] = get_global_id(0) - get_global_offset(0);
}


  //dec = ((hash.h4[0] & 24) != 0);
 __kernel void search30(__global hash_t *hashes, __global uint *BranchNonces)
  {
// Haval
  sph_u32 s0 = SPH_C32(0x243F6A88);
  sph_u32 s1 = SPH_C32(0x85A308D3);
  sph_u32 s2 = SPH_C32(0x13198A2E);
  sph_u32 s3 = SPH_C32(0x03707344);
  sph_u32 s4 = SPH_C32(0xA4093822);
  sph_u32 s5 = SPH_C32(0x299F31D0);
  sph_u32 s6 = SPH_C32(0x082EFA98);
  sph_u32 s7 = SPH_C32(0xEC4E6C89);

  sph_u32 X_var[32];
  
  hashes += BranchNonces[get_global_id(0) - get_global_offset(0)];

  for (int i = 0; i < 16; i++)
    X_var[i] = hashes->h4[i];

  X_var[16] = 0x00000001U;
  X_var[17] = 0x00000000U;
  X_var[18] = 0x00000000U;
  X_var[19] = 0x00000000U;
  X_var[20] = 0x00000000U;
  X_var[21] = 0x00000000U;
  X_var[22] = 0x00000000U;
  X_var[23] = 0x00000000U;
  X_var[24] = 0x00000000U;
  X_var[25] = 0x00000000U;
  X_var[26] = 0x00000000U;
  X_var[27] = 0x00000000U;
  X_var[28] = 0x00000000U;
  X_var[29] = 0x40290000U;
  X_var[30] = 0x00000200U;
  X_var[31] = 0x00000000U;

#define A(x) X_var[x]
  CORE5(A);

  hashes->h4[0] = s0;
  hashes->h4[1] = s1;
  hashes->h4[2] = s2;
  hashes->h4[3] = s3;
  hashes->h4[4] = s4;
  hashes->h4[5] = s5;
  hashes->h4[6] = s6;
  hashes->h4[7] = s7;
  hashes->h8[4] = 0;
  hashes->h8[5] = 0;
  hashes->h8[6] = 0;
  hashes->h8[7] = 0;
}

 __kernel void search31(__global hash_t *hashes, __global uint *BranchNonces)
{
	ulong8 n, h, st;
	__local ulong LT0[256];
	
	hashes += BranchNonces[get_global_id(0) - get_global_offset(0)];
	
	for(int i = get_local_id(0); i < 256; i += get_local_size(0)) LT0[i] = WPL_T0_G[i];
	
	st = n = vload8(0, hashes->h8);
	h = 0;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	n = WhirlpoolRound(n, h, LT0);
	
	st ^= n;
	n = (ulong8)(0x80UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0x2000000000000UL) ^ st;
	h = st;
	
	n = WhirlpoolRound(n, h, LT0);
	
	st ^= n;
	st.s0 ^= 0x80UL;
	st.s7 ^= 0x2000000000000UL;
	
	vstore8(st, 0, hashes->h8);
	
	barrier(CLK_GLOBAL_MEM_FENCE);
}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search32(__global hash_t *hashes, __global uint *output, const ulong target)
{
	ulong msg[16] = { 0 };
	hashes += get_global_id(0) - get_global_offset(0);	

	#pragma unroll
	for(int i = 0; i < 8; ++i) msg[i] = hashes->h8[i];

	msg[8] = 0x80UL;
	msg[15] = 512UL;
	
	#pragma unroll
	for(int i = 0; i < 2; ++i)
	{
		ulong h[16];
		for(int x = 0; x < 16; ++x) h[x] = ((i) ? BMW512_FINAL[x] : BMW512_IV[x]);
		BMW_Compression(msg, h);
	}
	
	#pragma unroll
	for(int i = 0; i < 8; ++i) hashes->h8[i] = msg[i + 8];
	
	if(hashes->h8[3] <= target) output[atomic_inc(output + 0xFF)] = SWAP4((uint)get_global_id(0));
}

#endif // HMQ1725_CL
