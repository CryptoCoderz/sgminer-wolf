/*-
 * Copyright 2014 phm
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#include "config.h"
#include "miner.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"
 /** ADDED FOR HMQ1725 */
#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"
#include "sph/sph_hamsi.h"
#include "sph/sph_fugue.h"
#include "sph/sph_shabal.h"
#include "sph/sph_whirlpool.h"
#include "sph/sph_sha2.h"
#include "sph/sph_haval.h"

/*
 * Encode a length len/4 vector of (uint32_t) into a length len vector of
 * (unsigned char) in big-endian form.  Assumes len is a multiple of 4.
 */
static inline void
be32enc_vect(uint32_t *dst, const uint32_t *src, uint32_t len)
{
	uint32_t i;

	for (i = 0; i < len; i++)
		dst[i] = htobe32(src[i]);
}


#ifdef __APPLE_CC__
static
#endif
inline void hmq1725hash(void *state, const void *input)
{
    sph_blake512_context     ctx_blake;
    sph_bmw512_context       ctx_bmw;
    sph_groestl512_context   ctx_groestl;
    sph_jh512_context        ctx_jh;
    sph_keccak512_context    ctx_keccak;
    sph_skein512_context     ctx_skein;

   sph_luffa512_context     ctx_luffa;
   sph_cubehash512_context  ctx_cubehash;
   sph_shavite512_context   ctx_shavite;
   sph_simd512_context      ctx_simd;
   sph_echo512_context      ctx_echo;
   sph_hamsi512_context     ctx_hamsi;
   sph_fugue512_context     ctx_fugue;
   sph_shabal512_context    ctx_shabal;
   sph_whirlpool_context    ctx_whirlpool;
   sph_sha512_context       ctx_sha2;
   sph_haval256_5_context   ctx_haval;

    uint32_t mask = 24;
    uint32_t zero = 0;

    uint32_t hashA[16], hashB[16];


  sph_bmw512_init(&ctx_bmw);
    sph_bmw512 (&ctx_bmw, input, 80);
    sph_bmw512_close (&ctx_bmw, hashA);  //0


    sph_whirlpool_init(&ctx_whirlpool);
    sph_whirlpool (&ctx_whirlpool, hashA, 64);    //0
    sph_whirlpool_close(&ctx_whirlpool, hashB);   //1


    if ((hashB[0] & mask) != zero)   //1
    {
        sph_groestl512_init(&ctx_groestl);
        sph_groestl512 (&ctx_groestl, hashB, 64); //1
        sph_groestl512_close(&ctx_groestl, hashA); //2
    }
    else
    {
        sph_skein512_init(&ctx_skein);
        sph_skein512 (&ctx_skein, hashB, 64); //1
        sph_skein512_close(&ctx_skein, hashA); //2
    }


    sph_jh512_init(&ctx_jh);
    sph_jh512 (&ctx_jh, hashA, 64); //2
    sph_jh512_close(&ctx_jh, hashB); //3

    sph_keccak512_init(&ctx_keccak);
    sph_keccak512 (&ctx_keccak, hashB, 64); //3
    sph_keccak512_close(&ctx_keccak, hashA); //4

    if ((hashA[0] & mask) != zero) //4
    {
        sph_blake512_init(&ctx_blake);
        sph_blake512 (&ctx_blake, hashA, 64); //
        sph_blake512_close(&ctx_blake, hashB); //5
    }
    else
    {
        sph_bmw512_init(&ctx_bmw);
        sph_bmw512 (&ctx_bmw, hashA, 64); //4
        sph_bmw512_close(&ctx_bmw, hashB);   //5
    }

    sph_luffa512_init(&ctx_luffa);
    sph_luffa512 (&ctx_luffa,hashB, 64); //5
    sph_luffa512_close(&ctx_luffa, hashA); //6

    sph_cubehash512_init(&ctx_cubehash);
    sph_cubehash512 (&ctx_cubehash, hashA, 64); //6
    sph_cubehash512_close(&ctx_cubehash, hashB); //7

    if ((hashB[0] & mask) != zero) //7
    {
        sph_keccak512_init(&ctx_keccak);
        sph_keccak512 (&ctx_keccak, hashB, 64); //
        sph_keccak512_close(&ctx_keccak, hashA); //8
    }
    else
    {
        sph_jh512_init(&ctx_jh);
        sph_jh512 (&ctx_jh, hashB, 64); //7
        sph_jh512_close(&ctx_jh, hashA); //8
    }


    sph_shavite512_init(&ctx_shavite);
    sph_shavite512 (&ctx_shavite,hashA, 64); //5
    sph_shavite512_close(&ctx_shavite, hashB); //6

    sph_simd512_init(&ctx_simd);
    sph_simd512 (&ctx_simd, hashB, 64); //6
    sph_simd512_close(&ctx_simd, hashA); //7

    if ((hashA[0] & mask) != zero) //4
    {
        sph_whirlpool_init(&ctx_whirlpool);
        sph_whirlpool (&ctx_whirlpool, hashA, 64); //
        sph_whirlpool_close(&ctx_whirlpool, hashB); //5
    }
    else
    {
        sph_haval256_5_init(&ctx_haval);
        sph_haval256_5 (&ctx_haval, hashA, 64); //4
        sph_haval256_5_close(&ctx_haval, hashB);   //5
	memset(&hashB[8], 0, 32);
    }

    sph_echo512_init(&ctx_echo);
    sph_echo512 (&ctx_echo,hashB, 64); //5
    sph_echo512_close(&ctx_echo, hashA); //6

    sph_blake512_init(&ctx_blake);
    sph_blake512 (&ctx_blake, hashA, 64); //6
    sph_blake512_close(&ctx_blake, hashB); //7

    if ((hashB[0] & mask) != zero) //7
    {
        sph_shavite512_init(&ctx_shavite);
        sph_shavite512 (&ctx_shavite, hashB, 64); //
        sph_shavite512_close(&ctx_shavite, hashA); //8
    }
    else
    {
        sph_luffa512_init(&ctx_luffa);
        sph_luffa512 (&ctx_luffa, hashB, 64); //7
        sph_luffa512_close(&ctx_luffa, hashA); //8
    }


    sph_hamsi512_init(&ctx_hamsi);
    sph_hamsi512 (&ctx_hamsi,hashA, 64); //5
    sph_hamsi512_close(&ctx_hamsi, hashB); //6

    sph_fugue512_init(&ctx_fugue);
    sph_fugue512 (&ctx_fugue, hashB, 64); //6
    sph_fugue512_close(&ctx_fugue, hashA); //7


    if ((hashA[0] & mask) != zero) //4
    {
        sph_echo512_init(&ctx_echo);
        sph_echo512 (&ctx_echo, hashA, 64); //
        sph_echo512_close(&ctx_echo, hashB); //5
    }
    else
    {
        sph_simd512_init(&ctx_simd);
        sph_simd512 (&ctx_simd, hashA, 64); //4
        sph_simd512_close(&ctx_simd, hashB);   //5
    }

    sph_shabal512_init(&ctx_shabal);
    sph_shabal512 (&ctx_shabal,hashB, 64); //5
    sph_shabal512_close(&ctx_shabal, hashA); //6

    sph_whirlpool_init(&ctx_whirlpool);
    sph_whirlpool (&ctx_whirlpool, hashA, 64); //6
    sph_whirlpool_close(&ctx_whirlpool, hashB); //7

   if ((hashB[0] & mask) != zero) //7
    {
        sph_fugue512_init(&ctx_fugue);
        sph_fugue512 (&ctx_fugue, hashB, 64); //
        sph_fugue512_close(&ctx_fugue, hashA); //8
    }
    else
    {
        sph_sha512_init(&ctx_sha2);
        sph_sha512 (&ctx_sha2, hashB, 64); //7
        sph_sha512_close(&ctx_sha2, hashA); //8
    }

    sph_groestl512_init(&ctx_groestl);
    sph_groestl512 (&ctx_groestl,hashA, 64); //5
    sph_groestl512_close(&ctx_groestl, hashB); //6

    sph_sha512_init(&ctx_sha2);
    sph_sha512 (&ctx_sha2, hashB, 64); //6
    sph_sha512_close(&ctx_sha2, hashA); //7


    if ((hashA[0] & mask) != zero) //4
    {
        sph_haval256_5_init(&ctx_haval);
        sph_haval256_5 (&ctx_haval, hashA, 64); //
        sph_haval256_5_close(&ctx_haval, hashB); //5
	memset(&hashB[8], 0, 32);
    }
    else
    {
        sph_whirlpool_init(&ctx_whirlpool);
        sph_whirlpool (&ctx_whirlpool, hashA, 64); //4
        sph_whirlpool_close(&ctx_whirlpool, hashB);   //5
    }


    sph_bmw512_init(&ctx_bmw);
    sph_bmw512 (&ctx_bmw,hashB, 64); //5
    sph_bmw512_close(&ctx_bmw, hashA); //6


	memcpy(state, hashA, 32);

}

static const uint32_t diff1targ = 0x0000ffff;


/* Used externally as confirmation of correct OCL code */
int hmq1725_test(unsigned char *pdata, const unsigned char *ptarget, uint32_t nonce)
{
	uint32_t tmp_hash7, Htarg = le32toh(((const uint32_t *)ptarget)[7]);
	uint32_t data[20], ohash[8];

	be32enc_vect(data, (const uint32_t *)pdata, 19);
	data[19] = htobe32(nonce);
	hmq1725hash(ohash, data);
	tmp_hash7 = be32toh(ohash[7]);

	applog(LOG_DEBUG, "htarget %08lx diff1 %08lx hash %08lx",
				(long unsigned int)Htarg,
				(long unsigned int)diff1targ,
				(long unsigned int)tmp_hash7);
	if (tmp_hash7 > diff1targ)
		return -1;
	if (tmp_hash7 > Htarg)
		return 0;
	return 1;
}

void hmq1725_regenhash(struct work *work)
{
        uint32_t data[20];
        uint32_t *nonce = (uint32_t *)(work->data + 76);
        uint32_t *ohash = (uint32_t *)(work->hash);

        be32enc_vect(data, (const uint32_t *)work->data, 19);
        data[19] = htobe32(*nonce);
        hmq1725hash(ohash, data);
}

bool scanhash_hmq1725(struct thr_info *thr, const unsigned char __maybe_unused *pmidstate,
		     unsigned char *pdata, unsigned char __maybe_unused *phash1,
		     unsigned char __maybe_unused *phash, const unsigned char *ptarget,
		     uint32_t max_nonce, uint32_t *last_nonce, uint32_t n)
{
	uint32_t *nonce = (uint32_t *)(pdata + 76);
	uint32_t data[20];
	uint32_t tmp_hash7;
	uint32_t Htarg = le32toh(((const uint32_t *)ptarget)[7]);
	bool ret = false;

	be32enc_vect(data, (const uint32_t *)pdata, 19);

	while(1) {
		uint32_t ostate[8];

		*nonce = ++n;
		data[19] = (n);
		hmq1725hash(ostate, data);
		tmp_hash7 = (ostate[7]);

		applog(LOG_INFO, "data7 %08lx",
					(long unsigned int)data[7]);

		if (unlikely(tmp_hash7 <= Htarg)) {
			((uint32_t *)pdata)[19] = htobe32(n);
			*last_nonce = n;
			ret = true;
			break;
		}

		if (unlikely((n >= max_nonce) || thr->work_restart)) {
			*last_nonce = n;
			break;
		}
	}

	return ret;
}



