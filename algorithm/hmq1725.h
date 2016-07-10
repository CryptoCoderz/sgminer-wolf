#ifndef HMQ1725_H
#define HMQ1725_H

#include "miner.h"

extern int hmq1725_test(unsigned char *pdata, const unsigned char *ptarget,
			uint32_t nonce);
extern void hmq1725_regenhash(struct work *work);

#endif /* HMQ1725_H */
