/*
 * config.h
 *
 *  Created on: Mar 16, 2018
 *      Author: mknap
 */

#ifndef CONFIG_H_
#define CONFIG_H_


const uint BATCH_W = 16;
const uint BATCH_H = 16;
const uint RAD = 1;
const uint DIAM = RAD *2 + 1;
const uint FILT_S = DIAM * DIAM;
const uint BLCK_W = BATCH_W + 2 * RAD;
const uint BLCK_H = BATCH_H + 2 * RAD;
const uint PXL_PER_THD = 100;
const uint PXL_PER_THD_SH = 16;
const float IMAGE_SCALE = 1.;

#endif /* CONFIG_H_ */
