/*
 * filters.h
 *
 *  Created on: Mar 17, 2018
 *      Author: mknap
 */

#ifndef FILTERS_H_
#define FILTERS_H_

#include "types.h"

#define PXL_ID(x,y,w) ( (x) + ((y)*w) )
#define CHANNEL_NUM 3

void testSobelStreamStd();
void testSobelStreamUM(bool withAdvise);
void testSobelOversubUM();
void testSobelOversubStd();
void testSobelOversubUMOpt();
void testSobelOversubMultiImgStd();
void sobel_filter_coalesc(uchar* in, uchar* out,size_t width, size_t height);
void sobel_filter_non_coalesc(uchar* in, uchar* out,size_t width, size_t height);
void rotation_global_mem(uchar* in, uchar* out,size_t width, size_t height);
void rotation_shared_mem(uchar* in, uchar* out,size_t width, size_t height);
void testFl16Cudnn();
void testFl16ConvCudaNN();
void testFl16PoolCudaNN();
void testFl16FullyConnectedFwdCudaNN();

uchar* createStdMem(uchar* data,ulong size);
uchar* createUMem(uchar* data,ulong size);
uchar* createUMemOpt(uchar* data,ulong size);
uchar* copyStdMemBack(uchar* d_data,ulong size);
uchar* copyMock(uchar* d_data,ulong size);
void freeStd(uchar* d_in,uchar* d_out,uchar* h_out);
void freeUM(uchar* d_in,uchar* d_out,uchar* h_out);


void initCuda(int dev = 0);
void exec_kernel(uchar* in, uchar* out,size_t width, size_t height,kernelPtr kernel_ptr);
size_t freeMemory();
#endif /* FILTERS_H_ */
