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


void sobel_filter_coalesc(uchar* in, uchar* out,size_t width, size_t height);
void sobel_filter_non_coalesc(uchar* in, uchar* out,size_t width, size_t height);
void rotation_global_mem(uchar* in, uchar* out,size_t width, size_t height);
void rotation_shared_mem(uchar* in, uchar* out,size_t width, size_t height);

uchar* createStdMem(uchar* data,uint size);
uchar* createUMem(uchar* data,uint size);
uchar* createUMemOpt(uchar* data,uint size);
uchar* copyStdMemBack(uchar* d_data,uint size);
uchar* copyMock(uchar* d_data,uint size);

void initCuda();
void exec_kernel(uchar* in, uchar* out,size_t width, size_t height,kernel kernel_ptr);

#endif /* FILTERS_H_ */
