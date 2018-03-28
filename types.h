/*
 * types.h
 *
 *  Created on: Mar 16, 2018
 *      Author: mknap
 */

#ifndef TYPES_H_
#define TYPES_H_

#include <helper_cuda.h>

typedef unsigned char  uchar;

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

typedef uchar* (*createMemFunc)(uchar* h_data,uint size);
typedef uchar* (*copyMemAfterFunc)(uchar* d_data, uint size);
typedef void   (*execKernel)(uchar* in, uchar* out,size_t width, size_t height);

typedef void (*kernel)(uchar *, uchar *,uint, uint ,uint ,float);

#endif /* TYPES_H_ */
