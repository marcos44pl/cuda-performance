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

typedef uchar* (*createMemFunc)(uchar* ,ulong );
typedef uchar* (*copyMemAfterFunc)(uchar* , ulong );
typedef void   (*execKernel)(uchar* , uchar* ,size_t , size_t );

typedef void (*kernelPtr)(uchar *, uchar *,uint, uint ,uint ,float);
typedef void (*freeMem)(uchar* ,uchar* ,uchar* );

#endif /* TYPES_H_ */
