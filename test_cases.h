/*
 * test_cases.h
 *
 *  Created on: Apr 8, 2018
 *      Author: mknap
 */

#ifndef TEST_CASES_H_
#define TEST_CASES_H_

#include "types.h"

class ImageManager;

void testOversubStd(kernelPtr kernel_ptr);
void testCudaMemGeneric(ImageManager&,std::string const&, createMemFunc,execKernel,copyMemAfterFunc,freeMem);
void testOversubUM(kernelPtr kernel);
void testFluidSimUM(bool withAdvise = true);
void testFluidSimStd();
void testStreamImgProcessingStd(kernelPtr kernel);
void testStreamImgProcessingUm(kernelPtr kernel,bool withAdvise= true);


#endif /* TEST_CASES_H_ */
