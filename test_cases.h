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
void testOversubNaiveUM(kernelPtr kernel,bool withAdvise=true);
void testOversubUMOpt(kernelPtr kernel);
void testOversubMultiImgStd(kernelPtr kernel);
void testFluidSimUM(bool withAdvise = true);
void testFluidSimStd();
void testStreamImgProcessingStd(kernelPtr kernel);
void testStreamImgProcessingUm(kernelPtr kernel,std::string name,int imgCount,bool withAdvise= true);


#endif /* TEST_CASES_H_ */
