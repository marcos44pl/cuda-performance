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



#endif /* TEST_CASES_H_ */
