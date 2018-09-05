#include <cuda_runtime.h>
#include <stdlib.h>
#include <cuda_profiler_api.h>
#include <vector>

#include "ImageManager.h"
#include "headers.h"
#include "Timer.h"
#include "test_cases.h"
#include "utils.h"
#include "computation.h"

std::map<std::string,execKernel> kernel_map =
	{
		{"rotation_global_mem", &rotation_global_mem},
		{"rotation_shared_mem", &rotation_shared_mem},
		{"sobel_filter_coalesc", &sobel_filter_coalesc},
		{"sobel_filter_non_coalesc", &sobel_filter_non_coalesc}
	};

Fraction* execDevice(StartArgs args);
void testFluidSimulations(std::vector<int> mods);
void testImageProcessing();
void testImageProcessingBasic(std::vector<int> size);


int main(int argc, char *argv[])
{
	initCuda();
	std::vector<int> cnnIters = { 100, 200, 800, 1600, 3200 };
	std::vector<int> imgSizes = { 30000 };
	std::vector<int> copySimulCount = {1,2,5,10,20,100};

	testCNNprecision(cnnIters);
	testImageProcessingBasic(imgSizes);
    testImageProcessing();
	testFluidSimulations(copySimulCount);
	Timer::getInstance().printResults();
	cudaDeviceSynchronize();
    cudaProfilerStop();
    cudaDeviceReset();
    printf("end\n");

	return 0;
}

void testCNNprecision(std::vector<int> cnnIters)
{
	for(auto iter : cnnIters)
	{
		testFl16FullyConnectedFwdCudaNN(iter);
		testFl16Cudnn(iter);
		testFl16PoolCudaNN(iter);
		testFl16ConvCudaNN(iter);
		cudaDeviceSynchronize();
	}
}

void testImageProcessingBasic(std::vector<int> sizes)
{
	ImageManager image;
	for(auto size : sizes)
	{
		image.createEmpty(size,size);
		for(auto const& pair : kernel_map)
		{
			for(int i = 0; i < 3;++i)
			{

				auto sizeStr = std::to_string(size);
				testCudaMemGeneric(image,pair.first + std::string(" UM std ") + sizeStr,
									createUMem,
									pair.second,
									copyMock,
									freeUM);
				testCudaMemGeneric(image,pair.first + std::string(" UM opt ") + sizeStr,
									createUMemOpt,
									pair.second,
									copyMock,
									freeUM);
				testCudaMemGeneric(image,pair.first + std::string(" MemCpy std ") + sizeStr,
									createStdMem,
									pair.second,
									copyStdMemBack,
									freeStd);
			}
		}
		image.clear();
	}
}
void testImageProcessing()
{
	for(int i = 0; i < 10;++i)
	{
		testSobelOversubUMOpt(true);
		testSobelOversubUMOpt(false);
		testSobelOversubMultiImgStd();
		testSobelOversubStd();
		testSobelOversubUM();
		testSobelStreamUM(false);
		testSobelStreamUM(true);
		testSobelStreamStd();
	    cudaDeviceReset();
	}
}

void testFluidSimulations(std::vector<int> mods)
{
	for(int i = 0; i < 5;++i)
		for(auto mod : mods)
		{
			testFluidSimStd(mod);
			testFluidSimUM(mod);
			testFluidSimUM(mod,false);
		}
}
