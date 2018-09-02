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

StartArgs parsInputArguments(const int argc, char *argv[])
{
	StartArgs args;

	//default simulation settings
	args.NUM_OF_ITERATIONS = 10;
	args.X_SIZE = 100;
	args.Y_SIZE = 100;
	args.Z_SIZE = 100;
	args.type = deviceSimulationType::GLOBAL;
	args.print = false;
	return args;
}

int main(int argc, char *argv[])
{
	initCuda();
	printf("Testing UM optimalizations\n");
	std::vector<int> sizes = { 100, 200, 800, 1600, 3200 };
	/*for(auto size : sizes)
	{
		testFl16FullyConnectedFwdCudaNN(size);
		testFl16Cudnn(size);
		testFl16PoolCudaNN(size);
		testFl16ConvCudaNN(size);
		cudaDeviceSynchronize();
	}*/

	ImageManager image;
	sizes = { 500, 1000, 5000, 10000, 30000 };
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
    cudaDeviceReset();
	for(int i = 0; i < 3;++i)
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
	testFluidSimStd();
	testFluidSimUM();
	testFluidSimUM(false);
	Timer::getInstance().printResults();
	cudaDeviceSynchronize();
    cudaProfilerStop();
    cudaDeviceReset();
    printf("end\n");

	return 0;
}
