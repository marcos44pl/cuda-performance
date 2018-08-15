#include <cuda_runtime.h>
#include <stdlib.h>
#include <cuda_profiler_api.h>

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
	/*ImageManager image;
	image.createEmpty(20000,20000);
	printf("Testing UM optimalizations\n");

	for(auto const& pair : kernel_map)
	{
		testCudaMemGeneric(image,pair.first + std::string(" UM std "),
							createUMem,
							pair.second,
							copyMock,
							freeUM);
		testCudaMemGeneric(image,pair.first + std::string(" UM opt "),
							createUMemOpt,
							pair.second,
							copyMock,
							freeUM);
		testCudaMemGeneric(image,pair.first + std::string(" MemCpy std "),
							createStdMem,
							pair.second,
							copyStdMemBack,
							freeStd);
	}
	image.clear();

	testSobelOversubStd();
	testSobelOversubUM();
	testFluidSimStd();
	testFluidSimUM();
	testFluidSimUM(false);
	testSobelStreamUM(false);
	testSobelStreamUM(true);
	testSobelStreamStd();
	testFl16Cudnn();*/
	testFl16ConvCudaNN();
	Timer::getInstance().printResults();
    cudaProfilerStop();
    cudaDeviceReset();
    printf("end\n");

	return 0;
}
