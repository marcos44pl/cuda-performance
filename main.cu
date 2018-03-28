#include <cuda_runtime.h>
#include <stdlib.h>
#include <cuda_profiler_api.h>

#include "ImageManager.h"
#include "headers.h"
#include "Timer.h"



std::map<std::string,execKernel> kernel_map =
	{
		{"rotation_global_mem", &rotation_global_mem},
		{"rotation_shared_mem", &rotation_shared_mem},
		{"sobel_filter_coalesc", &sobel_filter_coalesc},
		{"sobel_filter_non_coalesc", &sobel_filter_non_coalesc}
	};


void testCudaMemGeneric(ImageManager& 		image,
						 std::string const& name,
						 createMemFunc      beforeFunc,
						 execKernel 		kernelFunc,
						 copyMemAfterFunc	afterFunc)
{
	printf("%s\n",name.c_str());
	uchar* in,*out,*h_out;
	uint size = image.get_size();
	Timer::getInstance().start(name);
	in = beforeFunc(image.get_data(),size);
	out = in + size;
	kernelFunc(in,out,image.get_width(),image.get_height());
	cudaDeviceSynchronize();
	cudaCheckError();
	h_out = afterFunc(out,size);
	Timer::getInstance().stop(name);
	//image.save("result.jpg",h_out);
	cudaFree(in);
}

int main(int argc, char *argv[])
{
	initCuda();

	ImageManager image("huge.jpg");
	image.createEmpty(12000,12000);


	for(auto const& pair : kernel_map)
	{
		testCudaMemGeneric(image,pair.first + std::string(" UM std "),
							createUMem,
							pair.second,
							copyMock);
		testCudaMemGeneric(image,pair.first + std::string(" UM opt "),
							createUMemOpt,
							pair.second,
							copyMock);
		testCudaMemGeneric(image,pair.first + std::string(" MemCpy std "),
							createStdMem,
							pair.second,
							copyStdMemBack);
	}

	Timer::getInstance().printResults();
	// Free memory

    cudaProfilerStop();
    cudaDeviceReset();
    printf("end\n");
	return 0;
}
