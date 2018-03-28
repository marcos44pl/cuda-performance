#include "headers.h"
#include "config.h"

void initCuda()
{
	int nDevices;
    int devCount = cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        cudaCheckError()
        printf("CUDA device [%s] has %d Multi-Processors\n",
               props.name, props.multiProcessorCount);
    }
}

void exec_kernel(uchar* in, uchar* out,size_t width, size_t height,kernel kernel_ptr)
{
	dim3 thsPerBlck(BATCH_W,BATCH_H);
	dim3 blckNum(width/(BATCH_W * PXL_PER_THD) + 1,height/(BATCH_H) + 1);
	auto in_ptr = in;
	auto out_ptr = out;
	auto ch_size = sizeof(uchar) * width * height;
	for(int i = 0; i < CHANNEL_NUM;++i)
	{
		kernel_ptr<<<blckNum, thsPerBlck>>>(in_ptr,out_ptr,(uint)width,(uint)height,PXL_PER_THD,IMAGE_SCALE);
		in_ptr += ch_size;
		out_ptr += ch_size;
	}
}
