#include <cuda_runtime.h>


#include "headers.h"
#include "utils.h"

uchar* createStdMem(uchar* data,ulong size)
{
	uchar* d_data;
	cudaMalloc((void**)&d_data,size*2);
	cudaCheckError();
	if(data)
		cudaMemcpy(d_data,data,size,cudaMemcpyHostToDevice);
	cudaCheckError();
	return d_data;
}

uchar* createUMem(uchar* data,ulong size)
{
	uchar* um_data;
	cudaMallocManaged(&um_data, size * 2);
	cudaCheckError();
	if(data)
		memcpy(um_data,data,size);
	return um_data;
}

uchar* createUMemOpt(uchar* data,ulong size)
{
	int device =-1;
	uchar* um_data;
	um_data = createUMem(data,size);
	cudaGetDevice(&device);
	cudaCheckError();
	cudaMemAdvise(um_data,size,cudaMemAdviseSetReadMostly,device);
	cudaCheckError();
	cudaMemPrefetchAsync(um_data,size,device,NULL);
	cudaCheckError();
	return um_data;
}


uchar* copyStdMemBack(uchar* d_data,ulong size)
{
	uchar* h_data = new uchar[size];
	if (h_data == nullptr)
		printf("Out of memory\n");
	cudaMemcpy(h_data,d_data,size,cudaMemcpyDeviceToHost);
	cudaCheckError();
	testRead(h_data,size);
	return h_data;
}

uchar* copyMock(uchar* d_data,ulong size) // We don't need to copy mem, as it is UM
{
	testRead(d_data,size);
	return d_data;
}

void freeStd(uchar* d_in,uchar* d_out,uchar* h_out)
{
	cudaFree(d_in);
	cudaCheckError();
	delete h_out;
}


void freeUM(uchar* d_in,uchar* d_out,uchar* h_out)
{
	cudaFree(d_in);
	cudaCheckError();
}

