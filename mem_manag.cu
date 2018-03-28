#include <cuda_runtime.h>


#include "headers.h"


void testRead(uchar* data,uint size)
{
	int c = size / sizeof(uchar);
	for(int i = 0; i < c; ++i);
		*data = 0;
}


uchar* createStdMem(uchar* data,uint size)
{
	uchar* d_data;
	cudaMalloc((void**)&d_data,size*2);
	cudaCheckError();
	cudaMemcpy(d_data,data,size,cudaMemcpyHostToDevice);
	cudaCheckError();
	return d_data;
}

uchar* createUMem(uchar* data,uint size)
{
	uchar* um_data;
	cudaMallocManaged(&um_data, size * 2);
	memcpy(um_data,data,size);
	cudaCheckError();
	return um_data;
}

uchar* createUMemOpt(uchar* data,uint size)
{
	int device =-1;
	uchar* um_data;
	um_data = createUMem(data,size);
	cudaGetDevice(&device);
	cudaMemAdvise(um_data,size,cudaMemAdviseSetReadMostly,device);
	cudaCheckError();
	cudaMemPrefetchAsync(um_data,size,device,NULL);
	cudaCheckError();
	return um_data;
}


uchar* copyStdMemBack(uchar* d_data,uint size)
{
	uchar* h_data = new uchar[size];
	cudaMemcpy(h_data,d_data,size,cudaMemcpyDeviceToHost);
	cudaCheckError();
	testRead(h_data,size);
	return h_data;
}

uchar* copyMock(uchar* d_data,uint size) // We don't need to copy mem, as it is UM
{
	testRead(d_data,size);
	return d_data;
}

