#include <cmath>
#include <stdlib.h>
#include <stdio.h>

#include "ImageManager.h"
#include "Timer.h"
#include "headers.h"
#include "config.h"

#define IMG_SIZE_GB 6

void testCudaMemGeneric(ImageManager& 		image,
						 std::string const& name,
						 createMemFunc      beforeFunc,
						 execKernel 		kernelFunc,
						 copyMemAfterFunc	afterFunc,
						 freeMem 			freeFunc)
{
	printf("%s\n",name.c_str());
	uchar* in,*out,*h_out;
	auto size = image.get_size();
	freeMemory();
	Timer::getInstance().start(name);
	in = beforeFunc(image.get_data(),size);
	printf("mem allocated\n");
	out = in + size;
	kernelFunc(in,out,image.get_width(),image.get_height());
	cudaDeviceSynchronize();
	cudaCheckError();
	printf("kernel finished\n");
	h_out = afterFunc(out,size);
	printf("coping mem\n");
	Timer::getInstance().stop(name);
	//image.save("result.jpg",h_out);
	freeFunc(in,out,h_out);
	printf("freeing mem\n");
}

void testOversubStd(kernelPtr kernel)
{
	const ulong kb = 1024;
	const ulong imgSize = kb * kb * kb * IMG_SIZE_GB;
	const uint channelNum = 3;
	const std::string name = "Oversubscription standard mem with overlapping";
	const double freeMemLeave = 0.05;
	const uint streamNum = 10;

	Timer::getInstance().start(name);

	uint width = sqrt(imgSize / (channelNum * sizeof(uchar)));
	uint height = width;
	ulong realSizeCh = width * height * sizeof(uchar);
	ulong realSize = realSizeCh * channelNum;
	auto freeMem = freeMemory();
	ulong sizetoDevMAlloc = ulong((double)freeMem * (1. - freeMemLeave));
	uint minStreamsNum = ceil((double)(realSize * 2) / sizetoDevMAlloc);
	uint realStreamNum = max(streamNum,minStreamsNum);
	ulong streamSize = realSize / realStreamNum;
	int streamHeight = streamSize / (width * sizeof(uchar));

	uchar* pitchedHMem,*dev_mem;
	cudaMallocHost((void**)&pitchedHMem,realSize);
	cudaCheckError();
	cudaMalloc((void**)&dev_mem,sizetoDevMAlloc);
	cudaCheckError();
	memset(pitchedHMem, 0, realSize);
	realStreamNum += 2;
	cudaStream_t* streams = new cudaStream_t[realStreamNum];
	cudaEvent_t* cpyEvents = new cudaEvent_t[realStreamNum];
	int h_prior;
	cudaDeviceGetStreamPriorityRange(nullptr,&h_prior);
	for(int i = 0; i <realStreamNum; ++i)
	{
		cudaStreamCreateWithPriority(&streams[i],cudaStreamDefault,h_prior++);
		cudaCheckError();
		cudaEventCreate(&cpyEvents[i]);
		cudaCheckError();
	}
	dim3 thsPerBlck(BATCH_W,BATCH_H);

	uchar* d_ptr = dev_mem;
	uchar* h_ptr = pitchedHMem;
	int streamsEnded = 0;
	for(int i = 0,ch_i = 0; i <realStreamNum; ++i,++ch_i)
	{
		ulong offsetH = ch_i * streamSize;
		ulong curStremSize = streamSize;
		uint curHeight = streamHeight;
		if(offsetH > realSizeCh)
		{
			ch_i = 0;
			d_ptr += realSizeCh;
			h_ptr += realSizeCh;
			offsetH = 0;
		}
		if(offsetH + streamSize > realSizeCh)
		{
			curStremSize = realSizeCh - offsetH;
			curHeight = curStremSize / (width * sizeof(uchar));
		}
		ulong offsetD = offsetH * 2;

		if(d_ptr + offsetD + 2 *  curStremSize > dev_mem + sizetoDevMAlloc) // if we exceeded gpu mem we back to the begin
		{
			d_ptr = dev_mem;
			offsetD = 0;
			for(int j = streamsEnded; j <= i; ++j)
			{
				cudaEventSynchronize(cpyEvents[j]); // we have to wait for the streams to get memory
				cudaCheckError();
				streamsEnded++;
			}
		}

		dim3 blckNum(width/(BATCH_W * PXL_PER_THD) + 1,curHeight/(BATCH_H) + 1);


		/*printf("Stream%d  Host off: %d Dev off: %p curHeight: %d curStreamS: %d\n",i,
								(offsetH)/(kb*kb),
								(void *)(d_ptr +offsetD),
								curHeight,
								curStremSize/(kb*kb));*/

		cudaMemcpyAsync(&d_ptr[offsetD], &h_ptr[offsetH], curStremSize, cudaMemcpyHostToDevice, streams[i]);
		cudaCheckError();
		kernel<<<blckNum, thsPerBlck, 0, streams[i]>>>(&d_ptr[offsetD],&d_ptr[offsetD + curStremSize],width,curHeight,PXL_PER_THD,IMAGE_SCALE);
		cudaCheckError();
		cudaMemcpyAsync(&h_ptr[offsetH], &d_ptr[offsetD + curStremSize], curStremSize, cudaMemcpyDeviceToHost, streams[i]);
		cudaEventRecord(cpyEvents[i], streams[i]);
		cudaCheckError();
	}
	for(int i = 0; i <realStreamNum; ++i)
	{
		cudaStreamDestroy(streams[i]); // we have to free the stream resources for the others
		cudaCheckError();
		cudaEventDestroy(cpyEvents[i]);
		cudaCheckError();
	}
	cudaDeviceSynchronize();
	testRead(pitchedHMem,realSize);

	cudaFree(dev_mem);
	cudaCheckError();
	cudaFreeHost(pitchedHMem);
	cudaCheckError();
	Timer::getInstance().stop(name);
}

void testOversubUM(kernelPtr kernel)
{
	const ulong kb = 1024;
	const ulong imgSize = kb * kb * kb * IMG_SIZE_GB;
	const uint channelNum = 3;
	uint width = sqrt(imgSize / (channelNum * sizeof(uchar)));
	uint height = width;
	ulong realSizeCh = width * height * sizeof(uchar);
	ulong realSize = realSizeCh * channelNum;


	const std::string name = "Oversubscription unified memory";
	Timer::getInstance().start(name);
	int device =-1;
	cudaGetDevice(&device);
	uchar* um_data = createUMem(nullptr,realSize);
	memset(um_data, 0, realSize);
	exec_kernel(um_data,um_data + realSize,width,height,kernel);
	cudaDeviceSynchronize();
	cudaCheckError();
	testRead(um_data + realSize,realSize);

	cudaFree(um_data);
	cudaCheckError();
	Timer::getInstance().stop(name);
}
