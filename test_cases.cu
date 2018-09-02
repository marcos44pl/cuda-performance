#include <cmath>
#include <stdlib.h>
#include <stdio.h>

#include "ImageManager.h"
#include "Timer.h"
#include "headers.h"
#include "config.h"
#include "utils.h"

#define IMG_SIZE_GB 6
#define IMG_NUM 200
#define IMG_W 1920
#define IMG_H 1080

StartArgs parsInputArguments()
{
	StartArgs args;

	//default simulation settings
	args.NUM_OF_ITERATIONS = 100;
	args.X_SIZE = 200;
	args.Y_SIZE = 200;
	args.Z_SIZE = 200;
	args.type = deviceSimulationType::SHARED_3D_LAYER;
	args.print = false;
	return args;
}

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
	out = in + size;
	kernelFunc(in,out,image.get_width(),image.get_height());
	cudaDeviceSynchronize();
	cudaCheckError();
	h_out = afterFunc(out,size);
	Timer::getInstance().stop(name);
	//image.save("result.jpg",h_out);
	freeFunc(in,out,h_out);
}

void testStreamImgProcessingStd(kernelPtr kernel)
{
	std::string name = "Stream Image Processing Std";
    Timer::getInstance().start(name);
	const uint size = IMG_H * IMG_W * 3;
	const uint sizePerCh = IMG_H * IMG_W;
	const uint sizePerChBytes = IMG_H * IMG_W * sizeof(uchar);
	const int streamNum = 3;
	const ulong totalSize = sizeof(uchar) * IMG_NUM * size;
	cudaStream_t streams[streamNum];
	uchar* d_mem = createStdMem(nullptr,totalSize);
	uchar* h_mem;
	cudaMallocHost((void**)&h_mem,totalSize);
	memset(h_mem, 0, totalSize);
	uchar* c_res_mem = d_mem + size * IMG_NUM;

	for(int i = 0; i <streamNum; ++i)
	{
		cudaStreamCreate(&streams[i]);
		cudaCheckError();
	}

	dim3 thsPerBlck(BATCH_W,BATCH_H);
	dim3 blckNum(IMG_W/(BATCH_W * PXL_PER_THD) + 1,IMG_H/(BATCH_H) + 1);
	int totalChCount = IMG_NUM*3;
	for(int i =0; i< totalChCount;++i)
	{
		int s_id = i % streamNum;
		 cudaMemcpyAsync(&d_mem[i], &h_mem[i],sizePerChBytes,
		                 cudaMemcpyHostToDevice, streams[s_id]);
		cudaCheckError();
	}


	for(int i =0; i< totalChCount;++i)
	{
		int s_id = i % streamNum;
		kernel<<<blckNum, thsPerBlck, 0, streams[s_id]>>>(&d_mem[i * sizePerCh],&c_res_mem[i * sizePerCh],
														  IMG_W,IMG_H,PXL_PER_THD,IMAGE_SCALE);
		cudaCheckError();
	}

	for(int i =0; i< totalChCount;++i)
	{
		int s_id = i % streamNum;
		 cudaMemcpyAsync(&h_mem[i], &d_mem[i],sizePerChBytes,
				 	 	 cudaMemcpyDeviceToHost, streams[s_id]);
		cudaCheckError();
	}
	cudaDeviceSynchronize();
	cudaCheckError();
	testRead(h_mem,totalSize);
    Timer::getInstance().stop(name);
	cudaFree(d_mem);
	cudaCheckError();
	cudaFreeHost(h_mem);
	cudaCheckError();
}

void testStreamImgProcessingUm(kernelPtr kernel,std::string name,int imgCount,bool withAdvise)
{
	if(withAdvise)
		name += " Opt";
    Timer::getInstance().start(name);

	const ulong size = IMG_H * IMG_W * 3;
	const ulong totalSize = sizeof(uchar) * imgCount * size;
	const uint totalChCount = imgCount*3;
	const uint sizePerCh = IMG_H * IMG_W;
	const uint sizePerChBytes = sizeof(uchar) * sizePerCh;

	const int streamNum = 5;
	uchar* umem = createUMem(nullptr,totalSize);
	uchar* res_umem = umem + size * imgCount;
	memset(umem, 0, totalSize);
	cudaStream_t streams[streamNum];
	cudaEvent_t events[streamNum];
	for(int i = 0; i <streamNum; ++i)
	{
		cudaStreamCreate(&streams[i]);
		cudaCheckError();
		cudaEventCreate(&events[i]);
		cudaCheckError();
	}

	dim3 thsPerBlck(BATCH_W,BATCH_H);
	dim3 blckNum(IMG_W/(BATCH_W * PXL_PER_THD) + 1,IMG_H/(BATCH_H) + 1);

	int device =-1;
	cudaGetDevice(&device);
	cudaCheckError();
	if(withAdvise)
	{
		cudaMemPrefetchAsync(umem,sizePerChBytes, device,streams[1]);
		cudaCheckError();
		cudaEventRecord(events[0], streams[1]);
		cudaCheckError();
	}
	for(int i =0; i< totalChCount;++i)
	{
		int s_id = i % streamNum;
		int s1_id = (i + 1) % streamNum;
		cudaEventSynchronize(events[s_id]);
		cudaEventSynchronize(events[s1_id]);
		kernel<<<blckNum, thsPerBlck, 0, streams[s_id]>>>(umem + i * sizePerCh,res_umem + i * sizePerCh,
														  IMG_W,IMG_H,PXL_PER_THD,IMAGE_SCALE);
		cudaEventRecord(events[s_id], streams[s_id]);

		if(withAdvise)
		{
			if(i < totalChCount -1)
			{
				cudaStreamSynchronize(streams[s1_id]);
				cudaMemPrefetchAsync(umem + (i+1) * sizePerCh, sizePerChBytes, device, streams[s1_id]);
				cudaEventRecord(events[s1_id], streams[s1_id]);
			}
			cudaMemPrefetchAsync(res_umem + i * sizePerCh, sizePerChBytes, cudaCpuDeviceId, streams[s_id]);
		}
	}
	cudaDeviceSynchronize();
	cudaCheckError();
	testRead(res_umem,totalSize);
    Timer::getInstance().stop(name);
	cudaFree(umem);
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
	cudaDeviceSynchronize();
	for(int i = 0; i <realStreamNum; ++i)
	{
		cudaStreamDestroy(streams[i]); // we have to free the stream resources for the others
		cudaCheckError();
		cudaEventDestroy(cpyEvents[i]);
		cudaCheckError();
	}
	testRead(pitchedHMem,realSize);

	cudaFree(dev_mem);
	cudaCheckError();
	cudaFreeHost(pitchedHMem);
	cudaCheckError();
	Timer::getInstance().stop(name);
}

void testOversubNaiveUM(kernelPtr kernel,bool withAdvise /*=true*/)
{
	const ulong kb = 1024;
	const ulong imgSize = kb * kb * kb * IMG_SIZE_GB;
	const uint channelNum = 3;
	uint width = sqrt(imgSize / (channelNum * sizeof(uchar)));
	uint height = width;
	ulong realSizeCh = width * height * sizeof(uchar);
	ulong realSize = realSizeCh * channelNum;
	std::string name = "Oversubscription unified memory naive";
	if(withAdvise)
		name += " advised";
	Timer::getInstance().start(name);
	int device =-1;
	cudaGetDevice(&device);
	uchar* um_data;
	if(withAdvise)
		um_data = createUMemOpt(nullptr,realSize);
	else
		um_data = createUMem(nullptr,realSize);

	exec_kernel(um_data,um_data + realSize,width,height,kernel);
	cudaDeviceSynchronize();
	cudaCheckError();
	if(withAdvise)
	{
		cudaMemPrefetchAsync(um_data + realSize,realSize,cudaCpuDeviceId,NULL);
		cudaCheckError();
	}
	testRead(um_data + realSize,realSize);

	cudaFree(um_data);
	cudaCheckError();
	Timer::getInstance().stop(name);
}

void testOversubMultiImgStd(kernelPtr kernel)
{
	const std::string name = "Oversubscription multi_img std with overlapping";
	Timer::getInstance().start(name);

	const ulong kb = 1024;
	const ulong totalSize = kb * kb * kb * IMG_SIZE_GB;
	const uint channelNum = 3;
	const uint chSize = IMG_W * IMG_H;
	const uint chSizeBytes = chSize * sizeof(uchar);
	const uint size = chSize * channelNum;
	auto freeMem = freeMemory();
	float freeMemLeave = 0.05;
	ulong sizetoDevMAlloc = ulong((double)freeMem * (1. - freeMemLeave));
	uint imgCount = totalSize / (size);
	printf("ImgCount: %d\n",imgCount);
	uint chCount = imgCount * channelNum;
	uchar* d_mem,*d_result_mem,*h_mem,*h_result_mem;
	cudaMalloc(&d_mem,sizetoDevMAlloc);
	cudaCheckError();
	cudaMallocHost((void**)&h_mem,totalSize*2);
	cudaCheckError();
	memset(h_mem, 0, totalSize);
	h_result_mem = h_mem + size * imgCount;
	d_result_mem = d_mem + size * imgCount;
	uint chOnGpuCount = uint(sizetoDevMAlloc/ chSizeBytes);
	uint streamNum = 5;
	uint loopCount = chOnGpuCount / streamNum;

	cudaStream_t* streams = new cudaStream_t[streamNum];
	cudaEvent_t* cpyEvents = new cudaEvent_t[streamNum];
	int h_prior;
	cudaDeviceGetStreamPriorityRange(nullptr,&h_prior);
	cudaCheckError();

	for(int i = 0; i <streamNum; ++i)
	{
		cudaStreamCreateWithPriority(&streams[i],cudaStreamDefault,h_prior++);
		cudaCheckError();
		cudaEventCreate(&cpyEvents[i]);
		cudaCheckError();
	}

	dim3 thsPerBlck(BATCH_W,BATCH_H);
	dim3 blckNum(IMG_W/(BATCH_W * PXL_PER_THD) + 1,IMG_H/(BATCH_H) + 1);


	for(int i = 0; i < loopCount;++i)
	{
		int batch_num_i = min(streamNum,chCount - i * loopCount);
		for(int j =0; j< batch_num_i;++j)
		{
			cudaMemcpyAsync(&d_mem[j * chSize], &h_mem[j * chSize + i * streamNum * chSize],chSizeBytes,
			                 cudaMemcpyHostToDevice, streams[j]);
			cudaCheckError();
		}


		for(int j =0; j< batch_num_i;++j)
		{
			kernel<<<blckNum, thsPerBlck, 0, streams[j]>>>(&d_mem[i * chSize],
														   &d_result_mem[i * chSize],
															  IMG_W,IMG_H,PXL_PER_THD,IMAGE_SCALE);
			cudaCheckError();
		}

		for(int j =0; j< batch_num_i;++j)
		{
			 cudaMemcpyAsync(&h_result_mem[j * chSize + i * streamNum * chSize],
					         &d_result_mem[j * chSize],chSizeBytes,
					 	 	 cudaMemcpyDeviceToHost, streams[j]);
			cudaCheckError();
		}
	}

	cudaDeviceSynchronize();
	cudaCheckError();
	testRead(h_result_mem,totalSize/2);
    Timer::getInstance().stop(name);
	cudaFree(d_mem);
	cudaCheckError();
	cudaFreeHost(h_mem);
	cudaCheckError();
}

void testOversubUMOpt(kernelPtr kernel,bool advised)
{
	const ulong kb = 1024;
	const ulong totalSize = kb * kb * kb * IMG_SIZE_GB;
	const uint channelNum = 3;
	uint width = IMG_W;
	uint height = IMG_H;
	uint imgCount = totalSize / (width * height * channelNum * sizeof(uchar));
	printf("ImgCount: %d\n",imgCount);
	std::string name = "Oversubscription multi_img UM streams ";
	testStreamImgProcessingUm(kernel,name + std::to_string(advised),imgCount,advised);
}

void testFluidSimStd()
{
    Timer::getInstance().start("Fluid simulation std mem");
    StartArgs args = parsInputArguments();
    Fraction* space = initSpace(args,false,false);
    FluidParams *d_params, params = initParams();

    cudaMalloc((void **)&d_params, sizeof(FluidParams));
    cudaMemcpy(d_params, &params, sizeof(FluidParams), cudaMemcpyHostToDevice);

    if (NULL == space)
        exit(-1);

    Fraction *d_space, *d_result;
    int totalSize = sizeof(Fraction)*args.SIZE(), i;
    void *result = new Fraction[totalSize];

    cudaMalloc((void **)&d_space, totalSize);
    cudaMalloc((void **)&d_result, totalSize);
    cudaCheckErrors("Mallocs");
    cudaMemcpy(d_space, space, totalSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, result, totalSize, cudaMemcpyHostToDevice);

    cudaCheckErrors("Copy mem");
    printf("StdMem Simulation started\n");

    for (i = 0; i<args.NUM_OF_ITERATIONS; ++i)
    {
        simulation(args, d_params, d_space, d_result);
        swapPointers(d_space, d_result);
        cudaThreadSynchronize();

    	if (i % 2 == 0)
    		cudaMemcpy(space, d_space, totalSize, cudaMemcpyDeviceToHost);
    	else
    		cudaMemcpy(space, d_result, totalSize, cudaMemcpyDeviceToHost);

    	testRead(space,totalSize);
    }


    Timer::getInstance().stop("Fluid simulation std mem");
    printf("Simulation completed\n");
    free(space);
    cudaFree(d_params);
    cudaFree(d_space);
    cudaFree(d_result);
    cudaCheckErrors("Free mem");
}

void testFluidSimUM(bool withAdvise)
{
    std::string name = withAdvise ? "Fluid simulation UM advised" : "Fluid simulation UM std";
    Timer::getInstance().start(name);

	int device = -1;
	cudaGetDevice(&device);
    StartArgs args = parsInputArguments();
    int totalSize = sizeof(Fraction)*args.SIZE(), i;
    FluidParams *um_params, params = initParams();
    cudaMallocManaged(&um_params, sizeof(FluidParams));
    memcpy(um_params,&params,sizeof(FluidParams));
    Fraction *buffer,*result,*start;
    cudaMallocManaged(&buffer,totalSize);
    cudaCheckErrors("UM Mallocs");
    if(withAdvise)
    {
    	printf("UM advised Simulation started\n");
    	cudaMemAdvise(um_params,sizeof(FluidParams),cudaMemAdviseSetReadMostly,device);
    	cudaCheckError();
    	cudaMemPrefetchAsync(um_params,sizeof(FluidParams),device,NULL);
    	cudaCheckError();
    	cudaMemPrefetchAsync(buffer,totalSize,device,NULL);
    	cudaCheckError();
    }
    else
    {
    	printf("UM Simulation started\n");
    }
    Fraction* space = initSpace(args,true,withAdvise,device);
    for (i = 0; i<args.NUM_OF_ITERATIONS; ++i)
    {
        simulation(args, um_params, space, buffer);
        swapPointers(space, buffer);
        cudaThreadSynchronize();

    	result = (i % 2 == 0) ? space : buffer;
    	start = (i % 2 == 0)  ? buffer : space;

        if(withAdvise)
        {
        	cudaMemPrefetchAsync(result,totalSize,cudaCpuDeviceId,NULL);
        	cudaCheckError();
        	cudaMemPrefetchAsync(start,totalSize,device,NULL);
        	cudaCheckError();
        }

    	testRead(result,totalSize);

        if(withAdvise)
        {
        	cudaMemPrefetchAsync(result,totalSize,device,NULL);
        	cudaCheckError();
        }
    }

    Timer::getInstance().stop(name);
    printf("Simulation completed\n");

    cudaFree(space);
    cudaFree(buffer);
    cudaFree(um_params);
}

