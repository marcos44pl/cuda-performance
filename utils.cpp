/*
 * utils.cpp
 *
 *  Created on: 13 wrz 2017
 *      Author: mknap
 */
#include <helper_cuda.h>
#include <time.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <cmath>

#include "utils.h"
#include "Fraction.h"
#include "types.h"

FluidParams initParams()
{
    FluidParams params;
    params.d = make_float4(0.1, 0.1, 0.1, 0.04);
    params.omega = 0;
    params.mustaSteps = 4;
    return params;
}

void initCuda()
{
	int nDevices;
    int devCount = cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        cudaCheckErrors("Init Cuda")
        printf("CUDA device [%s] has %d Multi-Processors\n",
               props.name, props.multiProcessorCount);
    }
    if(nDevices > 1)
        cudaSetDevice(1); //Dla mnie bo mam SLI;
}

Fraction createFraction(bool random)
{
    Fraction f;
    if (true == random)
    {
        f.E = (float)(rand() % MAX_START_FORCE + 1);
        f.R = (float)(rand() % MAX_START_FORCE + 1);
        f.Vx = (float)(rand()%MAX_VELOCITY);
        f.Vy = (float)(rand() % MAX_VELOCITY);
        f.Vz = 0;
    }
    else
    {
        f.E = (float)MAX_START_FORCE;
        f.R = (float)MAX_START_FLUX;
        f.Vx = (float)MAX_VELOCITY;
        f.Vy = 0.0f;
        f.Vz = 0.0f;
    }
    return f;
}

Fraction* initSpace(StartArgs args,bool withUM,bool withPrefetch,int device)
{

    Fraction* space;
    int s = sizeof(Fraction)*args.SIZE();
    if(withUM)
    	cudaMallocManaged(&space,s);
    else
    	space = (Fraction*)malloc(s);
	cudaCheckError();

	if(nullptr==space)
	{
		printf("memory allocation error\n");
		return nullptr;
	}

	//IF RANDOM FLAG IS SET THEN INIT SPACE HAS DIFFERENT RESULT EACH TIME
	if(true==args.random)
		srand(time(NULL));

    const int3 mid = make_int3(15, args.Y_SIZE / 2, args.Z_SIZE / 2);
    const int3 rad = make_int3(5, args.Y_SIZE /6, args.Z_SIZE / 6);
    const int3 start = make_int3(mid.x - rad.x, mid.y - rad.y, mid.z - rad.z);
    const int3 end = make_int3(mid.x + rad.x, mid.y + rad.y, mid.z + rad.z);

	for(int z=0; z < args.Z_SIZE;++z)
	{
        for (int y = 0; y < args.Y_SIZE; ++y)
        {
            for (int x = 0; x < args.X_SIZE; ++x)
            {
            	if(start.x <= x && start.y <= y && start.z <= z &&
            	   x < end.x    && y < end.y    && z < end.z)
            		space[args.IDX_3D(x, y, z)] = createFraction(args.random);
            	else
            		space[args.IDX_3D(x, y, z)] = Fraction();
            	cudaCheckError();
            }
        }
	}

    if(withPrefetch)
    {
    	cudaMemPrefetchAsync(space,s,device);
    	cudaCheckError();
    }
	return space;
}

void swapPointers(Fraction*& p1,Fraction*& p2)
{
	Fraction* tmp;

	tmp=p1;
	p1=p2;
	p2=tmp;
}

void compare_results(StartArgs args, Fraction* hostSpace,Fraction* deviceSpace)
{
	float diffMax=0,diffMin=0;
	int numOfDiffs=0;
	bool firstDiff = true;
    float eps = 0.001;

    for (int z = 2; z<args.Z_SIZE - 2; ++z)
    {
        for (int y = 2; y < args.Y_SIZE - 2; ++y)
        {
            for (int x = 2; x < args.X_SIZE - 2; ++x)
            {
                int i = args.IDX_3D(x, y, z);
                auto const& h = hostSpace[i];
                auto const& d = deviceSpace[i];
                float err = fabs(hostSpace[i].E - deviceSpace[i].E);
                if (err > eps)
                {
                    numOfDiffs++;
                }
            }
        }
	}

	printf("Compare results:\n\tNum of differences: %d\n",numOfDiffs);
}

float* spaceToFloats(StartArgs args,Fraction* space)
{
    float* spaceFloats = (float*)(malloc(sizeof(float)*args.SIZE()*5));
	if(NULL==spaceFloats)
	{
		printf("memory allocation error\n");
		return NULL;
	}

	for(int i=0; i<args.SIZE();++i)
	{
		spaceFloats[5*i] = space[i].E;
		spaceFloats[5*i+1]=space[i].R;
		spaceFloats[5*i+2]=space[i].Vx;
		spaceFloats[5*i+3]=space[i].Vy;
		spaceFloats[5*i+4]=space[i].Vz;
	}

	return spaceFloats;
}

void floatsToSpace(StartArgs args,float* floats,Fraction* space)
{
	for(int i=0; i<args.SIZE();++i)
	{
		space[i].E = floats[5*i];
		space[i].R = floats[5*i+1];
		space[i].Vx= floats[5*i+2];
		space[i].Vy= floats[5*i+3];
		space[i].Vz= floats[5*i+4];
	}
}



