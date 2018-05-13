#pragma once

#include "computation.h"
#include <stdio.h>

Fraction*   initSpace(StartArgs args,bool withUM = true,bool withPrefetch = false,int device = -1);
FluidParams initParams();
void  	    swapPointers(Fraction*& p1, Fraction*& p2);
void        compare_results(StartArgs args,Fraction* hostSpace,Fraction* deviceSpace);
float* 		spaceToFloats(StartArgs args,Fraction* space);
void 		floatsToSpace(StartArgs args,float* floats,Fraction* space);


template<typename T>
void testRead(T* data,ulong size)
{
	ulong c = size / sizeof(T);
	T tmp = data[0];
	for(ulong i = 0; i < c; ++i)
	{
		data[i] = tmp;
	}
}
