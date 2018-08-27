#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "config.h"
#include "headers.h"
#include "test_cases.h"


__device__ uchar compute_sobel(uchar tl, //top left
							   uchar tm, //top middle
							   uchar tr, //top right
							   uchar ml, //middle left
							   uchar mr, //middle right
							   uchar bl, //bottom left
							   uchar bm, //bottom middle
							   uchar br, //bottom right
							   float norm)
{
	short horz = tl + 2 * tm + tr - br - 2 * bm - bl;
	short vert = tl + 2 * ml + bl - tr - 2 * mr - br;
	short all = (short)(norm * float(fabsf(horz) + fabsf(vert)));

	if(all < 0)
		all = 0;
	if(all > 0xff)
		all = 0xff;

	return all;
}

__device__ void read_write_filter_block(int x, int y,uchar *in, uchar *out,uint width,float norm)
{
	uchar value = compute_sobel(in[PXL_ID(x,y,width)],
							  in[PXL_ID(x,y+1,width)],
							  in[PXL_ID(x,y+2,width)],
							  in[PXL_ID(x+1,y,width)],
							  in[PXL_ID(x+1,y+2,width)],
							  in[PXL_ID(x+2,y,width)],
							  in[PXL_ID(x+2,y+1,width)],
							  in[PXL_ID(x+2,y+2,width)],
							  norm);
	out[PXL_ID(x+1,y+1,width)] = value;
}

__global__ void kernel_sobel_filter_coalesc(uchar *in, uchar *out,uint width, uint height,uint pxl_p_thd,float norm)
{
	int x = blockIdx.x*BATCH_W + threadIdx.x - RAD;
	int y = blockIdx.y*BATCH_H + threadIdx.y - RAD;

	if(x >= 0 && y >= 0 && y < height)
	{
		for(int ix = x; ix < width; ix += blockDim.x)
		{
			read_write_filter_block(ix,y,in,out,width,norm);
		}
	}
}

__global__ void kernel_sobel_filter_non_coalesc(uchar *in, uchar *out,uint width, uint height,uint pxl_p_thd,float norm)
{
	int x = blockIdx.x*BATCH_W + threadIdx.x - RAD;
	int y = blockIdx.y*BATCH_H + threadIdx.y - RAD;

	if(x >= 0 && y >= 0 && y < height)
	{
		for(int ix =  x * pxl_p_thd;  ix < (x+1) * pxl_p_thd; ++ix)
		{
			if(ix < width)
			{
				read_write_filter_block(ix,y,in,out,width,norm);
			}
		}
	}
}

void sobel_filter_coalesc(uchar* in, uchar* out,size_t width, size_t height)
{
	exec_kernel(in,out,width,height,kernel_sobel_filter_coalesc);
}

void sobel_filter_non_coalesc(uchar* in, uchar* out,size_t width, size_t height)
{
	exec_kernel(in,out,width,height,kernel_sobel_filter_non_coalesc);
}

void testSobelOversubStd()
{
	std::cout << "Testing oversubscription std\n";
	testOversubStd(kernel_sobel_filter_coalesc);
}

void testSobelOversubUM()
{
	std::cout << "Testing oversubscription UM naive\n";
	testOversubNaiveUM(kernel_sobel_filter_coalesc);
	testOversubNaiveUM(kernel_sobel_filter_coalesc,false);
}

void testSobelOversubUMOpt()
{
	std::cout << "Testing oversubscription UM MultiImg opt\n";
	testOversubUMOpt(kernel_sobel_filter_coalesc);
}

void testSobelOversubMultiImgStd()
{
	std::cout << "Testing oversubscription UM MultiImg Std\n";
	testOversubMultiImgStd(kernel_sobel_filter_coalesc);
}

void testSobelStreamUM(bool withAdvise)
{
	std::cout << "Testing streams img processing UM " << withAdvise << std::endl;
	testStreamImgProcessingUm(kernel_sobel_filter_non_coalesc,"Stream Image Processing UM",200,withAdvise);
}

void testSobelStreamStd()
{
	std::cout << "Testing streams img processing Std\n";
	testStreamImgProcessingStd(kernel_sobel_filter_non_coalesc);
}
