#include "headers.h"
#include "config.h"
#include "stdio.h"


__global__ void kernel_rotation(uchar *in, uchar *out,uint width, uint height,uint pxl_p_thd,float norm)
{
	int x = blockIdx.x*BATCH_W + threadIdx.x;
	int y = blockIdx.y*BATCH_H + threadIdx.y;

	if(y < height)
	{
		for(int ix = x; ix < width; ix += blockDim.x)
		{
			out[y + ix * height] = in[ix + y * width];
		}
	}
}

__global__ void kernel_rotation_shared(uchar *in, uchar *out,uint width, uint height,uint pxl_p_thd,float norm)
{
	int x = blockIdx.x*BATCH_W + threadIdx.x;
	int y = blockIdx.y*BATCH_H + threadIdx.y;

    __shared__ uchar tile[BATCH_H][BATCH_W+1];

    int idx_in = x + y*width;

    x = blockIdx.y * BATCH_H + threadIdx.x;
    y = blockIdx.x * BATCH_W + threadIdx.y;
    int idx_out = x + y*height;

    for (int i=0; i<BATCH_H; i+=pxl_p_thd)
    {
        tile[threadIdx.y+i][threadIdx.x] = in[idx_in+i*width];
    }

    __syncthreads();

    for (int i=0; i<BATCH_W; i+=pxl_p_thd)
    {
        out[idx_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
    }
}


void rotation_global_mem(uchar* in, uchar* out,size_t width, size_t height)
{
	exec_kernel(in,out,width,height,kernel_rotation);
}

void rotation_shared_mem(uchar* in, uchar* out,size_t width, size_t height)
{
	exec_kernel(in,out,width,height,kernel_rotation_shared);
}
