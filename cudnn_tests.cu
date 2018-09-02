#include <string>

#include "headers.h"
#include "mnistCUDNN.h"
#include "Timer.h"

const char *first_image = "one_28x28.pgm";
const char *second_image = "three_28x28.pgm";
const char *third_image = "five_28x28.pgm";
const char *lena_image = "lena.pgm";

const char *conv1_bin = "conv1.bin";
const char *conv1_bias_bin = "conv1.bias.bin";
const char *conv2_bin = "conv2.bin";
const char *conv2_bias_bin = "conv2.bias.bin";
const char *ip1_bin = "ip1.bin";
const char *ip1_bias_bin = "ip1.bias.bin";
const char *ip2_bin = "ip2.bin";
const char *ip2_bias_bin = "ip2.bias.bin";

template <class value_type>
void runNetworkLearning(int iter,fp16Import_t fp16ConvType, std::string const& name)
{
	cudaDeviceReset();
	std::string image_path = std::string("data/") + std::string(lena_image);
	network_t<value_type> mnist;
    value_type imgData_h[IMAGE_H*IMAGE_W];
	readImage(image_path.c_str(), imgData_h);
    Timer::getInstance().start(name);
	Layer_t<value_type> conv1(1,20,5,conv1_bin,conv1_bias_bin,fp16ConvType);
	Layer_t<value_type> conv2(20,50,5,conv2_bin,conv2_bias_bin,fp16ConvType);
	Layer_t<value_type>   ip1(800,500,1,ip1_bin,ip1_bias_bin,fp16ConvType);
	Layer_t<value_type>   ip2(500,10,1,ip2_bin,ip2_bias_bin,fp16ConvType);
	mnist.classify_example(iter,imgData_h,conv1, conv2, ip1, ip2);
    Timer::getInstance().stop(name);
}

template <class value_type>
void runConvolutionFwd(int iter,std::string  const& name)
{
	Layer_t<value_type> conv1(1,20,3,FP16_HOST);
	network_t<value_type> mnist;
    value_type *srcData = NULL, *dstData = NULL;
    int n,c,h,w;
	n = 1;c = 1; h = 1024; w = 1024;
    checkCudaErrors( cudaMallocManaged(&srcData, h*w*sizeof(value_type)) );
    memset(srcData,0,h*w*sizeof(value_type));
    Timer::getInstance().start(name);
    for(int i = 0; i < iter;++i)
    {
		n = 1;c = 1; h = 1024; w = 1024;
		mnist.convoluteForward(conv1,n,c,h,w,srcData,&dstData);
    }
	checkCudaErrors (cudaDeviceSynchronize());
    Timer::getInstance().stop(name);
	cudaFree(srcData);
	cudaFree(dstData);
}

template <class value_type>
void runPoolForward(int iter,std::string const& name)
{
	network_t<value_type> mnist;
    value_type *srcData = NULL, *dstData = NULL;
    int n,c,h,w;
	n = 1;c = 100; h = 104; w = 104;
    checkCudaErrors( cudaMallocManaged(&srcData, h*w*sizeof(value_type)) );
    memset(srcData,0,h*w*sizeof(value_type));
    Timer::getInstance().start(name);
    for(int i = 0; i < iter;++i)
    {
    	n = 1;c = 100; h = 104; w = 104;
		mnist.poolForward(n,c,h,w,srcData,&dstData);
    }
	checkCudaErrors (cudaDeviceSynchronize());
    Timer::getInstance().stop(name);
	cudaFree(srcData);
	cudaFree(dstData);
}

template <class value_type>
void runFullyConnectedForward(int iter,std::string const& name)
{
	Layer_t<value_type>   ip1(800,500,1,FP16_HOST);
	network_t<value_type> mnist;
    value_type *srcData = NULL, *dstData = NULL;
    int n,c,h,w;
	n = 1;c = 50; h = 8; w = 8;
    checkCudaErrors( cudaMallocManaged(&srcData, h*w*sizeof(value_type)) );
    memset(srcData,0,h*w*sizeof(value_type));
    Timer::getInstance().start(name);
    for(int i = 0; i < iter;++i)
    {
    	n = 1;c = 50; h = 8; w = 8;
		mnist.fullyConnectedForward(ip1,n,c,h,w,srcData,&dstData);
    }
	checkCudaErrors (cudaDeviceSynchronize());
    Timer::getInstance().stop(name);
	cudaFree(srcData);
	cudaFree(dstData);
}


 void testFl16Cudnn(int iter)
{
	auto sizeStr = std::to_string(iter);
	printf("Testing fl16 with learning networks...\n");
	runNetworkLearning<float>(iter,FP16_CUDA,std::string("CUDANN float") + sizeStr);
	runNetworkLearning<double>(iter,FP16_CUDA,std::string("CUDANN double") + sizeStr);
	runNetworkLearning<half1>(iter,FP16_CUDA,std::string("CUDANN half") + sizeStr);

}

void testFl16ConvCudaNN(int iter)
{
	printf("Testing fl16 with convolutionFwd...\n");
	auto sizeStr = std::to_string(iter);
	runConvolutionFwd<double>(iter,std::string("ConvolutionFwd test double") + sizeStr);
	runConvolutionFwd<half1>(iter,std::string("ConvolutionFwd test half") + sizeStr);
	runConvolutionFwd<float>(iter,std::string("ConvolutionFwd test float") + sizeStr);
}

void testFl16PoolCudaNN(int iter)
{
	auto sizeStr = std::to_string(iter);
	printf("Testing fl16 with poolFwd...\n");
	runPoolForward<double>(iter,std::string("PoolFwd test double ") + sizeStr);
	runPoolForward<half1>(iter,std::string("PoolFwd test half ") + sizeStr);
	runPoolForward<float>(iter,std::string("PoolFwd test float ") + sizeStr);
}

void testFl16FullyConnectedFwdCudaNN(int iter)
{
	printf("Testing fl16 with fully connected fwd...\n");
	auto sizeStr = std::to_string(iter);
	runFullyConnectedForward<double>(iter,std::string("fullyconnected fwd test double") + sizeStr);
	runFullyConnectedForward<half1>(iter,std::string("fullyconnected fwd test half") + sizeStr);
	runFullyConnectedForward<float>(iter,std::string("fullyconnected fwd test float") + sizeStr);
}
