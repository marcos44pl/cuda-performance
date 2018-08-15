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
void runNetworkLearning(fp16Import_t fp16ConvType, std::string const& name)
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
	mnist.classify_example(imgData_h,conv1, conv2, ip1, ip2);
    Timer::getInstance().stop(name);
}

template <class value_type>
void runConvolutionFwd(int iter,std::string name)
{
	Layer_t<value_type> conv1(1,20,3,FP16_HOST);
	network_t<value_type> mnist;
    value_type *srcData = NULL, *dstData = NULL;
    int n,c,h,w;
	n = 1;c = 1; h = 1024; w = 1024;
    checkCudaErrors( cudaMalloc(&srcData, h*w*sizeof(value_type)) );
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

 void testFl16Cudnn()
{
	printf("Testing fl16 with learning networks...");
	for(int i = 0; i < 10;++i)
	{
		runNetworkLearning<float>(FP16_CUDA,"CUDANN float");
		runNetworkLearning<double>(FP16_CUDA,"CUDANN double");
		runNetworkLearning<half1>(FP16_CUDA,"CUDANN half");
	}
}

void testFl16ConvCudaNN()
{
	int i = 1000;
	runConvolutionFwd<double>(i,"ConvolutionFwd test double");
	runConvolutionFwd<half1>(i,"ConvolutionFwd test half");
	runConvolutionFwd<float>(i,"ConvolutionFwd test float");
}
