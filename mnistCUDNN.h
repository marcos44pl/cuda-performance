/**
* Copyright 2014 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/*
 * This example demonstrates how to use CUDNN library to implement forward
 * pass. The sample loads weights and biases from trained network,
 * takes a few images of digits and recognizes them. The network was trained on 
 * the MNIST dataset using Caffe. The network consists of two 
 * convolution layers, two pooling layers, one relu and two 
 * fully connected layers. Final layer gets processed by Softmax. 
 * cublasSgemv is used to implement fully connected layers.

 * The sample can work in single, double, half precision, but it
 * assumes the data in files is stored in single precision
 */

#include <sstream>
#include <fstream>
#include <stdlib.h>

#include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>
#include "FreeImage.h"
#include "fp16_dev.h"
#include "fp16_emu.h"
#include "gemv.h"
#include "error_util.h"

#define IMAGE_H 28
#define IMAGE_W 28


/********************************************************
 * Prints the error message, and exits
 * ******************************************************/

#define EXIT_WAIVED 0


void get_path(std::string& sFilename, const char *fname)
{
    sFilename = (std::string("data/") + std::string(fname));
}

// Need the map, since scaling factor is of float type in half precision
// Also when one needs to use float instead of half, e.g. for printing
template <typename T> 
struct ScaleFactorTypeMap { typedef T Type;};
template <> struct ScaleFactorTypeMap<half1>  { typedef float Type;};

// float/double <-> half conversion class
template <class value_type>
class Convert
{
public:
    template <class T>
    value_type operator()(T x) {return value_type(x);}
    value_type operator()(half1 x) {return value_type(cpu_half2float(x));}
};

template <>
class Convert<half1>
{
public:
    template <class T>
    half1 operator()(T x) {return cpu_float2half_rn (T(x));} 
    half1 operator()(half1 x) {return x;}
};

// IO utils
template <class value_type>
void readBinaryFile(const char* fname, int size, value_type* data_h)
{
    std::ifstream dataFile (fname, std::ios::in | std::ios::binary);
    std::stringstream error_s;
    if (!dataFile)
    {
        error_s << "Error opening file " << fname; 
        FatalError(error_s.str());
    }
    // we assume the data stored is always in float precision
    float* data_tmp = new float[size];
    int size_b = size*sizeof(float);
    if (!dataFile.read ((char*) data_tmp, size_b)) 
    {
        error_s << "Error reading file " << fname; 
        FatalError(error_s.str());
    }
    // conversion
    Convert<value_type> fromReal;
    for (int i = 0; i < size; i++)
    {
        data_h[i] = fromReal(data_tmp[i]);
    }
    delete [] data_tmp;
}

template <class value_type>
void readAllocMemcpy(const char* fname, int size, value_type** data_h, value_type** data_d)
{
    *data_h = new value_type[size];
    readBinaryFile<value_type>(fname, size, *data_h);

    int size_b = size*sizeof(value_type);
    checkCudaErrors( cudaMalloc(data_d, size_b) );
    checkCudaErrors( cudaMemcpy(*data_d, *data_h,
                                size_b,
                                cudaMemcpyHostToDevice) );
}

void FreeImageErrorHandler(FREE_IMAGE_FORMAT oFif, const char *zMessage)
{
    FatalError(zMessage);
}
template <class value_type>
void readImage(const char* fname, value_type* imgData_h)
{
    // declare a host image object for an 8-bit grayscale image
    std::string sFilename(fname);
    // Take care of half precision
    Convert<value_type> fromReal;
    
    // load gray-scale image from disk    
    // set your own FreeImage error handler
    FreeImage_SetOutputMessage(FreeImageErrorHandler);

    FREE_IMAGE_FORMAT eFormat = FreeImage_GetFileType(sFilename.c_str());

    // no signature? try to guess the file format from the file extension
    if (eFormat == FIF_UNKNOWN)
    {
        eFormat = FreeImage_GetFIFFromFilename(sFilename.c_str());
    }

    if (eFormat == FIF_UNKNOWN)
    {
        FatalError("Unknown image format");
    }
    // check that the plugin has reading capabilities ...

    FIBITMAP *pBitmap;
    if (FreeImage_FIFSupportsReading(eFormat))
    {
        pBitmap = FreeImage_Load(eFormat, sFilename.c_str());
    }

    if (pBitmap == 0)
    {
        FatalError("Error reading image");
    }
    
    // make sure this is an 8-bit single channel image
    if (FreeImage_GetColorType(pBitmap) != FIC_MINISBLACK)
    {
        FatalError("This is not 8-bit single channel imagee");    
    }
    if (FreeImage_GetBPP(pBitmap) != 8)
    {
        FatalError("This is not 8-bit single channel imagee");   
    }

    // create an ImageCPU to receive the loaded image data
    //ImageCPU_8u_C1 oImage(FreeImage_GetWidth(pBitmap), FreeImage_GetHeight(pBitmap));

    int width = FreeImage_GetWidth(pBitmap);
    int height = FreeImage_GetHeight(pBitmap);
    
    if (width != IMAGE_W || height != IMAGE_H)
    {
        FatalError("Image dimensions missmatch");
    }
    
    // Normalize image to be in range [0,1]
    for (int i = 0; i < height; ++i)
    { 
        unsigned char *pSrcLine = FreeImage_GetScanLine(pBitmap, height - i - 1);
        for (int j = 0; j < width; j++)
        {
            int idx = IMAGE_W*i + j;
            imgData_h[idx] = fromReal(*(pSrcLine + j) / double(255));
        }
    }

    FreeImage_Unload(pBitmap); 
}

template <class value_type>
void printDeviceVector(int size, value_type* vec_d)
{
    typedef typename ScaleFactorTypeMap<value_type>::Type real_type;
    value_type *vec;
    vec = new value_type[size];
    cudaDeviceSynchronize();
    cudaMemcpy(vec, vec_d, size*sizeof(value_type), cudaMemcpyDeviceToHost);
    Convert<real_type> toReal;
    std::cout.precision(7);
    std::cout.setf( std::ios::fixed, std:: ios::floatfield );
    for (int i = 0; i < size; i++)
    {
        std::cout << toReal(vec[i]) << " ";
    }
    std::cout << std::endl;
    delete [] vec;
}

typedef enum {
        FP16_HOST  = 0, 
        FP16_CUDA  = 1,
        FP16_CUDNN = 2,
        FP16_COUNT
 } fp16Import_t;
template <class value_type>
struct Layer_t
{
    fp16Import_t fp16Import;
    int inputs;
    int outputs;
    // linear dimension (i.e. size is kernel_dim * kernel_dim)
    int kernel_dim;
    value_type *data_h, *data_d;
    value_type *bias_h, *bias_d;
    Layer_t() : data_h(NULL), data_d(NULL), bias_h(NULL), bias_d(NULL), 
                inputs(0), outputs(0), kernel_dim(0), fp16Import(FP16_HOST){}
    Layer_t(int _inputs, int _outputs, int _kernel_dim, fp16Import_t _fp16Import = FP16_HOST)
    : data_h(NULL), data_d(NULL), bias_h(NULL), bias_d(NULL),
      inputs(_inputs), outputs(_outputs), kernel_dim(_kernel_dim), fp16Import(_fp16Import)
    {
    	size_t size = inputs * outputs * kernel_dim * kernel_dim;
        data_h = new value_type[size];
        bias_h = new value_type[outputs];
        Convert<value_type> fromReal;
        for (int i = 0; i < size; i++)
        {
            data_h[i] = fromReal(1.0);
        }
        for (int i = 0; i < outputs; i++)
        {
        	bias_h[i] = fromReal(1.0);
        }
        int size_b = size*sizeof(value_type);
        checkCudaErrors( cudaMalloc(&data_d, size_b) );
        checkCudaErrors( cudaMemcpy(data_d, data_h, size_b,
                            cudaMemcpyHostToDevice) );
        checkCudaErrors( cudaMalloc(&bias_d, outputs*sizeof(value_type)) );
        checkCudaErrors( cudaMemcpy(bias_d, bias_h, outputs*sizeof(value_type),
                            cudaMemcpyHostToDevice) );
    }
    Layer_t(int _inputs, int _outputs, int _kernel_dim, const char* fname_weights,
            const char* fname_bias, fp16Import_t _fp16Import = FP16_HOST)
                  : inputs(_inputs), outputs(_outputs), kernel_dim(_kernel_dim)
    {
        fp16Import = _fp16Import;
        std::string weights_path, bias_path;
        weights_path = fname_weights; bias_path = fname_bias;

        readAllocInit(weights_path.c_str(), inputs * outputs * kernel_dim * kernel_dim, 
                        &data_h, &data_d);
        readAllocInit(bias_path.c_str(), outputs, &bias_h, &bias_d);
    }
    ~Layer_t()
    {
        if (data_h != NULL) delete [] data_h;
        if (data_d != NULL) checkCudaErrors( cudaFree(data_d) );
        if (bias_h != NULL) delete [] bias_h;
        if (bias_d != NULL) checkCudaErrors( cudaFree(bias_d) );
    }
private:
    void readAllocInit(const char* fname, int size, value_type** data_h, value_type** data_d)
    {
        readAllocMemcpy<value_type>(fname, size, data_h, data_d);
    }
};

template <>
void Layer_t<half1>::readAllocInit(const char* fname, int size, half1** data_h, half1** data_d)
{
    *data_h = new half1[size];
    int size_b = size*sizeof(half1);
    checkCudaErrors( cudaMalloc(data_d, size_b) );    
    float *data_tmp_h, *data_tmp_d;

    switch(fp16Import)
    {
        case FP16_HOST :
        {
            readBinaryFile<half1>(fname, size, *data_h);
            checkCudaErrors( cudaMemcpy(*data_d, *data_h, size_b,
                                cudaMemcpyHostToDevice) );
            break;
        }
        case FP16_CUDA :
        {
            readAllocMemcpy<float>(fname, size, &data_tmp_h, &data_tmp_d);

            gpu_float2half_rn<float>(size, data_tmp_d, *data_d);

            delete [] data_tmp_h;
            checkCudaErrors( cudaFree(data_tmp_d) );
            break;
        }
        case FP16_CUDNN :
        {
            readAllocMemcpy<float>(fname, size, &data_tmp_h, &data_tmp_d);
            delete [] data_tmp_h;
            cudnnHandle_t cudnnHandle;
            cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
            checkCUDNN( cudnnCreate(&cudnnHandle) );
            checkCUDNN( cudnnCreateTensorDescriptor(&srcTensorDesc) );
            checkCUDNN( cudnnCreateTensorDescriptor(&dstTensorDesc) );
            checkCUDNN( cudnnSetTensor4dDescriptorEx(srcTensorDesc,
                                                CUDNN_DATA_FLOAT,
                                                1, size,
                                                1, 1,
                                                size, 1, 1, 1) );
            checkCUDNN( cudnnSetTensor4dDescriptorEx(dstTensorDesc,
                                                CUDNN_DATA_HALF,
                                                1, size,
                                                1, 1,
                                                size, 1, 1, 1) );
            float alpha = 1.0f;
            float beta = 0.0f;
            checkCUDNN( cudnnTransformTensor(cudnnHandle, &alpha,
                                             srcTensorDesc,
                                             data_tmp_d, &beta,
                                             dstTensorDesc,
                                             *data_d) );
            checkCUDNN( cudnnDestroyTensorDescriptor(srcTensorDesc) );
            checkCUDNN( cudnnDestroyTensorDescriptor(dstTensorDesc) );
            checkCUDNN( cudnnDestroy(cudnnHandle) );
            checkCudaErrors( cudaFree(data_tmp_d) );
            break;
        }
    }
}

// demonstrate different ways of setting tensor descriptor
//#define SIMPLE_TENSOR_DESCRIPTOR
#define ND_TENSOR_DESCRIPTOR
void setTensorDesc(cudnnTensorDescriptor_t& tensorDesc, 
                    cudnnTensorFormat_t& tensorFormat,
                    cudnnDataType_t& dataType,
                    int n,
                    int c,
                    int h,
                    int w)
{
#if SIMPLE_TENSOR_DESCRIPTOR
    checkCUDNN( cudnnSetTensor4dDescriptor(tensorDesc,
                                            tensorFormat,
                                            dataType,
                                            n, c,
                                            h,
                                            w ) );
#elif defined(ND_TENSOR_DESCRIPTOR)
    const int nDims = 4;
    int dimA[nDims] = {n,c,h,w};
    int strideA[nDims] = {c*h*w, h*w, w, 1};
    checkCUDNN( cudnnSetTensorNdDescriptor(tensorDesc,
                                            dataType,
                                            4,
                                            dimA,
                                            strideA ) ); 
#else
    checkCUDNN( cudnnSetTensor4dDescriptorEx(tensorDesc,
                                            dataType,
                                            n, c,
                                            h, w,
                                            c*h*w, h*w, w, 1) );
#endif
}

template <class value_type>
class network_t
{
    typedef typename ScaleFactorTypeMap<value_type>::Type scaling_type;
    int convAlgorithm;
    cudnnDataType_t dataType;
    cudnnTensorFormat_t tensorFormat;
    cudnnHandle_t cudnnHandle;
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc, biasTensorDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnPoolingDescriptor_t     poolingDesc;
    cudnnActivationDescriptor_t  activDesc;
    cudnnLRNDescriptor_t   normDesc;
    cublasHandle_t cublasHandle;
    void createHandles()
    {
        checkCUDNN( cudnnCreate(&cudnnHandle) );
        checkCUDNN( cudnnCreateTensorDescriptor(&srcTensorDesc) );
        checkCUDNN( cudnnCreateTensorDescriptor(&dstTensorDesc) );
        checkCUDNN( cudnnCreateTensorDescriptor(&biasTensorDesc) );
        checkCUDNN( cudnnCreateFilterDescriptor(&filterDesc) );
        checkCUDNN( cudnnCreateConvolutionDescriptor(&convDesc) );
        checkCUDNN( cudnnCreatePoolingDescriptor(&poolingDesc) );
        checkCUDNN( cudnnCreateActivationDescriptor(&activDesc) );
        checkCUDNN( cudnnCreateLRNDescriptor(&normDesc) );

        checkCublasErrors( cublasCreate(&cublasHandle) );
    }
    void destroyHandles()
    {
        checkCUDNN( cudnnDestroyLRNDescriptor(normDesc) );
        checkCUDNN( cudnnDestroyPoolingDescriptor(poolingDesc) );
        checkCUDNN( cudnnDestroyActivationDescriptor(activDesc) );
        checkCUDNN( cudnnDestroyConvolutionDescriptor(convDesc) );
        checkCUDNN( cudnnDestroyFilterDescriptor(filterDesc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(srcTensorDesc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(dstTensorDesc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(biasTensorDesc) );
        checkCUDNN( cudnnDestroy(cudnnHandle) );

        checkCublasErrors( cublasDestroy(cublasHandle) );
    }
  public:
    network_t()
    {
        convAlgorithm = -1;
        switch (sizeof(value_type))
        {
            case 2 : dataType = CUDNN_DATA_HALF; break;
            case 4 : dataType = CUDNN_DATA_FLOAT; break;
            case 8 : dataType = CUDNN_DATA_DOUBLE; break;
            default : FatalError("Unsupported data type");
        }
        tensorFormat = CUDNN_TENSOR_NCHW;
        createHandles();    
    };
    ~network_t()
    {
        destroyHandles();
    }
    void resize(int size, value_type **data)
    {
        if (*data != NULL)
        {
            checkCudaErrors( cudaFree(*data) );
        }
        checkCudaErrors( cudaMalloc(data, size*sizeof(value_type)) );
    }
    void setConvolutionAlgorithm(const cudnnConvolutionFwdAlgo_t& algo)
    {
        convAlgorithm = (int) algo;
    }
    void addBias(const cudnnTensorDescriptor_t& dstTensorDesc, const Layer_t<value_type>& layer, int c, value_type *data)
    {
        setTensorDesc(biasTensorDesc, tensorFormat, dataType, 1, c, 1, 1);

        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(1);
        checkCUDNN( cudnnAddTensor( cudnnHandle, 
                                    &alpha, biasTensorDesc,
                                    layer.bias_d,
                                    &beta,
                                    dstTensorDesc,
                                    data) );
    }
    void fullyConnectedForward(const Layer_t<value_type>& ip,
                          int& n, int& c, int& h, int& w,
                          value_type* srcData, value_type** dstData, bool stdAlone = false)
    {
        if (n != 1)
        {
            FatalError("Not Implemented"); 
        }
        int dim_x = c*h*w;
        int dim_y = ip.outputs;
        if(stdAlone && dstData == NULL)
        	cudaMalloc(dstData,n*c*h*w*sizeof(value_type));
        else
        	resize(n*c*h*w, dstData);
        scaling_type alpha = scaling_type(1), beta = scaling_type(1);
        // place bias into dstData
        checkCudaErrors( cudaMemcpy(*dstData, ip.bias_d, dim_y*sizeof(value_type), cudaMemcpyDeviceToDevice) );
        
        gemv(cublasHandle, dim_x, dim_y, alpha,
                ip.data_d, srcData, beta,*dstData);

        h = 1; w = 1; c = dim_y;
    }
    void convoluteForward(const Layer_t<value_type>& conv,
                          int& n, int& c, int& h, int& w,
                          value_type* srcData, value_type** dstData, bool stdAlone = false)
    {
        cudnnConvolutionFwdAlgo_t algo;

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);

        const int tensorDims = 4;
        int tensorOuputDimA[tensorDims] = {n,c,h,w};
        const int filterDimA[tensorDims] = {conv.outputs, conv.inputs, 
                                        conv.kernel_dim, conv.kernel_dim};
                                       
        checkCUDNN( cudnnSetFilterNdDescriptor(filterDesc,
                                              dataType,
                                              CUDNN_TENSOR_NCHW,
                                              tensorDims,
                                              filterDimA) );
 
        const int convDims = 2;
        int padA[convDims] = {0,0};
        int filterStrideA[convDims] = {1,1};
        int upscaleA[convDims] = {1,1};
        cudnnDataType_t  convDataType = dataType;
        if (dataType == CUDNN_DATA_HALF) {
            convDataType = CUDNN_DATA_FLOAT; //Math are done in FP32 when tensor are in FP16
        }
        checkCUDNN( cudnnSetConvolutionNdDescriptor(convDesc,
                                                    convDims,
                                                    padA,
                                                    filterStrideA,
                                                    upscaleA,
                                                    CUDNN_CROSS_CORRELATION,
                                                    convDataType) );
        // find dimension of convolution output
        checkCUDNN( cudnnGetConvolutionNdForwardOutputDim(convDesc,
                                                srcTensorDesc,
                                                filterDesc,
                                                tensorDims,
                                                tensorOuputDimA) );
        n = tensorOuputDimA[0]; c = tensorOuputDimA[1];
        h = tensorOuputDimA[2]; w = tensorOuputDimA[3];

        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

        if (convAlgorithm < 0)
        {
            checkCUDNN( cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
                                                    srcTensorDesc,
                                                    filterDesc,
                                                    convDesc,
                                                    dstTensorDesc,
                                                    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                    0,
                                                    &algo
                                                    ) );
            convAlgorithm = algo;

            int requestedAlgoCount = 5; 
            int returnedAlgoCount[1];
            cudnnConvolutionFwdAlgoPerf_t *results = (cudnnConvolutionFwdAlgoPerf_t*)malloc(sizeof(cudnnConvolutionFwdAlgoPerf_t)*requestedAlgoCount);
            checkCUDNN(cudnnFindConvolutionForwardAlgorithm( cudnnHandle, 
                                                     srcTensorDesc,
                                                     filterDesc,
                                                     convDesc,
                                                     dstTensorDesc,
                                                     requestedAlgoCount,
                                                     returnedAlgoCount,
                                                     results
                                                   ) );
            free(results);
        }
        else
        {
            algo = (cudnnConvolutionFwdAlgo_t)convAlgorithm;
        }

        if(stdAlone && dstData == NULL)
        	cudaMalloc(dstData,n*c*h*w*sizeof(value_type));
        else
        	resize(n*c*h*w, dstData);

        size_t sizeInBytes=0;
        void* workSpace=NULL;
        checkCUDNN( cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                srcTensorDesc,
                                                filterDesc,
                                                convDesc,
                                                dstTensorDesc,
                                                algo,
                                                &sizeInBytes) );
        if (sizeInBytes!=0)
        {
          checkCudaErrors( cudaMalloc(&workSpace,sizeInBytes) );
        }
        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(0);
        checkCUDNN( cudnnConvolutionForward(cudnnHandle,
                                              &alpha,
                                              srcTensorDesc,
                                              srcData,
                                              filterDesc,
                                              conv.data_d,
                                              convDesc,
                                              algo,
                                              workSpace,
                                              sizeInBytes,
                                              &beta,
                                              dstTensorDesc,
                                              *dstData) );
        addBias(dstTensorDesc, conv, c, *dstData);
        if (sizeInBytes!=0)
        {
          checkCudaErrors( cudaFree(workSpace) );
        }
    }

    void poolForward( int& n, int& c, int& h, int& w,
                      value_type* srcData, value_type** dstData, bool stdAlone = false)
    {
        const int poolDims = 2;
        int windowDimA[poolDims] = {2,2};
        int paddingA[poolDims] = {0,0};
        int strideA[poolDims] = {2,2};
        checkCUDNN( cudnnSetPoolingNdDescriptor(poolingDesc,
                                                CUDNN_POOLING_MAX,
                                                CUDNN_PROPAGATE_NAN,
                                                poolDims,
                                                windowDimA,
                                                paddingA,
                                                strideA ) );

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);        

        const int tensorDims = 4;
        int tensorOuputDimA[tensorDims] = {n,c,h,w};
        checkCUDNN( cudnnGetPoolingNdForwardOutputDim(poolingDesc,
                                                    srcTensorDesc,
                                                    tensorDims,
                                                    tensorOuputDimA) );
        n = tensorOuputDimA[0]; c = tensorOuputDimA[1];
        h = tensorOuputDimA[2]; w = tensorOuputDimA[3];

        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);  
     
        if(stdAlone && dstData == NULL)
        	cudaMalloc(dstData,n*c*h*w*sizeof(value_type));
        else
        	resize(n*c*h*w, dstData);
        scaling_type alpha = scaling_type(1);
        scaling_type beta = scaling_type(0);
        checkCUDNN( cudnnPoolingForward(cudnnHandle,
                                          poolingDesc,
                                          &alpha,
                                          srcTensorDesc,
                                          srcData,
                                          &beta,
                                          dstTensorDesc,
                                          *dstData) );
    }
    void softmaxForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
    {
        resize(n*c*h*w, dstData);

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(0);
        checkCUDNN( cudnnSoftmaxForward(cudnnHandle,
                                          CUDNN_SOFTMAX_ACCURATE ,
                                          CUDNN_SOFTMAX_MODE_CHANNEL,
                                          &alpha,
                                          srcTensorDesc,
                                          srcData,
                                          &beta,
                                          dstTensorDesc,
                                          *dstData) );
    }
    void lrnForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
    {
        unsigned lrnN = 5;
        double lrnAlpha, lrnBeta, lrnK;
        lrnAlpha = 0.0001; lrnBeta = 0.75; lrnK = 1.0;
        checkCUDNN( cudnnSetLRNDescriptor(normDesc,
                                            lrnN,
                                            lrnAlpha,
                                            lrnBeta,
                                            lrnK) );

        resize(n*c*h*w, dstData);

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(0);
        checkCUDNN( cudnnLRNCrossChannelForward(cudnnHandle,
                                            normDesc,
                                            CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                            &alpha,
                                            srcTensorDesc,
                                            srcData,
                                            &beta,
                                            dstTensorDesc,
                                            *dstData) );
    }
    void activationForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
    {
        checkCUDNN( cudnnSetActivationDescriptor(activDesc,
                                                CUDNN_ACTIVATION_RELU,
                                                CUDNN_PROPAGATE_NAN,
                                                0.0) );
    
        resize(n*c*h*w, dstData);

        setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
        setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);

        scaling_type alpha = scaling_type(1);
        scaling_type beta  = scaling_type(0);
        checkCUDNN( cudnnActivationForward(cudnnHandle,
                                            activDesc,
                                            &alpha,
                                            srcTensorDesc,
                                            srcData,
                                            &beta,
                                            dstTensorDesc,
                                            *dstData) );    
    }

    void classify_example(value_type* imgData_h,
    					  const Layer_t<value_type>& conv1,
                          const Layer_t<value_type>& conv2,
                          const Layer_t<value_type>& ip1,
                          const Layer_t<value_type>& ip2)
    {
        int n,c,h,w;
        value_type *srcData = NULL, *dstData = NULL;
        checkCudaErrors( cudaMalloc(&srcData, IMAGE_H*IMAGE_W*sizeof(value_type)) );
        for(int i = 0; i < 1000; ++i)
        {
			n = c = 1; h = IMAGE_H; w = IMAGE_W;
			resize(IMAGE_H*IMAGE_W*sizeof(value_type),&srcData);
			checkCudaErrors(cudaMemcpy(srcData, imgData_h,
										IMAGE_H*IMAGE_W*sizeof(value_type),
										cudaMemcpyHostToDevice));

			convoluteForward(conv1, n, c, h, w, srcData, &dstData);
			//printf("Conv n %d c %d h %d w %d\n",n,c,h,w);
			poolForward(n, c, h, w, dstData, &srcData);
			//printf("Pool n %d c %d h %d w %d\n",n,c,h,w);

			convoluteForward(conv2, n, c, h, w, srcData, &dstData);
			//printf("Conv2 n %d c %d h %d w %d\n",n,c,h,w);

			poolForward(n, c, h, w, dstData, &srcData);
			//printf("Pool2 n %d c %d h %d w %d\n",n,c,h,w);

			fullyConnectedForward(ip1, n, c, h, w, srcData, &dstData);
			//printf("FulC n %d c %d h %d w %d\n",n,c,h,w);

			activationForward(n, c, h, w, dstData, &srcData);
			//printf("Activ n %d c %d h %d w %d\n",n,c,h,w);

			lrnForward(n, c, h, w, srcData, &dstData);
			//printf("LrnFwd n %d c %d h %d w %d\n",n,c,h,w);

			fullyConnectedForward(ip2, n, c, h, w, dstData, &srcData);
			//printf("FullCon2 n %d c %d h %d w %d\n",n,c,h,w);

			softmaxForward(n, c, h, w, srcData, &dstData);
			//printf("SoftMax n %d c %d h %d w %d\n",n,c,h,w);

			checkCudaErrors (cudaDeviceSynchronize());
        }
        checkCudaErrors( cudaFree(srcData) );
        checkCudaErrors( cudaFree(dstData) );
    }
};

#if !defined(CUDA_VERSION) || (CUDA_VERSION <= 7000)
// using 1x1 convolution to emulate gemv in half precision when cuBLAS version <= 7.0
template <>
void network_t<half1>::fullyConnectedForward(const Layer_t<half1>& ip,
                          int& n, int& c, int& h, int& w,
                          half1* srcData, half1** dstData)
{
    c = c*h*w; h = 1; w = 1;
    network_t<half1>::convoluteForward(ip, n, c, h, w, srcData, dstData);
    c = ip.outputs;
}
#endif

