/*
 * ImageReader.cpp
 *
 *  Created on: Mar 16, 2018
 *      Author: mknap
 */

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdio.h>

#include "ImageManager.h"
using namespace cimg_library;

ImageManager::ImageManager(std::string const& imagePath) :
				dispatch_map({
				{"bmp", &ImageManager::loadBmp},
				{"jpg", &ImageManager::loadAny},
				{"png", &ImageManager::loadAny}}),
				image(nullptr),
				data(nullptr),
				imagePath(imagePath)
{
}

ImageManager::ImageManager() : ImageManager::ImageManager("")
{
}

ImageManager::~ImageManager()
{
	clear();
}

uchar* ImageManager::load()
{
	std::string ext = imagePath.substr(imagePath.rfind("."));
	loaderPtr loadingMethod;
	if(dispatch_map.count(ext) > 0)

		loadingMethod = dispatch_map.at(ext);
	else
		loadingMethod = &ImageManager::loadAny;
	(this->*loadingMethod)();
	return data;
}

void ImageManager::loadBmp()
{
    int i;
    FILE* f = fopen(imagePath.c_str(), "rb");
    uchar info[54];
    fread(info, sizeof(uchar), 54, f); // read the 54-byte header

    // extract image height and width from header
    int width = *(int*)&info[18];
    int height = *(int*)&info[22];

    int size = 3 * width * height;
    data = new uchar[size];
    fread(data, sizeof(uchar), size, f); // read the rest of the data at once
    fclose(f);
    for(i = 0; i < size; i += 3)
    {
            uchar tmp = data[i];
            data[i] = data[i+2];
            data[i+2] = tmp;
    }
}

void ImageManager::save(const std::string& name, uchar* new_data)
{
	uint w = image->width(), h = image->height();
	image->_height = w;
	image->_width = h;
	image->_data = new_data;
	image->save(name.c_str());
	printf("Image saved at: %s\n",name.c_str());
}

void ImageManager::loadAny()
{
	image = new CImg<uchar>(imagePath.c_str());	// Allocate Unified Memory -- accessible from CPU or GPU
	data = new uchar[get_size()];
	memcpy(data,image->_data,get_size());
}

uchar* ImageManager::createEmpty(uint width,uint height)
{
	image = new CImg<uchar>(width,height,1,3,0);	// Allocate Unified Memory -- accessible from CPU or GPU
	//data = new uchar[get_size()];
	//memcpy(data,image->_data,get_size());
	data = image->_data;
	return image->_data;
}
