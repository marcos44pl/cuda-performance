/*
 * ImageReader.h
 *
 *  Created on: Mar 16, 2018
 *      Author: mknap
 */

#ifndef IMAGEREADER_H_
#define IMAGEREADER_H_


#include <string>
#include <map>
#include <string.h>

#include "types.h"
#include "CImg.h"

class ImageManager;

typedef void (ImageManager::*loaderPtr)(void);

class ImageManager {
public:
	ImageManager(std::string const& imagePath);
	ImageManager();
	~ImageManager();

	uchar* createEmpty(int width,int height);
	uchar* load();
	void   save(std::string const& name,uchar* data);
	size_t get_width() const { return image->width(); }
	size_t get_height() const { return image->height(); }
	size_t get_size() const { return image->size() * sizeof(uchar); }
	uchar* get_data() const { return data; }

private:
	void loadBmp();
	void loadAny();

	std::string  imagePath;
	cimg_library::CImg<uchar>* image;
	uchar* data;
	const std::map<std::string,loaderPtr> dispatch_map;
};



#endif /* IMAGEREADER_H_ */
