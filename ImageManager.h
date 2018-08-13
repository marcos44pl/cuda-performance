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

class ImageManager;

typedef void (ImageManager::*loaderPtr)(void);

class ImageManager {
public:
	ImageManager(std::string const& imagePath);
	ImageManager();
	~ImageManager();

	uchar* createEmpty(uint width,uint height);
	uchar* load();
	void   save(std::string const& name,uchar* data);
	size_t get_width() const { return w; }
	size_t get_height() const { return h; }
	ulong  get_size() const { return w * h* 3 * sizeof(uchar); }
	uchar* get_data() const { return data; }
	void   clear() { delete data; }
private:
	void loadBmp();
	void loadAny();

	size_t w;
	size_t h;
	std::string  imagePath;
	//cimg_library::CImg<uchar>* image;
	uchar* data;
	const std::map<std::string,loaderPtr> dispatch_map;
};



#endif /* IMAGEREADER_H_ */
