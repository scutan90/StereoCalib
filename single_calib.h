#ifndef SINGLE_CALIB_H
#define SINGLE_CALIB_H

#include "opencv2/core/core.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/calib3d/calib3d.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include <iostream>  
#include <fstream> 
#include <string.h>
#include <io.h>

using namespace std;
using namespace cv;

class SingleCalib{
public:
	//单目标定
	void SingleCalibrate(string intrinsic_filename = "intrinsics.yml", string pics_path="single_calib_pic");
	//将要读取的图片路径存储在fileList中
	void InitFileList(string path, vector<string>& files);
private:

	const double patLen = 5.0f;    // unit: mm  标定板每个格的宽度（金属标定板）
	double imgScale = 1.0;          //图像缩放的比例因子
	Size patSize = Size(6, 4);
	Size img_size;
};

#endif