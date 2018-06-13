#include <opencv2/opencv.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\legacy\legacy.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include <algorithm>
#include <iterator>
#include <cstdio>
#include <string>

using namespace cv;
using namespace std;

typedef unsigned int uint;

Size imgSize(1280, 720);
Size patSize(14, 12);           //每张棋盘寻找的角点个数是14*12个
const double patLen = 5.0f;    // unit: mm  标定板每个格的宽度（金属标定板）
double imgScale = 1.0;          //图像缩放的比例因子


//将要读取的图片路径存储在fileList中
vector<string> fileList;
void initFileList(string dir, int first, int last){
	fileList.clear();
	for(int cur = first; cur <= last; cur++){
		string str_file = dir + "/" + to_string(cur) + ".jpg";
		fileList.push_back(str_file);
	}
}


// 生成点云坐标后保存
static void saveXYZ(string filename, const Mat& mat)
{
    const double max_z = 1.0e4;
    ofstream fp(filename);
	if (!fp.is_open())
    {  
         std::cout<<"打开点云文件失败"<<endl;    
         fp.close();  
		 return ;
    }  
	//遍历写入
    for(int y = 0; y < mat.rows; y++)
    {
        for(int x = 0; x < mat.cols; x++)
        {
            Vec3f point = mat.at<Vec3f>(y, x);   //三通道浮点型
            if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z)   
				continue;   
            fp<<point[0]<<" "<<point[1]<<" "<<point[2]<<endl;
        }
    }
	fp.close();
}


// 存储视差数据
void saveDisp(const string filename, const Mat& mat)		
{
	ofstream fp(filename, ios::out);
	fp<<mat.rows<<endl;
	fp<<mat.cols<<endl;
	for(int y = 0; y < mat.rows; y++)
	{
		for(int x = 0; x < mat.cols; x++)
		{
			double disp = mat.at<short>(y, x); // 这里视差矩阵是CV_16S 格式的，故用 short 类型读取
			fp<<disp<<endl;       // 若视差矩阵是 CV_32F 格式，则用 float 类型读取
		}
	}
	fp.close();
}


void F_Gray2Color(Mat gray_mat, Mat& color_mat)
{
	color_mat = Mat::zeros(gray_mat.size(), CV_8UC3);
	int rows = color_mat.rows, cols = color_mat.cols;
	
	Mat red = Mat(gray_mat.rows, gray_mat.cols, CV_8U);
	Mat green = Mat(gray_mat.rows, gray_mat.cols, CV_8U);
	Mat blue = Mat(gray_mat.rows, gray_mat.cols, CV_8U);
	Mat mask = Mat(gray_mat.rows, gray_mat.cols, CV_8U);

	subtract(gray_mat, Scalar(255), blue);         // blue(I) = 255 - gray(I)
	red = gray_mat.clone();                        // red(I) = gray(I)
	green = gray_mat.clone();                      // green(I) = gray(I),if gray(I) < 128

	compare(green, 128, mask, CMP_GE);             // green(I) = 255 - gray(I), if gray(I) >= 128
	subtract(green, Scalar(255), green, mask);
	convertScaleAbs(green, green, 2.0, 2.0);

	vector<Mat> vec;
	vec.push_back(red);
	vec.push_back(green);
	vec.push_back(blue);
	cv::merge(vec, color_mat);
}

Mat F_mergeImg(Mat img1, Mat disp8){
	Mat color_mat = Mat::zeros(img1.size(), CV_8UC3);

	Mat red = img1.clone();
	Mat green = disp8.clone();
	Mat blue = Mat::zeros(img1.size(), CV_8UC1);

	vector<Mat> vec;
	vec.push_back(red);
	vec.push_back(blue);
	vec.push_back(green);
	cv::merge(vec, color_mat);
	
	return color_mat;
}


// 双目立体标定
int stereoCalibrate(string intrinsic_filename="intrinsics.yml", string extrinsic_filename="extrinsics.yml")
{
	vector<int> idx;
	//左侧相机的角点坐标和右侧相机的角点坐标
	
	vector<vector<Point2f>> imagePoints[2];

	//vector<vector<Point2f>> leftPtsList(fileList.size());
	//vector<vector<Point2f>> rightPtsList(fileList.size());

	for(uint i = 0; i < fileList.size();++i)
	{
		vector<Point2f> leftPts, rightPts;      // 存储左右相机的角点位置
		Mat rawImg = imread(fileList[i]);					   //原始图像
		if(rawImg.empty()){
			std::cout<<"the Image is empty..."<<fileList[i]<<endl;
			continue;
		}
		//截取左右图片
		Rect leftRect(0, 0, imgSize.width, imgSize.height);
		Rect rightRect(imgSize.width, 0, imgSize.width, imgSize.height);

		Mat leftRawImg = rawImg(leftRect);       //切分得到的左原始图像
		Mat rightRawImg = rawImg(rightRect);     //切分得到的右原始图像
		
		//imwrite("left.jpg", leftRawImg);
		//imwrite("right.jpg", rightRawImg);
		//std::cout<<"左侧图像：  宽度"<<leftRawImg.size().width<<"  高度"<<rightRawImg.size().height<<endl;
		//std::cout<<"右侧图像：  宽度"<<rightRawImg.size().width<<"  高度"<<rightRawImg.size().height<<endl;  //
		
		Mat leftImg, rightImg, leftSimg, rightSimg, leftCimg, rightCimg, leftMask, rightMask;
		// BGT -> GRAY	
		if(leftRawImg.type() == CV_8UC3)
            cvtColor(leftRawImg, leftImg, CV_BGR2GRAY);  //转为灰度图
        else
			leftImg = leftRawImg.clone();
		if(rightRawImg.type() == CV_8UC3)
			cvtColor(rightRawImg, rightImg, CV_BGR2GRAY); 
		else
			rightImg = rightRawImg.clone();

		imgSize = leftImg.size();
		
		//图像滤波预处理
		resize(leftImg, leftMask, Size(200, 200));		//resize对原图像img重新调整大小生成mask图像大小为200*200
		resize(rightImg, rightMask, Size(200, 200));
        GaussianBlur(leftMask, leftMask, Size(13, 13), 7);   
		GaussianBlur(rightMask, rightMask, Size(13, 13), 7); 
		resize(leftMask, leftMask, imgSize);
		resize(rightMask, rightMask, imgSize);
        medianBlur(leftMask, leftMask, 9);    //中值滤波
		medianBlur(rightMask, rightMask, 9);
		
        for (int v = 0; v < imgSize.height; v++) {
            for (int u = 0; u < imgSize.width; u++) {
                int leftX = ((int)leftImg.at<uchar>(v, u) - (int)leftMask.at<uchar>(v, u)) * 2 + 128;
				int rightX = ((int)rightImg.at<uchar>(v, u) - (int)rightMask.at<uchar>(v, u)) * 2 + 128;
				leftImg.at<uchar>(v, u)  = max(min(leftX, 255), 0);
				rightImg.at<uchar>(v, u) = max(min(rightX, 255), 0);
            }
        }
		
		//寻找角点， 图像缩放
		resize(leftImg, leftSimg, Size(), imgScale, imgScale);      //图像以0.5的比例缩放
		resize(rightImg, rightSimg, Size(), imgScale, imgScale);
        cvtColor(leftSimg, leftCimg, CV_GRAY2BGR);     //转为BGR图像，cimg和simg是800*600的图像
		cvtColor(rightSimg, rightCimg, CV_GRAY2BGR);
        

		//寻找棋盘角点
		bool leftFound = findChessboardCorners(leftCimg, patSize, leftPts, CV_CALIB_CB_ADAPTIVE_THRESH|CV_CALIB_CB_FILTER_QUADS);
		bool rightFound = findChessboardCorners(rightCimg, patSize, rightPts, CV_CALIB_CB_ADAPTIVE_THRESH|CV_CALIB_CB_FILTER_QUADS);

		if(leftFound)
			cornerSubPix(leftSimg, leftPts, Size(11, 11), Size(-1,-1 ), 
				TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 300, 0.01));
		if(rightFound)
			cornerSubPix(rightSimg, rightPts, Size(11, 11), Size(-1, -1), 
				TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 300, 0.01));   //亚像素
		
		//放大为原来的尺度
        for(uint j = 0;j < leftPts.size();j++) //该幅图像共132个角点，坐标乘以2，还原角点位置
			leftPts[j] *= 1./imgScale;
		for(uint j = 0;j < rightPts.size();j++)
			rightPts[j] *= 1./imgScale;

		//显示
		string leftWindowName = "Left Corner Pic", rightWindowName = "Right Corner Pic";
		
		Mat leftPtsTmp = Mat(leftPts) * imgScale;      //再次乘以 imgScale
		Mat rightPtsTmp = Mat(rightPts) * imgScale;

		drawChessboardCorners(leftCimg, patSize, leftPtsTmp, leftFound);      //绘制角点坐标并显示
		imshow(leftWindowName, leftCimg);
		imwrite("输出/DrawChessBoard/"+to_string(i)+"_left.jpg", leftCimg);
		waitKey(200);

		drawChessboardCorners(rightCimg, patSize, rightPtsTmp, rightFound);   //绘制角点坐标并显示
		imshow(rightWindowName, rightCimg);
		imwrite("输出/DrawChessBoard/"+to_string(i)+"_right.jpg", rightCimg);
		waitKey(200);

		cv::destroyAllWindows();
		
		//保存角点坐标
		if(leftFound && rightFound)   
		{
			imagePoints[0].push_back(leftPts);
			imagePoints[1].push_back(rightPts);  //保存角点坐标
			std::cout<<"图片 "<<i<<" 处理成功！"<<endl;
			idx.push_back(i);
		}
	}
	cv::destroyAllWindows();
	imagePoints[0].resize(idx.size());
	imagePoints[1].resize(idx.size());
	std::cout<<"成功标定的标定板个数为"<<idx.size()<<"  序号分别为: ";
	for(unsigned int i = 0;i < idx.size();++i)
		std::cout<<idx[i]<<"  ";
	
	//生成物点坐标
    vector<vector<Point3f>> objPts(idx.size());  //idx.size代表成功检测的图像的个数
    for (int y = 0; y < patSize.height; y++) {
        for (int x = 0; x < patSize.width; x++) {
            objPts[0].push_back(Point3f((float)x, (float)y, 0) * patLen);
        }
    }
    for (uint i = 1; i < objPts.size(); i++) {
        objPts[i] = objPts[0];
    }

	//
	// 双目立体标定
	Mat cameraMatrix[2], distCoeffs[2];
	vector<Mat> rvecs[2], tvecs[2]; 
    cameraMatrix[0] = Mat::eye(3, 3, CV_64F);
    cameraMatrix[1] = Mat::eye(3, 3, CV_64F);
    Mat R, T, E, F;

	cv::calibrateCamera(objPts, imagePoints[0], imgSize, cameraMatrix[0],    
        distCoeffs[0], rvecs[0], tvecs[0], CV_CALIB_FIX_K3);
    
	cv::calibrateCamera(objPts, imagePoints[1], imgSize, cameraMatrix[1],    
        distCoeffs[1], rvecs[1], tvecs[1], CV_CALIB_FIX_K3);

	std::cout<<endl<<"Left Camera Matrix: "<<endl<<cameraMatrix[0]<<endl;
	std::cout<<endl<<"Right Camera Matrix："<<endl<<cameraMatrix[1]<<endl;
	std::cout<<endl<<"Left Camera DistCoeffs: "<<endl<<distCoeffs[0]<<endl;
	std::cout<<endl<<"Right Camera DistCoeffs: "<<endl<<distCoeffs[1]<<endl;
	
    double rms = stereoCalibrate(objPts, imagePoints[0], imagePoints[1],
                    cameraMatrix[0], distCoeffs[0],
                    cameraMatrix[1], distCoeffs[1],
                    imgSize, R, T, E, F,
                    TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5));
                    //CV_CALIB_USE_INTRINSIC_GUESS);
					

    std::cout<<endl<<endl<<"立体标定完成！ "<<endl<<"done with RMS error=" << rms << endl;  //反向投影误差
	std::cout<<endl<<"Left Camera Matrix: "<<endl<<cameraMatrix[0]<<endl;
	std::cout<<endl<<"Right Camera Matrix："<<endl<<cameraMatrix[1]<<endl;
	std::cout<<endl<<"Left Camera DistCoeffs: "<<endl<<distCoeffs[0]<<endl;
	std::cout<<endl<<"Right Camera DistCoeffs: "<<endl<<distCoeffs[1]<<endl;

	
	// 标定精度检测
	// 通过检查图像上点与另一幅图像的极线的距离来评价标定的精度。为了实现这个目的，使用 undistortPoints 来对原始点做去畸变的处理
	// 随后使用 computeCorrespondEpilines 来计算极线，计算点和线的点积。累计的绝对误差形成了误差
	std::cout<<endl<<" 极线计算...  误差计算... ";
	double err = 0;
    int npoints = 0;
    vector<Vec3f> lines[2];
    for(unsigned int i = 0; i < idx.size(); i++)
    {
        int npt = (int)imagePoints[0][i].size();  //角点个数
        Mat imgpt[2];
        for(int k = 0; k < 2; k++ )
        {
            imgpt[k] = Mat(imagePoints[k][i]);  //
            undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]); // 畸变
            computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);  // 计算极线
        }
        for(int j = 0; j < npt; j++ )
        {
            double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
                                imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
                           fabs(imagePoints[1][i][j].x*lines[0][j][0] +
                                imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
            err += errij;   // 累计误差
        }
        npoints += npt;
    }
    std::cout << "  平均误差 average reprojection err = " <<  err/npoints << endl;  // 平均误差

    // 相机内参数和畸变系数写入文件
    FileStorage fs(intrinsic_filename, CV_STORAGE_WRITE);
    if(fs.isOpened())
    {
        fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
            "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
        fs.release();
    }
    else
        std::cout << "Error: can not save the intrinsic parameters\n";

	// 立体矫正  BOUGUET'S METHOD
    Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];

    cv::stereoRectify(cameraMatrix[0], distCoeffs[0],
                  cameraMatrix[1], distCoeffs[1],
                  imgSize, R, T, R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, 1, imgSize, &validRoi[0], &validRoi[1]);

    fs.open(extrinsic_filename, CV_STORAGE_WRITE);
    if( fs.isOpened() )
    {
        fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
        fs.release();
    }
    else
        std::cout << "Error: can not save the intrinsic parameters\n";
	
	std::cout<<"双目标定完成..."<<endl;
	//getchar();getchar();
	return 0;
}


//------------------------------------------------------
//------------------------------------------------------
// 双目立体匹配和测量
int stereoMatch(int picNum, 
				string intrinsic_filename="intrinsics.yml", 
				string extrinsic_filename="extrinsics.yml", 
				bool no_display=false, 
				string point_cloud_filename="输出/point3D.txt"
				)
{
	//获取待处理的左右相机图像
	int color_mode = 0;
	Mat rawImg = imread(fileList[picNum], color_mode);    //待处理图像  grayScale
	if(rawImg.empty()){
		std::cout<<"In Function stereoMatch, the Image is empty..."<<endl;
		return 0;
	}
	//截取
	Rect leftRect(0, 0, imgSize.width, imgSize.height);
	Rect rightRect(imgSize.width, 0, imgSize.width, imgSize.height);
	Mat img1 = rawImg(leftRect);       //切分得到的左原始图像
	Mat img2 = rawImg(rightRect);      //切分得到的右原始图像
	//图像根据比例缩放
    if(imgScale != 1.f){
        Mat temp1, temp2;
        int method = imgScale < 1 ? INTER_AREA : INTER_CUBIC;
        resize(img1, temp1, Size(), imgScale, imgScale, method);
        img1 = temp1;
        resize(img2, temp2, Size(), imgScale, imgScale, method);
        img2 = temp2;
    }
	imwrite("输出/原始左图像.jpg", img1);
	imwrite("输出/原始右图像.jpg", img2);

    Size img_size = img1.size();

    // reading intrinsic parameters
	FileStorage fs(intrinsic_filename, CV_STORAGE_READ);
    if(!fs.isOpened())
    {
        std::cout<<"Failed to open file "<<intrinsic_filename<<endl;
        return -1;
    }
	Mat M1, D1, M2, D2;      //左右相机的内参数矩阵和畸变系数
    fs["M1"] >> M1;
    fs["D1"] >> D1;
	fs["M2"] >> M2;
	fs["D2"] >> D2;

    M1 *= imgScale;
    M2 *= imgScale;

	// 读取双目相机的立体矫正参数
    fs.open(extrinsic_filename, CV_STORAGE_READ);
    if(!fs.isOpened())
    {
        std::cout<<"Failed to open file  "<<extrinsic_filename<<endl;
        return -1;
    }

	// 立体矫正
	Rect roi1, roi2;
    Mat Q;
    Mat R, T, R1, P1, R2, P2;
    fs["R"] >> R;
    fs["T"] >> T;

	//Alpha取值为-1时，OpenCV自动进行缩放和平移
    cv::stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2 );

	// 获取两相机的矫正映射
    Mat map11, map12, map21, map22;
	initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
    initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

	// 矫正原始图像
    Mat img1r, img2r;
    remap(img1, img1r, map11, map12, INTER_LINEAR);
    remap(img2, img2r, map21, map22, INTER_LINEAR);
    img1 = img1r;
    img2 = img2r;

	// 初始化 stereoBMstate 结构体
	StereoBM bm;
	
	int unitDisparity = 15;//40
	int numberOfDisparities = unitDisparity * 16;
	bm.state->roi1 = roi1;
    bm.state->roi2 = roi2;
    bm.state->preFilterCap = 13;
    bm.state->SADWindowSize = 19;                                     // 窗口大小
    bm.state->minDisparity = 0;                                       // 确定匹配搜索从哪里开始  默认值是0
    bm.state->numberOfDisparities = numberOfDisparities;                // 在该数值确定的视差范围内进行搜索
    bm.state->textureThreshold = 1000;//10                                  // 保证有足够的纹理以克服噪声
    bm.state->uniquenessRatio = 1;     //10                               // !!使用匹配功能模式
    bm.state->speckleWindowSize = 200;   //13                             // 检查视差连通区域变化度的窗口大小, 值为 0 时取消 speckle 检查
    bm.state->speckleRange = 32;    //32                                  // 视差变化阈值，当窗口内视差变化大于阈值时，该窗口内的视差清零，int 型
    bm.state->disp12MaxDiff = -1;

	// 计算
	Mat disp, disp8;
    int64 t = getTickCount();
    bm(img1, img2, disp);
    t = getTickCount() - t;
    printf("立体匹配耗时: %fms\n", t*1000/getTickFrequency());
	
	// 将16位符号整形的视差矩阵转换为8位无符号整形矩阵
    disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));
	
	// 视差图转为彩色图
	Mat vdispRGB = disp8;
	F_Gray2Color(disp8, vdispRGB);
	// 将左侧矫正图像与视差图融合
	Mat merge_mat = F_mergeImg(img1, disp8);

	saveDisp("输出/视差数据.txt", disp);

	//显示
    if(!no_display){
        imshow("左侧矫正图像", img1);
		imwrite("输出/left_undistortRectify.jpg", img1);
        imshow("右侧矫正图像", img2);
		imwrite("输出/right_undistortRectify.jpg", img2);
        imshow("视差图", disp8);
		imwrite("输出/视差图.jpg", disp8);
		imshow("视差图_彩色.jpg", vdispRGB);
		imwrite("输出/视差图_彩色.jpg", vdispRGB);
		imshow("左矫正图像与视差图合并图像", merge_mat);
		imwrite("输出/左矫正图像与视差图合并图像.jpg", merge_mat);
        cv::waitKey();
        std::cout<<endl;
    }
	cv::destroyAllWindows();

	// 视差图转为深度图
	cout<<endl<<"计算深度映射... "<<endl;
    Mat xyz;
    reprojectImageTo3D(disp, xyz, Q, true);    //获得深度图  disp: 720*1280 
	cv::destroyAllWindows(); 
	cout<<endl<<"保存点云坐标... "<<endl;
	saveXYZ(point_cloud_filename, xyz);
    
    cout<<endl<<endl<<"结束"<<endl<<"Press any key to end... ";
	
	getchar(); 
	return 0;
}




int main_test(){
	string intrinsic_filename = "intrinsics.yml";
	string extrinsic_filename = "extrinsics.yml";
	string point_cloud_filename = "输出/point3D.txt";

	/* 立体标定 运行一次即可 */
	initFileList("calib_pic", 1, 26);
	stereoCalibrate(intrinsic_filename, extrinsic_filename); 

	/* 立体匹配 */
	initFileList("test_pic", 1, 2);
	stereoMatch(0, intrinsic_filename, extrinsic_filename, false, point_cloud_filename);

	return 0;
}


