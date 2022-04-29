#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui_c.h>
#include<opencv2/core/core.hpp>
#include<opencv2/calib3d.hpp>
#include<iostream>  
 
using namespace std;
using namespace cv;
 
const int imageWidth = 480;    
const int imageHeight =640;
const double focus = 1.8;
const double baseline = 6;	
double real_distance;




Size imageSize = Size(imageWidth, imageHeight);
 
Mat grayImageL,grayImageR;
Mat rectifyImageL, rectifyImageR;
 
Rect validROIL;//图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域, 其内部的所有像素都有效
Rect validROIR;
 
Mat mapLx, mapLy, mapRx, mapRy;     //映射表  
Mat Rl, Rr, Pl, Pr, Q;              //校正旋转矩阵R，投影矩阵P 重投影矩阵Q
Mat xyz;              //三维坐标
 
Point origin;     //鼠标按下的起始点
Rect selection;      //定义矩形选框
bool selectObject = false;    //是否选择对象
 
int numberOfDisparities = ((imageSize.width / 8) + 15) & -16;
int numDisparities = 6;
 
cv::Ptr<cv::StereoSGBM> sgbm = StereoSGBM::create(0, 16, 3);
 
Mat CML = (Mat_<double>(3,3) <<649.34553,   0.     , 290.56612,
           					   0.     , 650.444  , 262.19935,
           					   0.     ,   0.     ,   1.     );//内参  
Mat DCL = (Mat_<double>(5,1)  << 0.028907, -0.150870, 0.007238, -0.013711, 0.000000);//畸变矩阵
Mat RL = (Mat_<double>(3,3)	 <<1., 0., 0.,
         					   0., 1., 0.,
        					   0., 0., 1.);//旋转矩阵 
Mat PL = (Mat_<double>(3,4)	 <<644.46301,   0.     , 282.37982,   0.     ,
           					   0.     , 652.88019, 264.34808,   0.     ,
           					   0.     ,   0.     ,   1.     ,   0.);//投影矩阵

								   
Mat CMR = (Mat_<double>(3,3) <<649.34553,   0.     , 290.56612,
           					   0.     , 650.444  , 262.19935,
           					   0.     ,   0.     ,   1.     );
Mat DCR = (Mat_<double>(5,1)  << 0.028907, -0.150870, 0.007238, -0.013711, 0.000000);
Mat RR = (Mat_<double>(3,3)	 <<1., 0., 0.,
         					   0., 1., 0.,
        					   0., 0., 1.);
Mat PR = (Mat_<double>(3,4)	 <<644.46301,   0.     , 282.37982,   0.     ,
           					   0.     , 652.88019, 264.34808,   0.     ,
           					   0.     ,   0.     ,   1.     ,   0.);






// //左右目之间的R,t可通过stereoCalibrate()或matlab工具箱calib求得
// Mat T = (Mat_<double>(3, 1) << -119.61078, -0.06806, 0.08105);//T平移向量
// Mat rec = (Mat_<double>(3, 1) << 0.00468, 0.02159, 0.00015);//rec旋转向量
Mat R;//R 旋转矩阵(Stereo)
Mat t;//
Mat frame, f1, f2;
Mat disp, disp8;
  
/*****立体匹配*****/
void stereo_match(int, void*)
{
	sgbm->setPreFilterCap(16);
	int SADWindowSize = 9;
	int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize :3;
	sgbm->setBlockSize(sgbmWinSize);
	int cn = rectifyImageL.channels();
	sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);//8
	sgbm->setP2(32 * cn*sgbmWinSize*sgbmWinSize);//32
	sgbm->setMinDisparity(0);//0
	sgbm->setNumDisparities(numberOfDisparities);//3
	sgbm->setUniquenessRatio(10);//10
	sgbm->setSpeckleWindowSize(100);//100
	sgbm->setSpeckleRange(32);//30
	sgbm->setDisp12MaxDiff(2);//1
	sgbm->setMode(cv::StereoSGBM::MODE_SGBM);
	sgbm->compute(rectifyImageL, rectifyImageR, disp);//输入图像必须为灰度图
 
	disp.convertTo(disp8, CV_8U, 255 / ((numDisparities * 16 + 16)*16.));//计算出的视差是CV_16S格式
	reprojectImageTo3D(disp, xyz, Q, true); //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
	xyz = xyz * 16;
	imshow("disparity", disp8);
}

/*****描述：鼠标操作回调*****/
static void onMouse(int event, int x, int y, int, void*)
{	
	
	if (selectObject)
	{
		selection.x = MIN(x, origin.x);
		selection.y = MIN(y, origin.y);
		selection.width = std::abs(x - origin.x);
		selection.height = std::abs(y - origin.y);
	}
 	
	switch (event)
	{
	case EVENT_LBUTTONDOWN:   //鼠标左按钮按下的事件
		origin = Point(x, y);
		selection = Rect(x, y, 0, 0);
		selectObject = true;
		//cout<<xyz.at<Vec3f>(origin)(1)<<endl;
		real_distance = (focus * baseline / xyz.at<double>(0));
		std::cout<<"!!!Real_Distance::"<<real_distance<<std::endl;
		std::cout<<xyz.at<Vec3f>(origin)(0)<<std::endl;
		// std::cout<<xyz.at<Vec3f>(origin)(2)<<std::endl;
		break;
	case EVENT_LBUTTONUP:    //鼠标左按钮释放的事件
		selectObject = false;
		if (selection.width > 0 && selection.height > 0);
			break;
	}
}

void insertDepth32f(cv::Mat& disp8)
{    
	const int width = disp8.cols;    
	const int height = disp8.rows;    
	float* data = (float*)disp8.data;    
	cv::Mat integralMap = cv::Mat::zeros(height, width, CV_64F);    
	cv::Mat ptsMap = cv::Mat::zeros(height, width, CV_32S);    
	double* integral = (double*)integralMap.data;    
	int* ptsIntegral = (int*)ptsMap.data;    
	memset(integral, 0, sizeof(double) * width * height);    
	memset(ptsIntegral, 0, sizeof(int) * width * height);    
	for (int i = 0; i < height; ++i)    
	{        
		int id1 = i * width;        
		for (int j = 0; j < width; ++j)        
		{            
			int id2 = id1 + j;            
			if (data[id2] > 1e-3)            
			{                
				integral[id2] = data[id2];                
				ptsIntegral[id2] = 1;            
				}        
			}    
		}    // 积分区间    
		for (int i = 0; i < height; ++i)    
		{        
			int id1 = i * width;        
			for (int j = 1; j < width; ++j)        
			{            
				int id2 = id1 + j;            
				integral[id2] += integral[id2 - 1];            
				ptsIntegral[id2] += ptsIntegral[id2 - 1];        
			}    
		}    
		for (int i = 1; i < height; ++i)    
		{        
			int id1 = i * width;        
			for (int j = 0; j < width; ++j)        
			{            
				int id2 = id1 + j;            
				integral[id2] += integral[id2 - width];            
				ptsIntegral[id2] += ptsIntegral[id2 - width];        
			}    
		}    
		int wnd;    
		double dWnd = 2;    
		while (dWnd > 1)    
		{        
			wnd = int(dWnd);        
			dWnd /= 2;        
			for (int i = 0; i < height; ++i)        
			{            
				int id1 = i * width;            
				for (int j = 0; j < width; ++j)            
				{                
					int id2 = id1 + j;                
					int left = j - wnd - 1;                
					int right = j + wnd;                
					int top = i - wnd - 1;                
					int bot = i + wnd;                
					left = max(0, left);                
					right = min(right, width - 1);                
					top = max(0, top);                
					bot = min(bot, height - 1);                
					int dx = right - left;                
					int dy = (bot - top) * width;                
					int idLeftTop = top * width + left;                
					int idRightTop = idLeftTop + dx;                
					int idLeftBot = idLeftTop + dy;                
					int idRightBot = idLeftBot + dx;                
					int ptsCnt = ptsIntegral[idRightBot] + ptsIntegral[idLeftTop] - (ptsIntegral[idLeftBot] + ptsIntegral[idRightTop]);                
					double sumGray = integral[idRightBot] + integral[idLeftTop] - (integral[idLeftBot] + integral[idRightTop]);                
					if (ptsCnt <= 0)                
					{                    continue;                }                
					data[id2] = float(sumGray / ptsCnt);            		
				}        
			}       
				int s = wnd / 2 * 2 + 1;        
				if (s > 201)        
				{            s = 201;        }        
				cv::GaussianBlur(disp8, disp8, cv::Size(s, s), s, s);    
		}
}
/*****主函数*****/
int main()
{	
	// Rodrigues(rec, R); //Rodrigues变换
	// //经过双目标定得到摄像头的各项参数后，采用OpenCV中的stereoRectify(立体校正)得到校正旋转矩阵R、投影矩阵P、重投影矩阵Q
	// //flags-可选的标志有两种零或者 CV_CALIB_ZERO_DISPARITY ,如果设置 CV_CALIB_ZERO_DISPARITY 的话，该函数会让两幅校正后的图像的主点有相同的像素坐标。否则该函数会水平或垂直的移动图像，以使得其有用的范围最大
	// //alpha-拉伸参数。如果设置为负或忽略，将不进行拉伸。如果设置为0，那么校正后图像只有有效的部分会被显示（没有黑色的部分），如果设置为1，那么就会显示整个图像。设置为0~1之间的某个值，其效果也居于两者之间。
	// stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY,
	// 0, imageSize, &validROIL, &validROIR);
	// //std::cout<<R.at<double>(2)<<std::endl;
	// //waitKey(0);
	// //再采用映射变换计算函数initUndistortRectifyMap得出校准映射参数,该函数功能是计算畸变矫正和立体校正的映射变换
	// initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pr, imageSize, CV_32FC1, mapLx, mapLy);
	// initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);
 
	VideoCapture cap(0);
	cap.set(CAP_PROP_FRAME_HEIGHT,480);
	cap.set(CAP_PROP_FRAME_WIDTH, 1280);
	namedWindow("disparity", CV_WINDOW_AUTOSIZE);
	namedWindow("paramemnt", CV_WINDOW_NORMAL);
	createTrackbar("numDisparities:\n", "paramemnt", &numDisparities, 20, stereo_match);
	setMouseCallback("disparity", onMouse, 0);
	
	while (1)
	{
		cap >> frame;
		//imshow("video", frame);
		f1 = frame.colRange(0,640);
		imshow("L",f1);
		f2 = frame.colRange(640,1280);
		imshow("R",f2);

		//通过GFTT来获取角点
		vector<KeyPoint>kp1;
		Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20);//500
		detector->detect(frame, kp1);

		vector<Point2f> pt1, pt2;
		for (auto &kp: kp1) pt1.push_back(kp.pt);
		vector<uchar> status;
		vector<float> error;
		cv::calcOpticalFlowPyrLK(f1, f2, pt1, pt2, status, error);
		cv::Mat E = cv::findEssentialMat(pt1, pt2, CML,RANSAC);//通过特征点和内参解算本质矩阵E
		cv::Mat R1, R2;    
		cv::decomposeEssentialMat(E, R1, R2, t);//通过本质矩阵求解R&t矩阵
		R = R1.clone();    
		t = -t.clone();
		//waitKey(0);
	
		cvtColor(f1, grayImageL, COLOR_BGR2GRAY);
		cvtColor(f2, grayImageR, COLOR_BGR2GRAY);


		cv::stereoRectify(CML,DCL,CMR,DCR,f1.size(), R, -R*t,  R1, R2, PL, PR, Q);



		cv::initUndistortRectifyMap(RL(cv::Rect(0, 0, 3, 3)), DCL, R1, PL(cv::Rect(0, 0, 3, 3)), f1.size(), CV_32FC1, mapLx, mapLy);    
   		remap(grayImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
		cv::imwrite("data/rectifyImageL.png", rectifyImageL);
		cv::initUndistortRectifyMap(RR(cv::Rect(0, 0, 3, 3)), DCR, R2, PR(cv::Rect(0, 0, 3, 3)), f2.size(), CV_32FC1, mapRx, mapRy);    
		remap(grayImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);
		cv::imwrite("data/rectifyImageR.png", rectifyImageR);
		//insertDepth32f(disp8);
		//imshow("disp",disp8);
		stereo_match(0, 0);

		// double distance = focus * baseline / xyz.at<Vec3f>(origin)(0);
		
		//std::cout <<xyz.at<Vec3f>(origin);
		waitKey(1);
	}
	waitKey(0);
	return 0;
}