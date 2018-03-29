
//添加使用到的头文件
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv/cv.hpp>
#include <iostream>
#include <fstream>
#include "stdlib.h"
//声明命名空间
using namespace std;
using namespace cv;
using namespace cv::ml;

//!训练数据参数
const int sample_num_perclass = 62;     //训练每类图片数量
const int class_num = 3;                //训练类数
//!所有图片尺寸归一化
const int image_cols = 70;              //定义图片尺寸
const int image_rows = 134;              //定义图片尺寸
//!生成的训练文件保存位置
char SVMName[80] = "H:/Bachor/Graduate/txt/code/SVMSamples/SVM.xml";              //分类器的训练生成的名字,读取时也按照这个名字来
#define RW      0                       //0为读取现有的分类器,1表示重新训练一个分类器
//HOG检测器，用来计算HOG描述子的  
//检测窗口(48,48),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9   
cv::HOGDescriptor hog(cv::Size(48, 48), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);
int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定 

//!程序入口
double Hu[7];       //存储得到的Hu矩阵
Moments mo;         //矩变量
cv::Size size = cv::Size(image_cols, image_rows);
//HOG描述子向量   
std::vector<float> descriptors;
int main(void)
{
#if RW

	Mat trainingLabel = Mat::zeros(sample_num_perclass*class_num, 1, CV_32SC1);
	Mat trainingData;
	char buf[80];                       //字符缓冲区
	for (int i = 0; i<class_num; i++)        //不同了类的循环
	{
		for (int j = 0; j<sample_num_perclass; j++)      //一个类中的图片数量
		{
			//!生成图片的路径(不同类的图片被放在了不同的文件夹下)
			sprintf(buf, "H:/Bachor/Graduate/txt/code/SVMSamples/%d/%d.jpg", i, j + 1);
			cout << buf << endl;

			//!读取
			Mat src = imread(buf, 0);
			//!重设尺寸（归一化）
			Mat reImg;
			resize(src, reImg, size, CV_INTER_CUBIC);



			//计算HOG描述子，检测窗口移动步长(8,8)  
			hog.compute(reImg, descriptors, cv::Size(8, 8));
			if (i == 0 || j ==0)
			{
				//HOG描述子的维数   
				DescriptorDim = descriptors.size();
				//!读取训练数据
				trainingData = Mat::zeros(sample_num_perclass*class_num, DescriptorDim + 7, CV_32FC1);          //填入图像的7个Hu矩
			}

			//Mat canny;
			// Canny(reImg, canny, 150, 200);
			//reImg = canny;
			//imshow("gray",reImg);
			//imshow("canny",canny);
			//waitKey(0);
			//!求Hu矩
			mo = moments(reImg);
			cv::HuMoments(mo, Hu);
			//!将Hu矩填入训练数据集里
			float *dstPoi = trainingData.ptr<float>(i*sample_num_perclass + j);  //指向源的指针
			for (int r = 0; r<7; r++)
				dstPoi[r] = (float)Hu[r];

			int num = 0;

			//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat    
			for (int i = 7; i < DescriptorDim+7; i++)
			{

				//第num个样本的特征向量中的第i个元素   
				dstPoi[i] = (float)descriptors[num];
				num++;
			}

			//!添加对该数据的分类标签
			int *labPoi = trainingLabel.ptr<int>(i*sample_num_perclass + j);
			labPoi[0] = i;
		}
	}
	imwrite("H:/Bachor/Graduate/txt/code/SVMSamples/res.jpg", trainingData);

	//!创建SVM支持向量机并训练数据
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setC(0.01);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10000, 1e-6));
	svm->train(trainingData, ROW_SAMPLE, trainingLabel);
	svm->save(SVMName);
#else
	//读取xml文件
	Ptr<SVM> svm = SVM::load(SVMName);
#endif
	//!读取一副图片进行测试
	Mat temp = imread("H:/Bachor/Graduate/txt/code/SVMSamples/2/5.jpg", 0);

	Mat dst;
	resize(temp, dst, size, CV_INTER_CUBIC);
	Mat canny;
	Canny(dst, canny, 50, 150);
	//waitKey(0);
	mo = moments(canny);
	//	imshow("dst", dst);
	//	waitKey(0);
	cv::HuMoments(mo, Hu);


	hog.compute(dst, descriptors, cv::Size(8, 8));
	Mat pre(1, descriptors.size()+7, CV_32FC1);

	float *p = pre.ptr<float>(0);
	for (int i = 0; i<7; i++)
		p[i] = Hu[i];

	int num = 0;
	//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat    
	for (int i = 7; i < DescriptorDim + 7; i++)
	{
		//第num个样本的特征向量中的第i个元素   
		p[i] = (float)descriptors[num];
		num++;
	}

	float res = svm->predict(pre);
	// cv::imshow(canny);
	cout << res << endl;
	system("pause");
	return 0;
}
