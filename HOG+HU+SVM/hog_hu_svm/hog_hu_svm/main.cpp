
//���ʹ�õ���ͷ�ļ�
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv/cv.hpp>
#include <iostream>
#include <fstream>
#include "stdlib.h"
//���������ռ�
using namespace std;
using namespace cv;
using namespace cv::ml;

//!ѵ�����ݲ���
const int sample_num_perclass = 62;     //ѵ��ÿ��ͼƬ����
const int class_num = 3;                //ѵ������
//!����ͼƬ�ߴ��һ��
const int image_cols = 70;              //����ͼƬ�ߴ�
const int image_rows = 134;              //����ͼƬ�ߴ�
//!���ɵ�ѵ���ļ�����λ��
char SVMName[80] = "H:/Bachor/Graduate/txt/code/SVMSamples/SVM.xml";              //��������ѵ�����ɵ�����,��ȡʱҲ�������������
#define RW      0                       //0Ϊ��ȡ���еķ�����,1��ʾ����ѵ��һ��������
//HOG���������������HOG�����ӵ�  
//��ⴰ��(48,48),��ߴ�(16,16),�鲽��(8,8),cell�ߴ�(8,8),ֱ��ͼbin����9   
cv::HOGDescriptor hog(cv::Size(48, 48), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);
int DescriptorDim;//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin�������� 

//!�������
double Hu[7];       //�洢�õ���Hu����
Moments mo;         //�ر���
cv::Size size = cv::Size(image_cols, image_rows);
//HOG����������   
std::vector<float> descriptors;
int main(void)
{
#if RW

	Mat trainingLabel = Mat::zeros(sample_num_perclass*class_num, 1, CV_32SC1);
	Mat trainingData;
	char buf[80];                       //�ַ�������
	for (int i = 0; i<class_num; i++)        //��ͬ�����ѭ��
	{
		for (int j = 0; j<sample_num_perclass; j++)      //һ�����е�ͼƬ����
		{
			//!����ͼƬ��·��(��ͬ���ͼƬ�������˲�ͬ���ļ�����)
			sprintf(buf, "H:/Bachor/Graduate/txt/code/SVMSamples/%d/%d.jpg", i, j + 1);
			cout << buf << endl;

			//!��ȡ
			Mat src = imread(buf, 0);
			//!����ߴ磨��һ����
			Mat reImg;
			resize(src, reImg, size, CV_INTER_CUBIC);



			//����HOG�����ӣ���ⴰ���ƶ�����(8,8)  
			hog.compute(reImg, descriptors, cv::Size(8, 8));
			if (i == 0 || j ==0)
			{
				//HOG�����ӵ�ά��   
				DescriptorDim = descriptors.size();
				//!��ȡѵ������
				trainingData = Mat::zeros(sample_num_perclass*class_num, DescriptorDim + 7, CV_32FC1);          //����ͼ���7��Hu��
			}

			//Mat canny;
			// Canny(reImg, canny, 150, 200);
			//reImg = canny;
			//imshow("gray",reImg);
			//imshow("canny",canny);
			//waitKey(0);
			//!��Hu��
			mo = moments(reImg);
			cv::HuMoments(mo, Hu);
			//!��Hu������ѵ�����ݼ���
			float *dstPoi = trainingData.ptr<float>(i*sample_num_perclass + j);  //ָ��Դ��ָ��
			for (int r = 0; r<7; r++)
				dstPoi[r] = (float)Hu[r];

			int num = 0;

			//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat    
			for (int i = 7; i < DescriptorDim+7; i++)
			{

				//��num�����������������еĵ�i��Ԫ��   
				dstPoi[i] = (float)descriptors[num];
				num++;
			}

			//!��ӶԸ����ݵķ����ǩ
			int *labPoi = trainingLabel.ptr<int>(i*sample_num_perclass + j);
			labPoi[0] = i;
		}
	}
	imwrite("H:/Bachor/Graduate/txt/code/SVMSamples/res.jpg", trainingData);

	//!����SVM֧����������ѵ������
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setC(0.01);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10000, 1e-6));
	svm->train(trainingData, ROW_SAMPLE, trainingLabel);
	svm->save(SVMName);
#else
	//��ȡxml�ļ�
	Ptr<SVM> svm = SVM::load(SVMName);
#endif
	//!��ȡһ��ͼƬ���в���
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
	//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat    
	for (int i = 7; i < DescriptorDim + 7; i++)
	{
		//��num�����������������еĵ�i��Ԫ��   
		p[i] = (float)descriptors[num];
		num++;
	}

	float res = svm->predict(pre);
	// cv::imshow(canny);
	cout << res << endl;
	system("pause");
	return 0;
}
