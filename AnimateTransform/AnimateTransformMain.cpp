#include "../common/stitchingCommonFuc.h"
#include <sstream>
#include <iomanip>
#include <direct.h>
#include <io.h>

cv::Mat SimilarityTransform(double theta, double sx, double sy, double t1, double t2, double tRatio)
{
	cv::Mat S = cv::Mat::eye(3, 3, CV_64FC1);
	theta *= tRatio;
	t1 *= tRatio;
	t2 *= tRatio;
	sx = (sx - 1)*tRatio + 1;
	sy = (sy - 1)*tRatio + 1;

	S.at<double>(0, 2) = t1;
	S.at<double>(1, 2) = t2;
	double s = sin(theta), c = cos(theta);
	S.at<double>(0, 0) = c * sx;
	S.at<double>(0, 1) = s * sx;
	S.at<double>(1, 0) = -s * sy;
	S.at<double>(1, 1) = c * sy;

	return S;
}

cv::Mat Get2DRotation(double theta)
{
	cv::Mat R2D = cv::Mat::eye(2, 2, CV_64FC1);
	double s = sin(theta), c = cos(theta);
	R2D.at<double>(0, 0) = c;
	R2D.at<double>(1, 0) = -s;
	R2D.at<double>(0, 1) = s;
	R2D.at<double>(1, 1) = c;
	return R2D;
}

cv::Mat HomographTransform(double theta, double phi, double sx, double sy, double t1, double t2, double v1, double v2, double tRatio)
{
	theta *= tRatio;
	phi *= tRatio;
	sx = (sx - 1)*tRatio + 1;
	sy = (sy - 1)*tRatio + 1;
	t1 *= tRatio;
	t2 *= tRatio;
	v1 *= tRatio;
	v2 *= tRatio;

	cv::Mat Hs = cv::Mat::eye(3, 3, CV_64FC1);
	cv::Mat thetaR = Get2DRotation(theta);
	thetaR.copyTo(Hs(cv::Rect(0, 0, 2, 2)));
	Hs.at<double>(0, 2) = t1;
	Hs.at<double>(1, 2) = t2;

	cv::Mat Ha = cv::Mat::eye(3, 3, CV_64FC1);
	cv::Mat phiR = Get2DRotation(phi), phiR_ = Get2DRotation(-phi);
	cv::Mat Scale = cv::Mat::eye(2, 2, CV_64FC1);
	Scale.at<double>(0, 0) = sx;
	Scale.at<double>(1, 1) = sy;
	Ha(cv::Rect(0, 0, 2, 2)) = phiR_ * Scale * phiR;

	cv::Mat Hp = cv::Mat::eye(3, 3, CV_64FC1);
	Hp.at<double>(2, 0) = v1;
	Hp.at<double>(2, 1) = v2;

	return Hp * Ha * Hs;
}


void GetROICorners(const cv::Rect& ROI, std::vector<cv::Point2d>& vCorner)
{
	vCorner.resize(4);
	vCorner[0] = cv::Point2d(ROI.x, ROI.y);
	vCorner[1] = cv::Point2d(ROI.x + ROI.width, ROI.y);
	vCorner[2] = cv::Point2d(ROI.x, ROI.y + ROI.height);
	vCorner[3] = cv::Point2d(ROI.x + ROI.width, ROI.y + ROI.height);
}

template <class T>
bool PointSetHTransform(const std::vector<cv::Point_<T>> &vSrcPt, const cv::Mat &H,
						std::vector<cv::Point_<T>> &vDstPt)
{
	vDstPt.resize(vSrcPt.size());
	bool valid = true;
	for (size_t i = 0; i < vSrcPt.size() && valid; i++)
	{
		valid &= PointHTransform(vSrcPt[i], H, vDstPt[i]);
	}
	return valid;
}

int main(int argc, char *argv)
{
	cv::Mat srcImage = cv::imread("2DTransformTest2.jpg");
	cv::Rect ROI1(0, 0, srcImage.cols * 0.6, srcImage.rows);
	cv::Rect ROI2(srcImage.cols * (1 - 0.6) , 0, srcImage.cols * 0.6, srcImage.rows);
	cv::Mat subImage1 = srcImage(ROI1).clone();
	cv::Mat subImage2 = srcImage(ROI2).clone();
	//cv::rectangle(subImage2, cv::Rect(0, 0, subImage2.cols, subImage2.rows), cv::Scalar(0, 0, 255), 10);
	//cv::rectangle(subImage1, cv::Rect(0, 0, subImage1.cols, subImage1.rows), cv::Scalar(0, 255, 255), 10);
	DrawGrid(subImage1, cv::Point(1, 1), subImage1.size(), 10, 0, false);
	double theta = CV_PI * 0.3, phi = -CV_PI * 0.3, sx = 0.3, sy = 0.7, t1 = 1600, t2 = 0, v1 = -0.0001, v2 = 0.001;

	int fraction = 7;

	std::vector<cv::Mat> vResult;
	std::vector<cv::Rect> vResultROI;
	
	for (size_t i = 0; i <= fraction; i++)
	{
		double tRatio = double(fraction - i) / fraction;
		//cv::Mat S = SimilarityTransform(theta, sx, sy, t1, t2, tRatio);
		cv::Mat S = HomographTransform(theta, phi, sx, sy, t1, t2, v1, v2, tRatio);
		std::vector<cv::Point2d> vROI2Corner(4), vROI2WarpedCorner(4);
		GetROICorners(ROI2, vROI2Corner);
		PointSetHTransform(vROI2Corner, S, vROI2WarpedCorner);

		cv::Mat result, resultMask;
		cv::Mat subMask1 = cv::Mat(subImage1.size(), CV_8UC1, cv::Scalar(255));
		cv::Mat subMask2 = cv::Mat(subImage2.size(), CV_8UC1, cv::Scalar(255));
		cv::Rect resultROI;
		GridWarping(subImage2, subMask2, cv::Point(1, 1), cv::Size(subImage2.size()),
					vROI2WarpedCorner, result, resultMask, resultROI);
		

		std::vector<cv::Mat> vPreparedImg, vPreparedMask;
		std::vector<cv::Rect> vROI;


		vPreparedImg.push_back(subImage1);
		vPreparedMask.push_back(subMask1);
		vROI.push_back(ROI1);
		vPreparedImg.push_back(result);
		vPreparedMask.push_back(resultMask);
		vROI.push_back(resultROI);

		cv::Mat resultImg, resultImgMask;
		cv::Rect currResultROI;
		currResultROI = AverageMerge(vPreparedImg, vPreparedMask, vROI, resultImg, resultImgMask);
		currResultROI = DrawGridVertices(resultImg, currResultROI, vROI2WarpedCorner, cv::Point(1, 1), 10, 0, false);

		vResultROI.push_back(currResultROI);
		vResult.push_back(resultImg);
	}


	cv::Rect finalROI = vResultROI[0];
	for (size_t i = 1; i < vResultROI.size(); i++)
		finalROI = GetUnionRoi(finalROI, vResultROI[i]);
	
	for (size_t i = 0; i < vResult.size(); i++)
	{
		cv::Mat tmp(finalROI.height, finalROI.width, CV_8UC3, cv::Scalar(0));
		cv::Rect tmpROI = vResultROI[i];
		tmpROI.x -= finalROI.x;
		tmpROI.y -= finalROI.y;
		vResult[i].copyTo(tmp(tmpROI));
		vResult[i] = tmp;
	}

	std::string foldername = "2DTransformTest";
	//Check and Create the folder
	{
		if (_access(foldername.c_str(), 0) == -1)
		{
			_mkdir(foldername.c_str());
		}
	}

	std::stringstream ioStr;
	for (size_t i = 0; i < vResult.size(); i++)
	{
		ioStr.str("");
		resizeShow("test", vResult[i], true);
		cv::waitKey(0);
		ioStr << foldername << "/result_" << std::setw(4) << std::setfill('0') << i << ".jpg";
		cv::imwrite(ioStr.str(), vResult[i]);
	}

	return 0;
}