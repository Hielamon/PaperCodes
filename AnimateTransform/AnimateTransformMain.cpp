#define MAIN_FILE
#include "AnimateTransform.h"
#include "../common/SequenceMatcher.h"
#include <string>

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

//The function to decompose the homograph to 8 reasonable parameters
void DecomposeHomograph(const cv::Mat H, double &theta, double &phi, double &sx, double &sy,
						double &t1, double &t2, double &v1, double &v2)
{
	//-v1*h1 - v2*h4 + h7 = 0
	//-v1*h2 - v2*h5 + h8 = 0
	double *pH = reinterpret_cast<double *>(H.data);
	cv::Point3d line1(-pH[0], -pH[3], pH[6]), line2(-pH[1], -pH[4], pH[7]);
	cv::Point3d crossPt = line1.cross(line2);
	//TODO : check the valid of croosPt.z
	v1 = crossPt.x / crossPt.z;
	v2 = crossPt.y / crossPt.z;

	cv::Mat HpInv = cv::Mat::eye(3, 3, CV_64FC1);
	HpInv.at<double>(2, 0) = -v1;
	HpInv.at<double>(2, 1) = -v2;

	cv::Mat A = HpInv * H;
	A *= 1.0 / (A.at<double>(2, 2));
	cv::Mat RComposed = A(cv::Rect(0, 0, 2, 2));
	cv::Mat W, U, VT;
	cv::SVD::compute(RComposed, W, U, VT);

	phi = -std::atan2(U.at<double>(0, 1), U.at<double>(0, 0));

	sx = W.at<double>(0, 0);
	sy = W.at<double>(1, 0);

	cv::Mat UVt = U * VT;
	theta = std::atan2(UVt.at<double>(0, 1), UVt.at<double>(0, 0));

	cv::Mat A22 = RComposed * UVt.t();
	double *pA22 = reinterpret_cast<double *>(A22.data);
	double t1_ = A.at<double>(0, 2), t2_ = A.at<double>(1, 2);
	//a1*t1 + a2*t2 - t1_ = 0
	//a3*t1 + a4*t2 - t2_ = 0
	line1 = cv::Point3d(pA22[0], pA22[1], -t1_);
	line2 = cv::Point3d(pA22[2], pA22[3], -t2_);
	crossPt = line1.cross(line2);
	//TODO : check the valid of croosPt.z
	t1 = crossPt.x / crossPt.z;
	t2 = crossPt.y / crossPt.z;

	//cv::Mat recoverH = HomographTransform(theta, phi, sx, sy, t1, t2, v1, v2, 1.0);
}

void HomographAnimateTest()
{
	cv::Mat srcImage = cv::imread("2DTransformTest2.jpg");
	cv::Rect ROI1(0, 0, srcImage.cols * 0.6, srcImage.rows);
	cv::Rect ROI2(srcImage.cols * (1 - 0.6), 0, srcImage.cols * 0.6, srcImage.rows);
	cv::Mat subImage1 = srcImage(ROI1).clone();
	cv::Mat subImage2 = srcImage(ROI2).clone();
	ROI1 = DrawGrid(subImage1, cv::Point(1, 1), subImage1.size(), 10, 0, false);
	ROI2 = DrawGrid(subImage2, cv::Point(1, 1), subImage2.size(), 10, 0, false);
	double theta = CV_PI * 0.3, phi = -CV_PI * 0.3, sx = 0.3, sy = 0.7, t1 = 1600, t2 = 0, v1 = -0.0001, v2 = 0.001;

	cv::Mat mask1(subImage1.size(), CV_8UC1, cv::Scalar(255));
	cv::Mat mask2(subImage2.size(), CV_8UC1, cv::Scalar(255));

	int fraction = 7;
	std::shared_ptr<HomographWarper> pHomographWarper = std::make_shared<HomographWarper>(
		theta, phi, sx, sy, t1, t2, v1, v2);

	AnimateTransform animator(std::static_pointer_cast<WarperBase>(pHomographWarper), fraction);

	animator.run(subImage1, mask1, ROI1,
				 subImage2, mask2, ROI2, "2DTransformTestNew");
}

void PTISAnimateTest()
{
	std::vector<cv::Mat> images;
	std::string dir = "IPTSTransformTestDIR2";
	LoadSameSizeImages(images, dir);

	images.resize(2);
	int height = images[0].rows, width = images[0].cols;

	SequenceMatcher smatcher(SequenceMatcher::F_SIFT);
	std::list<PairInfo> pairinfos;
	smatcher.process(images, pairinfos);

	PairInfo &firstPair = *(pairinfos.begin());
	double threshold = std::min(width, height) * 0.04;
	std::cout << "Homography Ransac threshold = " << threshold << std::endl;
	cv::Mat globalH = firstPair.findHomography(cv::RANSAC, threshold);

	cv::Size cellNum(50, 50), gridSize(std::ceil(width / double(cellNum.width)), std::ceil(height / double(cellNum.height)));
	cv::Point gridDim(cellNum.width, cellNum.height);

	cv::Mat image1 = images[firstPair.index2];
	cv::Mat image2 = images[firstPair.index1];

	cv::Rect ROI1(0, 0, image1.cols, image1.rows);
	cv::Rect ROI2(0, 0, image2.cols, image2.rows);

	ROI1 = DrawGrid(image1, cv::Point(1, 1), image1.size(), 10, 0, false);
	ROI2 = DrawGrid(image2, cv::Point(1, 1), image2.size(), 10, 0, false);

	cv::Mat mask1(image1.size(), CV_8UC1, cv::Scalar(255));
	cv::Mat mask2(image2.size(), CV_8UC1, cv::Scalar(255));

	int fraction = 7;
	cv::Vec2d gammaRange(0.05, 0.000), alphaRange(1e-7, 1e-7), betaRange(1e-5, 1e-5);
	std::shared_ptr<PTISWarper> pPTISWarper = std::make_shared<PTISWarper>(
		firstPair, images, globalH, gridDim, gridSize, gammaRange, alphaRange, betaRange);

	AnimateTransform animator(std::static_pointer_cast<WarperBase>(pPTISWarper), fraction);

	animator.run(image1, mask1, ROI1,
				 image2, mask2, ROI2, "IPTSTransformTestGrid1");
}

void DollPTISAnimateTest()
{
	//Prepare for the doll images and Pairinfo
	std::string dir = "DollIPTSTransformTest2";
	int height = 480, width = 360;
	cv::Size imgSize(width, height);
	cv::Size cellNum(5, 5), gridSize(std::ceil(width / double(cellNum.width)), std::ceil(height / double(cellNum.height)));
	cv::Point gridDim(cellNum.width, cellNum.height);
	std::vector<cv::Mat> images(2, cv::Mat(height, width, CV_8UC3, cv::Scalar(0)));
	images[0] = images[1].clone();
	PairInfo firstPair, firstPairHTest;
	firstPair.index1 = 0;
	firstPair.index2 = 1;
	firstPair.pairs_num = 0;

	std::vector<int> vIndex = { 7, 9, 13, 14, 17, 22, 15, 18 };
	//std::vector<int> vIndex = { 25, 34, 38, 55, 56, 71 };
	int maxPointNum = 7;
	cv::Mat H = HomographTransform(0.0, 0.0, 1.0, 1.0, 60, -60, 0, 0, 1.0);
	for (size_t i = 0; i < vIndex.size(); i++)
	{
		int &gridIdx = vIndex[i];
		if (gridIdx < 0 || gridIdx >= gridDim.x * gridDim.y)
		{
			HL_CERR("The grid index is not valid");
		}

		int r = gridIdx / gridDim.x, c = gridIdx - r * gridDim.x;
		cv::Point2d gridTl(c * gridSize.width, r*gridSize.height);
		int pointNum = rand() % maxPointNum;

		int maxNoise = rand() % 15 + 15;
		double noiseTheta = CV_PI * (rand() / double(RAND_MAX));
		cv::Point2d noise1(maxNoise * cos(noiseTheta), maxNoise * sin(noiseTheta));
		cv::Point2d noise2(noise1);

		for (size_t j = 0; j < pointNum; j++)
		{
			int local_x = rand() % gridSize.width;
			int local_y = rand() % gridSize.height;
			cv::Point2d localPt(local_x, local_y);
			cv::Point2d pt1, pt2;
			pt1 = localPt + gridTl;

			PointHTransform(pt1, H, pt2);
			pt2 += noise2;
			if (CheckInSize(pt2, imgSize))
			{
				
				//pt1 += noise1;
				
				firstPair.points1.push_back(pt1);
				firstPair.points2.push_back(pt2);

				cv::Scalar color = RandomColor();
				cv::circle(images[firstPair.index1], pt1, 4, color, -1, cv::LINE_AA);
				cv::circle(images[firstPair.index2], pt2, 4, color, -1, cv::LINE_AA);
				firstPair.pairs_num++;
			}
		}
	}
	firstPair.inliers_num = firstPair.pairs_num;
	firstPair.mask.resize(firstPair.pairs_num, 1);

	cv::Mat image1 = images[firstPair.index2];
	cv::Mat image2 = images[firstPair.index1];

	cv::Mat mask1(image1.size(), CV_8UC1, cv::Scalar(255));
	cv::Mat mask2(image2.size(), CV_8UC1, cv::Scalar(255));

	DrawGrid(image1, cv::Point(1, 1), image1.size(), 3, 0, false);
	//DrawGrid(image2, cv::Point(1, 1), image2.size(), 2, 0, false);

	double threshold = std::min(width, height) * 0.04;
	std::cout << "Homography Ransac threshold = " << threshold << std::endl;

	/////////////////////////////////////////////////////////////////////////////
	//Homograph Procedure
	firstPairHTest = firstPair;
	cv::Point shiftOfImg1(-width * 1.5, 0);
	//Do the Shift operation to points in points1;
	for (size_t i = 0; i < firstPairHTest.pairs_num; i++)
		firstPairHTest.points1[i] += cv::Point2f(shiftOfImg1);

	cv::Mat globalHHTest = firstPairHTest.findHomography(0, threshold);
	cv::Rect ROI1HTest(0, 0, image1.cols, image1.rows);
	cv::Rect ROI2HTest(-image2.cols * 1.5, 0, image2.cols, image2.rows);

	double theta, phi, sx, sy, t1, t2, v1, v2;
	//H.at<double>(0, 2) += width * 1.5;
	DecomposeHomograph(globalHHTest, theta, phi, sx, sy, t1, t2, v1, v2);
	int fractionHTest = 7;
	std::shared_ptr<HomographWarper> pHomographWarper = std::make_shared<HomographWarper>(
		theta, phi, sx, sy, t1, t2, v1, v2);

	AnimateTransform animator(std::static_pointer_cast<WarperBase>(pHomographWarper), fractionHTest, true);

	animator.runCurrentAnimate(image1, mask1, ROI1HTest,
				 image2, mask2, ROI2HTest);

	/////////////////////////////////////////////////////////////////////////////
	//PTIS Procedure
	cv::Mat globalH = firstPair.findHomography(0, threshold);
	cv::Rect ROI1(0, 0, image1.cols, image1.rows);
	cv::Rect ROI2(0, 0, image2.cols, image2.rows);
	gridDim = cv::Point(10, 10);
	gridSize = cv::Size(std::ceil(width / double(gridDim.x)), std::ceil(height / double(gridDim.y)));
	cv::Vec2d gammaRange(0, 10), alphaRange(1e-5, 1e-5), betaRange(1e-5, 1e-5);
	
	//Save a grided image2
	cv::Mat gridImg2 = image2.clone();
	DrawGrid(gridImg2, gridDim, gridSize, 2, 0, false);
	cv::imwrite(dir + "/gridImg2.jpg", gridImg2);

	int fraction = 7;
	std::shared_ptr<PTISWarper> pPTISWarper = std::make_shared<PTISWarper>(
		firstPair, images, globalH, gridDim, gridSize, gammaRange, alphaRange, betaRange);

	animator.resetAnimateParam(pPTISWarper, fraction, true);
	animator.runCurrentAnimate(image1, mask1, ROI1,
				 image2, mask2, ROI2);

	animator.normalizeROI();
	animator.saveResults(dir, true);
}

void HomographDecomposeTest()
{
	std::vector<cv::Mat> images;
	std::string dir = "HomographDecomposeDIR1";
	LoadSameSizeImages(images, dir);

	images.resize(2);
	int height = images[0].rows, width = images[0].cols;

	SequenceMatcher smatcher(SequenceMatcher::F_SIFT);
	std::list<PairInfo> pairinfos;
	smatcher.process(images, pairinfos);

	PairInfo &firstPair = *(pairinfos.begin());
	cv::Point shiftOfImg1(-width * 1.5, 0);
	//Do the Shift operation to points in points1;
	for (size_t i = 0; i < firstPair.pairs_num; i++)
		firstPair.points1[i] += cv::Point2f(shiftOfImg1);
	
	double threshold = std::min(width, height) * 0.04;
	std::cout << "Homography Ransac threshold = " << threshold << std::endl;
	cv::Mat globalH = firstPair.findHomography(cv::RANSAC, threshold);

	cv::Mat image1 = images[firstPair.index2];
	cv::Mat image2 = images[firstPair.index1];

	cv::Rect ROI1(0, 0, image1.cols, image1.rows);
	cv::Rect ROI2(-image2.cols * 1.5, 0, image2.cols, image2.rows);

	cv::Mat mask1(image1.size(), CV_8UC1, cv::Scalar(255));
	cv::Mat mask2(image2.size(), CV_8UC1, cv::Scalar(255));

	int contourWidth = std::max(ROI1.width, ROI1.height) * 0.005;
	DrawGrid(image1, cv::Point(1, 1), image1.size(), contourWidth, 0, false);
	DrawGrid(image2, cv::Point(1, 1), image2.size(), contourWidth, 0, false);
	double theta, phi, sx, sy, t1, t2, v1, v2;
	DecomposeHomograph(globalH, theta, phi, sx, sy, t1, t2, v1, v2);

	int fraction = 7;
	std::shared_ptr<HomographWarper> pHomographWarper = std::make_shared<HomographWarper>(
		theta, phi, sx, sy, t1, t2, v1, v2);

	AnimateTransform animator(std::static_pointer_cast<WarperBase>(pHomographWarper), fraction, true);

	animator.run(image1, mask1, ROI1,
				 image2, mask2, ROI2, "HomographDecomposeTest");
}

/*
int Oldmain(int argc, char *argv)
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
}*/
int main(int argc, char *argv[])
{
	//HomographDecomposeTest();
	//DollPTISAnimateTest();
	PTISAnimateTest();
	return 0;
}