#define MAIN_FILE
#include <commonMacro.h>

#include <OpencvCommon.h>
#include "../common/SequenceMatcher.h"
#include <Eigen/Sparse>
#include <sstream>
#include "../common/stitchingCommonFuc.h"

#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/util.hpp>

cv::Rect AverageMergeForSeam(const std::vector<cv::Mat> &vImage, const std::vector<cv::Mat> &vMask,
					  const std::vector<cv::Rect> &vROI, cv::Mat &result, cv::Mat &resultMask)
{
	//Check the input
	assert(vImage.size() == vROI.size() && vImage.size() == vMask.size());
	//TODO : more checks

	//decide the size of final image
	cv::Point2f resultTl(H_FLOAT_MAX, H_FLOAT_MAX);
	cv::Point2f resultBr(H_FLOAT_MIN, H_FLOAT_MIN);
	for (size_t i = 0; i < vROI.size(); i++)
	{
		cv::Point tl = vROI[i].tl(), br = vROI[i].br();
		if (resultTl.x > tl.x) { resultTl.x = tl.x; }
		if (resultTl.y > tl.y) { resultTl.y = tl.y; }
		if (resultBr.x < br.x) { resultBr.x = br.x; }
		if (resultBr.y < br.y) { resultBr.y = br.y; }
	}
	cv::Rect resultROI(resultTl, resultBr);
	cv::Size resultSize(resultROI.width, resultROI.height);
	result = cv::Mat(resultSize, CV_8UC3, cv::Scalar(0));
	resultMask = cv::Mat(resultSize, CV_8UC1, cv::Scalar(0));
	int shiftX = -resultROI.tl().x, shiftY = -resultROI.tl().y;

	for (size_t k = 0; k < vImage.size(); k++)
	{
		const cv::Mat &mask = vMask[k], &image = vImage[k];
		cv::Rect ROI = vROI[k];
		ROI.x += shiftX;
		ROI.y += shiftY;

		for (size_t i = 0; i < ROI.height; i++)
		{
			const cv::Vec3b *rowImg = reinterpret_cast<const cv::Vec3b *>(image.ptr(i));
			const uchar *rowMask = mask.ptr(i);
			cv::Vec3b *rowResult = reinterpret_cast<cv::Vec3b *>(result.ptr(i + ROI.y));
			uchar *rowResultMask = resultMask.ptr(i + ROI.y);

			for (size_t j = 0; j < ROI.width; j++)
			{
				if (rowMask[j])
				{
					int resultCol = j + ROI.x;
					rowResultMask[resultCol]++;
					int count = rowResultMask[resultCol];
					cv::Vec3d resultColor = rowResult[resultCol];
					resultColor = ((count - 1) * resultColor + cv::Vec3d(rowImg[j])) * (1.0 / count);
					rowResult[resultCol] = resultColor;
				}
			}
		}
	}

	cv::Mat resultSeamMask = cv::Mat(resultSize, CV_8UC1, cv::Scalar(0));
	for (size_t k = 0; k < vImage.size(); k++)
	{
		const cv::Mat &image = vImage[k];
		cv::Mat mask;
		cv::dilate(vMask[k], mask, cv::Mat());
		cv::dilate(vMask[k], mask, cv::Mat());

		cv::Rect ROI = vROI[k];
		ROI.x += shiftX;
		ROI.y += shiftY;

		for (size_t i = 0; i < ROI.height; i++)
		{
			const uchar *rowMask = mask.ptr(i);
			uchar *rowResultSeamMask = resultSeamMask.ptr(i + ROI.y);

			for (size_t j = 0; j < ROI.width; j++)
			{
				if (rowMask[j])
				{
					int resultCol = j + ROI.x;
					rowResultSeamMask[resultCol]++;
				}
			}
		}
	}

	for (size_t i = 0; i < resultSize.height; i++)
	{
		cv::Vec3b *rowResult = reinterpret_cast<cv::Vec3b *>(result.ptr(i));
		uchar *rowResultSeamMask = resultSeamMask.ptr(i);
		for (size_t j = 0; j < resultSize.width; j++)
		{
			if (rowResultSeamMask[j] > 1)
				rowResult[j] = cv::Vec3b(0, 0, 255);
		}
	}

	cv::threshold(resultMask, resultMask, 0, 255, cv::THRESH_BINARY);
	return resultROI;
}

cv::Mat MultibandBlending(std::vector<cv::Mat>& blend_warpeds, std::vector<cv::Mat>& blend_warped_masks,
						  std::vector<cv::Point> &blend_corners, double strength = 100)
{
	int m_blend_type = cv::detail::Blender::MULTI_BAND;
	cv::Ptr<cv::detail::Blender> blender = cv::detail::Blender::createDefault(m_blend_type);
	std::vector<cv::Size> blend_sizes;
	int imgNum = blend_warpeds.size();
	for (size_t i = 0; i < imgNum; i++)
		blend_sizes.push_back(blend_warpeds[i].size());

	cv::Size dst_sz = cv::detail::resultRoi(blend_corners, blend_sizes).size();
	double blend_width = sqrt(static_cast<double>(dst_sz.area())) * strength / 100.f;
	cv::detail::MultiBandBlender* mb = dynamic_cast<cv::detail::MultiBandBlender*>(blender.get());
	mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
	blender->prepare(blend_corners, blend_sizes);
	cv::Mat blend_mask_warped, blend_warped_s;
	for (size_t i = 0; i < imgNum; i++)
	{
		blend_warpeds[i].convertTo(blend_warped_s, CV_16S);
		blend_mask_warped = blend_warped_masks[i]/* & blend_mask_warped*/;
		blender->feed(blend_warped_s, blend_mask_warped, blend_corners[i]);
	}
	cv::Mat multibandResult, multibandMask;
	blender->blend(multibandResult, multibandMask);
	return multibandResult;
}


inline void GetROICorners(const cv::Rect& ROI, std::vector<cv::Point2d>& vCorner)
{
	vCorner.resize(4);
	vCorner[0] = cv::Point2d(ROI.x, ROI.y);
	vCorner[1] = cv::Point2d(ROI.x + ROI.width, ROI.y);
	vCorner[2] = cv::Point2d(ROI.x, ROI.y + ROI.height);
	vCorner[3] = cv::Point2d(ROI.x + ROI.width, ROI.y + ROI.height);
}

int SeamShowWithRealImage(int argc, char *argv[])
{
	std::vector<cv::Mat> images;
	//std::string dir = "HomographDecomposeDIR1";
	std::string dir = "test6";
	if (argc == 2)
		dir = std::string(argv[1]);
	LoadSameSizeImages(images, dir, "JPG");

	

	images.resize(2);
	int height = images[0].rows, width = images[0].cols;

	SequenceMatcher smatcher(SequenceMatcher::F_SIFT);
	std::list<PairInfo> pairinfos;
	smatcher.process(images, pairinfos);

	PairInfo &firstPair = *(pairinfos.begin());
	double threshold = std::min(width, height) * 0.04;
	std::cout << "Homography Ransac threshold = " << threshold << std::endl;
	cv::Mat globalH = firstPair.findHomography(cv::RANSAC, threshold);

	DrawPairInfos(images, pairinfos, !true);

	cv::Size cellNum(50, 50), gridSize(std::ceil(width / double(cellNum.width)), std::ceil(height / double(cellNum.height)));
	cv::Point gridDim(cellNum.width, cellNum.height);
	std::vector<cv::Point2d> vVertices;

	
	EstimateGridVertices(firstPair, globalH, gridDim, gridSize, images, vVertices, 3);

	{
		cv::Mat showTest = images[1].clone();
		DrawGrid(showTest, gridDim, gridSize, 1, 3);

		cv::Mat resultTest = images[1].clone() * 0.6;
		DrawGridVertices(resultTest, cv::Rect(0, 0, resultTest.cols, resultTest.rows), vVertices, gridDim, 2, 3);

		cv::imwrite("resultTest.jpg", resultTest);
		cv::imwrite("showTest.jpg", showTest);
	}

	DrawGrid(images[0], cv::Point(1, 1), images[0].size(), 3, 0, false);
	DrawGrid(images[1], cv::Point(1, 1), images[1].size(), 3, 0, false);
	
	cv::Mat mask1(images[0].size(), CV_8UC1, cv::Scalar(255));
	cv::Mat mask2(images[1].size(), CV_8UC1, cv::Scalar(255));
	

	cv::Mat &image1 = images[firstPair.index1];
	//DrawGrid(extImg0, gridDim, cellSize, 2, 3);

	cv::Mat warpedResult, warpedMask;
	cv::Rect warpedROI;
	GridWarping(image1, cv::Mat(), gridDim, gridSize, vVertices, warpedResult, warpedMask, warpedROI);
	cv::imwrite("warpedResult.jpg", warpedResult);
	cv::imwrite("warpedMask.jpg", warpedMask);

	cv::Mat warpedResultGrid = warpedResult.clone();
	DrawGridVertices(warpedResultGrid, warpedROI, vVertices, gridDim, 1, 1);
	cv::imwrite("warpedResultGrid.jpg", warpedResultGrid);

	cv::Rect img1ROI(0, 0, images[1].cols, images[1].rows);
	cv::Mat img1Mask(images[1].rows, images[1].cols, CV_8UC1, cv::Scalar(255));
	std::vector<cv::Mat> vPreparedImg, vPreparedMask;
	std::vector<cv::Rect> vROI;

	vPreparedImg.push_back(images[1]);
	vPreparedMask.push_back(img1Mask);
	vROI.push_back(img1ROI);
	vPreparedImg.push_back(warpedResult);
	vPreparedMask.push_back(warpedMask);
	vROI.push_back(warpedROI);

	cv::Mat resultImg, resultImgMask;
	AverageMerge(vPreparedImg, vPreparedMask, vROI, resultImg, resultImgMask);
	cv::imwrite("resultImg.jpg", resultImg);
	cv::imwrite("resultImgMask.jpg", resultImgMask);

	DrawPairInfos(images, pairinfos, true);

	std::vector<cv::Point2d> vSrcCorner(4), vSrcWarpedCorner(4);
	GetROICorners(cv::Rect(0, 0, images[0].cols, images[0].rows), vSrcCorner);
	PointSetHTransform(vSrcCorner, globalH, vSrcWarpedCorner);
	cv::Rect img1WarpedROI;
	cv::Mat img1Warped, img1WarpedMask;
	GridWarping(images[0], mask1, cv::Point(1, 1), cv::Size(images[0].size()),
				vSrcWarpedCorner, img1Warped, img1WarpedMask, img1WarpedROI);

	//produce the global result
	if(true)
	{
		cv::Mat globalResult, globalResultMask;

		vPreparedImg[1] = (img1Warped);
		vPreparedMask[1] = (img1WarpedMask);
		vROI[1] = (img1WarpedROI);
		AverageMerge(vPreparedImg, vPreparedMask, vROI, globalResult, globalResultMask);
		cv::imwrite("GlobalHStitching.jpg", globalResult);
	}
	

	cv::Ptr<cv::detail::SeamFinder> seam_finder = cv::makePtr<cv::detail::GraphCutSeamFinder>(cv::detail::GraphCutSeamFinderBase::COST_COLOR);
	
	//std::vector<cv::UMat> seam_warped(2);
	std::vector<cv::UMat> seam_warped_f(2);
	std::vector<cv::UMat> seam_masks_warped(2);
	std::vector<cv::Point> seam_corners(2);
	uchar _setImg1[9] = { 23, 45, 12, 100, 200, 23, 15, 89, 90 };
	uchar _setImg2[9] = { 73, 55, 22, 90, 160, 213, 25, 120, 70 };
	cv::Mat setImg1(3, 3, CV_8UC1, _setImg1);
	cv::Mat setImg2(3, 3, CV_8UC1, _setImg2);
	images[0] = setImg1;
	images[1] = setImg2;

	for (size_t i = 0; i < 2; i++)
	{
		vPreparedImg[i].convertTo(seam_warped_f[i], CV_32F);
		vPreparedMask[i].copyTo(seam_masks_warped[i]);
		seam_corners[i] = vROI[i].tl();
	}
	seam_finder->find(seam_warped_f, seam_corners, seam_masks_warped);
	std::vector<cv::Mat> seam_masks_mat(2);
	for (size_t i = 0; i < 2; i++)
	{
		seam_masks_warped[i].copyTo(seam_masks_mat[i]);

	}

	cv::Mat seamResultImg, seamResultMask;
	AverageMergeForSeam(vPreparedImg, seam_masks_mat, vROI, seamResultImg, seamResultMask);
	cv::imwrite("seamResultImg.jpg", seamResultImg);

	AverageMerge(vPreparedImg, seam_masks_mat, vROI, seamResultImg, seamResultMask);
	cv::imwrite("seamResultImg2.jpg", seamResultImg);

	std::vector<cv::Point> blend_corners(2, cv::Point(0, 0));
	blend_corners[0] = vROI[0].tl();
	blend_corners[1] = vROI[1].tl();
	cv::Mat multibandResult = MultibandBlending(vPreparedImg, seam_masks_mat, blend_corners);
	cv::imwrite("multibandResult.jpg", multibandResult);


	return 0;
}

int SeamShowWithManualData()
{
	cv::Ptr<cv::detail::SeamFinder> seam_finder = cv::makePtr<cv::detail::GraphCutSeamFinder>(cv::detail::GraphCutSeamFinderBase::COST_COLOR);

	//std::vector<cv::UMat> seam_warped(2);
	std::vector<cv::UMat> seam_warped_f(2);
	std::vector<cv::UMat> seam_masks_warped(2);
	std::vector<cv::Point> seam_corners(2, cv::Point(0, 0));
	uchar _setImg1[9] = { 23, 45, 12, 100, 200, 23, 15, 89, 90 };
	uchar _setImg2[9] = { 73, 45, 22, 90, 200, 213, 25, 120, 90 };
	cv::Mat setImg1(3, 3, CV_8UC3);
	cv::Mat setImg2(3, 3, CV_8UC3);
	cv::Mat setImg1U(3, 3, CV_8UC1);
	cv::Mat setImg2U(3, 3, CV_8UC1);
	for (size_t i = 0; i < 3; i++)
	{
		for (size_t j = 0; j < 3; j++)
		{
			int index = i * 3 + j;
			setImg1.at<cv::Vec3b>(i, j) = cv::Vec3b(_setImg1[index], _setImg1[index], _setImg1[index]);
			setImg2.at<cv::Vec3b>(i, j) = cv::Vec3b(_setImg2[index], _setImg2[index], _setImg2[index]);

			setImg1U.at<uchar>(i, j) = _setImg1[index];
			setImg2U.at<uchar>(i, j) = _setImg2[index];

		}
	}

	seam_corners[1].x = 1;

	cv::Mat setImg1_(3, 4, CV_8UC3, cv::Scalar(255, 255, 255));
	setImg1.copyTo(setImg1_(cv::Rect(1, 0, 3, 3)));
	cv::Mat setImg2_(3, 4, CV_8UC3, cv::Scalar(255, 255, 255));
	setImg2.copyTo(setImg2_(cv::Rect(0, 0, 3, 3)));

	setImg1_.convertTo(seam_warped_f[0], CV_32F);
	setImg2_.convertTo(seam_warped_f[1], CV_32F);
	cv::Mat commonMask(3, 4, CV_8UC1, cv::Scalar(255));
	commonMask.copyTo(seam_masks_warped[0]);
	commonMask.copyTo(seam_masks_warped[1]);

	seam_finder->find(seam_warped_f, seam_corners, seam_masks_warped);

	cv::Mat seam_masks_mat1, seam_masks_mat2;

	seam_masks_warped[0].copyTo(seam_masks_mat1);
	seam_masks_warped[1].copyTo(seam_masks_mat2);
}

int main(int argc, char *argv[])
{
	
	SeamShowWithRealImage(argc, argv);

	return 0;
}