#define MAIN_FILE
#include <commonMacro.h>
#include <OpencvCommon.h>
#include "../common/stitchingCommonFuc.h"
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/util.hpp>

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

int main(int argc, char *argv[])
{
	cv::Mat img1 = cv::imread("star_sky0.jpg");
	cv::Mat img2 = cv::imread("star_sky1.jpg");

	std::vector<cv::Mat> vImage(2), vMaskWhole(2);
	std::vector<cv::Rect> vROIWhole(2);
	vImage[0] = img1;
	vImage[1] = img2;
	for (size_t i = 0; i < 2; i++)
	{
		vMaskWhole[i] = cv::Mat(vImage[i].size(), CV_8UC1, cv::Scalar(255));
		vROIWhole[i] = cv::Rect(0, 0, vImage[i].cols, vImage[i].rows);
	}
	
	cv::Mat avgImg, avgImgMask;
	AverageMerge(vImage, vMaskWhole, vROIWhole, avgImg, avgImgMask);
	cv::imwrite("avgImg.jpg", avgImg);

	int w = img1.cols, h = img2.rows;
	std::vector<cv::Mat> vMaskSplited(2);
	vMaskSplited[0] = vMaskWhole[0].clone();
	cv::rectangle(vMaskSplited[0], cv::Rect(w / 2, 0, w - w / 2, h), cv::Scalar(0), -1);
	vMaskSplited[1] = vMaskWhole[1].clone();
	cv::rectangle(vMaskSplited[1], cv::Rect(0, 0, w / 2, h), cv::Scalar(0), -1);
	cv::Mat splitedImg, splitedImgMask;
	AverageMerge(vImage, vMaskSplited, vROIWhole, splitedImg, splitedImgMask);
	cv::imwrite("splitedImg.jpg", splitedImg);

	std::vector<cv::Point> blend_corners(2, cv::Point(0, 0));
	cv::Mat multibandResult = MultibandBlending(vImage, vMaskSplited, blend_corners);
	cv::imwrite("multibandResult.jpg", multibandResult);

	cv::Mat hand = cv::imread("hand.jpg");
	cv::Mat eye = cv::imread("eye.jpg");

	cv::Point eyeCorner(166, 349);
	vImage.clear();
	vImage.push_back(hand);
	vImage.push_back(eye);
	vMaskSplited.clear();
	for (size_t i = 0; i < vImage.size(); i++)
		vMaskSplited[i] = cv::Mat(vImage[i].size(), CV_8UC1, cv::Scalar(255));
	cv::Rect eyeROI(eyeCorner.x, eyeCorner.y, eye.cols, eye.rows);
	cv::rectangle(vMaskSplited[0], eyeROI, cv::Scalar(0), -1);
	blend_corners[1] = eyeCorner;
	cv::Mat eyeHandResult = MultibandBlending(vImage, vMaskSplited, blend_corners, 80);
	cv::imwrite("eyeHandResult.jpg", eyeHandResult);
	cv::Mat eyeHandNaive = hand.clone();
	eye.copyTo(eyeHandNaive(eyeROI));
	cv::imwrite("eyeHandNaive.jpg", eyeHandNaive);
	return 0;
}
