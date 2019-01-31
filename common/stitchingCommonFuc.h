#pragma once
#include <OpencvCommon.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Sparse>
#include <commonMacro.h>
#include <algorithm>
#include "SequenceMatcher.h"
#include <commonMacro.h>

//The Mapping function use the inverse Homograph matrix
inline void HomographMapping(cv::Mat Hinv, cv::Rect srcROI, cv::Rect dstROI,
							 const cv::Mat &src, cv::Mat &dst,
							 const cv::Mat &srcMask, cv::Mat &dstMask)
{
	assert(src.rows == srcROI.height && src.cols == srcROI.width);
	assert(dst.rows == dstROI.height && dst.cols == dstROI.width);
	assert(srcMask.rows == srcROI.height && srcMask.cols == srcROI.width);
	assert(dstMask.rows == dstROI.height && dstMask.cols == dstROI.width);
	assert(src.type() == CV_8UC3 && dst.type() == CV_8UC3);
	assert(srcMask.type() == CV_8UC1 && dstMask.type() == CV_8UC1);
	cv::Point2d srcPt, dstPt;
	cv::Size srcSize(src.size());
	for (int i = 0; i < dstROI.height; i++)
	{
		cv::Vec3b *dstRow = reinterpret_cast<cv::Vec3b *>(dst.ptr(i));
		uchar *dstMaskRow = dstMask.ptr(i);
		for (int j = 0; j < dstROI.width; j++)
		{
			dstPt.x = dstROI.x + j;
			dstPt.y = dstROI.y + i;
			PointHTransform(dstPt, Hinv, srcPt);
			srcPt.x = floor(srcPt.x - srcROI.x);
			srcPt.y = floor(srcPt.y - srcROI.y);
			if (CheckInSize(srcPt, srcSize) && srcMask.at<uchar>(srcPt.y, srcPt.x))
			{
				dstRow[j] = src.at<cv::Vec3b>(srcPt.y, srcPt.x);
				dstMaskRow[j] = 255;
			}
		}
	}


}

//GridMapping function which is used to mapping the grid
//vCornerPt is the vector(4) in the destination image ROI, which is correspond to the vGridPt vector(4) in src
//dstROIShift is the shift magnitude for moving the dstROI's tl point to (0, 0)
inline bool GridMapping(const std::vector<cv::Point2d> &vCornerPt, const std::vector<cv::Point2d> &vGridPt,
						cv::Point dstROIShift, const cv::Mat &src, cv::Mat &dst,
						const cv::Mat &srcMask, cv::Mat &dstMask, bool processTwist = false)
{

	//Check the vector size
	if (vCornerPt.size() != 4 || vGridPt.size() != 4)
		HL_CERR("The input vCornerPt and vGridPt must only contains 4 elements");

	//Draw grid for test
	/*cv::line(dst, vCornerPt[0] + cv::Point2d(dstROIShift), vCornerPt[1] + cv::Point2d(dstROIShift), cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
	cv::line(dst, vCornerPt[1] + cv::Point2d(dstROIShift), vCornerPt[3] + cv::Point2d(dstROIShift), cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
	cv::line(dst, vCornerPt[3] + cv::Point2d(dstROIShift), vCornerPt[2] + cv::Point2d(dstROIShift), cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
	cv::line(dst, vCornerPt[2] + cv::Point2d(dstROIShift), vCornerPt[0] + cv::Point2d(dstROIShift), cv::Scalar(255, 255, 0), 1, cv::LINE_AA);*/

	//Check the validity of vCornerPt and vGridPt
	cv::Size srcSize = src.size(), dstSize = dst.size();
	bool ptValid = true;
	for (size_t m = 0; m < 4; m++)
	{
		ptValid &= CheckInSize(vGridPt[m], srcSize);
		ptValid &= CheckInSize(vCornerPt[m] + cv::Point2d(dstROIShift), dstSize);
	}

	if (!ptValid)
		HL_CERR("The input point exceed the image size");

	//Analysis the grid quadrilateral
	std::vector<cv::Vec3d> vCornerPt_(4);
	for (size_t m = 0; m < 4; m++)
		vCornerPt_[m] = cv::Vec3d(vCornerPt[m].x, vCornerPt[m].y, 1);

	//Get the cross point of diagonal line
	cv::Vec3d line03 = vCornerPt_[0].cross(vCornerPt_[3]);
	cv::Vec3d line12 = vCornerPt_[1].cross(vCornerPt_[2]);
	cv::Vec3d crossPt_ = line03.cross(line12);
	if (std::abs(crossPt_.dot(crossPt_)) <= 1e-16)
	{
		//std::cout << "std::abs(crossPt_.dot(crossPt_)) <= 1e-16" << std::endl;
		return false;
	}

	cv::Point2d crossPt(crossPt_[0] / crossPt_[2], crossPt_[1] / crossPt_[2]);

	//Get the delata vector from grid corner to the cross point
	std::vector<cv::Vec2d> vDelta(4);
	for (size_t m = 0; m < 4; m++)
		vDelta[m] = cv::Vec2d(crossPt.x - vCornerPt[m].x, crossPt.y - vCornerPt[m].y);

	//decide whether the cross point on the diagonal lines
	double proof03 = vDelta[0].dot(vDelta[3]);
	double proof12 = vDelta[1].dot(vDelta[2]);
	cv::Point cornerTl, cornerBr;
	if (proof03 < 0 && proof12 < 0)
	{
		cornerTl = vCornerPt[0];
		cornerBr = cornerTl;
		for (size_t m = 0; m < 4; m++)
		{
			const cv::Point2d &pt = vCornerPt[m];
			if (cornerTl.x > pt.x)cornerTl.x = floor(pt.x);
			if (cornerTl.y > pt.y)cornerTl.y = floor(pt.y);
			if (cornerBr.x < pt.x)cornerBr.x = ceil(pt.x);
			if (cornerBr.y < pt.y)cornerBr.y = ceil(pt.y);
		}

		std::vector<uchar> inlierMask(4, 1);
		cv::Mat gridHinv = cv::findHomography(vCornerPt, vGridPt, inlierMask);

		cv::Rect srcGridROI(vGridPt[0], vGridPt[3]), dstGridROI(cornerTl, cornerBr),
			dstGridROI_(cornerTl + dstROIShift, cornerBr + dstROIShift);

		HomographMapping(gridHinv, srcGridROI, dstGridROI, src(srcGridROI), dst(dstGridROI_), srcMask(srcGridROI), dstMask(dstGridROI_));
	}
	else if (proof03 > 0 && proof12 > 0)
	{
		//The quadrangle is twisty, then we change the order of vCornerPt
		//and mapping the new grid vCornerTwistyPt which just change the vCornerPt's 3 and 1 pos
		//std::cout << "twisty situation" << std::endl;
		if (processTwist)
		{

			bool find_mapping = false;
			for (size_t m = 1; m < 4; m++)
			{
				std::vector<cv::Point2d> vCornerTwistyPt = vCornerPt;
				vCornerTwistyPt[0] += vCornerTwistyPt[m];
				vCornerTwistyPt[m] = vCornerTwistyPt[0] - vCornerTwistyPt[m];
				vCornerTwistyPt[0] = vCornerTwistyPt[0] - vCornerTwistyPt[m];

				//TODO: Maybe there is a more reasonable strategy
				if ((find_mapping = GridMapping(vCornerTwistyPt, vGridPt, dstROIShift, src, dst, srcMask, dstMask)))
					break;
			}
			return find_mapping;
		}
		else
			return false;
		//continue;
	}
	else if (proof03 == 0 && proof12 == 0)
	{
		//Only choose one bigest triangle for warp
		//std::cout << "Only choose one bigest triangle for warp" << std::endl;
		int maxIdx1 = 0, maxIdx2 = 0, minIdx = 0;
		double max1 = 0, max2 = 0, min = vDelta[0].dot(vDelta[0]);
		for (int m = 0; m < 4; m++)
		{
			double distant2 = vDelta[m].dot(vDelta[m]);
			if (distant2 > max1)
			{
				max2 = max1;
				maxIdx2 = maxIdx1;
				max1 = distant2;
				maxIdx1 = m;
			}
			else if (distant2 > max2)
			{
				max2 = distant2;
				maxIdx2 = m;
			}

			if (distant2 < min)
			{
				min = distant2;
				minIdx = m;
			}
		}

		cv::Point2f vTrianglePt[3], vTriangleGridPt[3];
		vTrianglePt[0] = crossPt;
		vTrianglePt[1] = vCornerPt[maxIdx1];
		vTrianglePt[2] = vCornerPt[maxIdx2];

		vTriangleGridPt[0] = vGridPt[minIdx];
		vTriangleGridPt[1] = vGridPt[maxIdx1];
		vTriangleGridPt[2] = vGridPt[maxIdx2];

		cornerTl.x = floor(std::min(crossPt.x, std::min(vCornerPt[maxIdx1].x, vCornerPt[maxIdx2].x)));
		cornerTl.y = floor(std::min(crossPt.y, std::min(vCornerPt[maxIdx1].y, vCornerPt[maxIdx2].y)));
		cornerBr.x = ceil(std::max(crossPt.x, std::max(vCornerPt[maxIdx1].x, vCornerPt[maxIdx2].x)));
		cornerBr.y = ceil(std::max(crossPt.y, std::max(vCornerPt[maxIdx1].y, vCornerPt[maxIdx2].y)));

		cv::Mat warpAffine = cv::getAffineTransform(vTrianglePt, vTriangleGridPt);
		cv::Mat warpH = cv::Mat::eye(3, 3, CV_64FC1);
		warpAffine.copyTo(warpH(cv::Rect(0, 0, 3, 2)));

		cv::Rect srcGridROI(vGridPt[0], vGridPt[3]), dstGridROI(cornerTl, cornerBr),
			dstGridROI_(cornerTl + dstROIShift, cornerBr + dstROIShift);

		HomographMapping(warpH, srcGridROI, dstGridROI, src(srcGridROI), dst(dstGridROI_), srcMask(srcGridROI), dstMask(dstGridROI_));
	}
	else
	{
		std::vector<int> vIdx1 = { 0, 3 }, vIdx2 = { 1, 2 };

		std::vector<int> &digIdx = proof03 < 0 ? vIdx2 : vIdx1;
		std::vector<int> &angIdx = proof03 < 0 ? vIdx1 : vIdx2;

		//std::cout << "proof03 = " << proof03 << "    proof12 = " << proof12 << std::endl;

		for (size_t m = 0; m < 2; m++)
		{
			std::vector<int> tIdx = { digIdx[0], digIdx[1], angIdx[m] };
			cv::Point2f vTrianglePt[3], vTriangleGridPt[3];
			cornerTl = vCornerPt[tIdx[0]];
			cornerBr = cornerTl;
			for (size_t n = 0; n < 3; n++)
			{
				int &idxTmp = tIdx[n];
				vTrianglePt[n] = vCornerPt[idxTmp];
				vTriangleGridPt[n] = vGridPt[idxTmp];

				cornerTl.x = std::min(cornerTl.x, int(floor(vTrianglePt[n].x)));
				cornerTl.y = std::min(cornerTl.y, int(floor(vTrianglePt[n].y)));
				cornerBr.x = std::max(cornerBr.x, int(ceil(vTrianglePt[n].x)));
				cornerBr.y = std::max(cornerBr.y, int(ceil(vTrianglePt[n].y)));
			}

			cv::Mat warpAffine = cv::getAffineTransform(vTrianglePt, vTriangleGridPt);
			cv::Mat warpH = cv::Mat::eye(3, 3, CV_64FC1);
			warpAffine.copyTo(warpH(cv::Rect(0, 0, 3, 2)));

			cv::Rect srcGridROI(vGridPt[0], vGridPt[3]), dstGridROI(cornerTl, cornerBr),
				dstGridROI_(cornerTl + dstROIShift, cornerBr + dstROIShift);

			HomographMapping(warpH, srcGridROI, dstGridROI, src(srcGridROI), dst(dstGridROI_), srcMask(srcGridROI), dstMask(dstGridROI_));
		}
	}



	//Draw grid for test
	/*cv::line(dstMask, vCornerPt[0] + cv::Point2d(dstROIShift), vCornerPt[1] + cv::Point2d(dstROIShift), cv::Scalar(100), 1, cv::LINE_AA);
	cv::line(dstMask, vCornerPt[1] + cv::Point2d(dstROIShift), vCornerPt[3] + cv::Point2d(dstROIShift), cv::Scalar(200), 1, cv::LINE_AA);
	cv::line(dstMask, vCornerPt[3] + cv::Point2d(dstROIShift), vCornerPt[2] + cv::Point2d(dstROIShift), cv::Scalar(100), 1, cv::LINE_AA);
	cv::line(dstMask, vCornerPt[2] + cv::Point2d(dstROIShift), vCornerPt[0] + cv::Point2d(dstROIShift), cv::Scalar(200), 1, cv::LINE_AA);*/

	return true;
}

inline void GridWarping(const cv::Mat& src, const cv::Mat& srcMask, const cv::Point& gridDim, const cv::Size &gridSize,
						const std::vector<cv::Point2d> &vVertices,
						cv::Mat &dst, cv::Mat &mask, cv::Rect &resultROI)
{
	assert(src.type() == CV_8UC3);
	assert(srcMask.type() == CV_8UC1 || srcMask.empty());
	assert(vVertices.size() == (gridDim.x + 1) * (gridDim.y + 1));

	cv::Size srcSize_(gridDim.x*gridSize.width + 1, gridDim.y*gridSize.height + 1);
	srcSize_.width = src.cols > srcSize_.width ? src.cols : srcSize_.width;
	srcSize_.height = src.rows > srcSize_.height ? src.rows : srcSize_.height;
	cv::Mat src_(srcSize_, CV_8UC3, cv::Scalar(0));
	cv::Mat srcMask_(srcSize_, CV_8UC1, cv::Scalar(0));
	cv::Rect srcROI(0, 0, src.cols, src.rows);
	src.copyTo(src_(srcROI));
	if (!srcMask.empty() && srcMask.size() == src.size())
		srcMask.copyTo(srcMask_(srcROI));
	else
		cv::rectangle(srcMask_, srcROI, cv::Scalar(255), -1);

	cv::Point gridTl(vVertices[0]), gridBr = gridTl;
	for (int i = 0; i < vVertices.size(); i++)
	{
		const cv::Point2d &pt = vVertices[i];
		if (gridTl.x > pt.x)gridTl.x = floor(pt.x);
		if (gridTl.y > pt.y)gridTl.y = floor(pt.y);
		if (gridBr.x < pt.x)gridBr.x = ceil(pt.x);
		if (gridBr.y < pt.y)gridBr.y = ceil(pt.y);
	}

	resultROI = cv::Rect(gridTl, gridBr);
	cv::Point vertShift = -resultROI.tl();

	//To avoid the size check;
	resultROI.width++;
	resultROI.height++;

	dst = cv::Mat(resultROI.height, resultROI.width, CV_8UC3, cv::Scalar(0));
	mask = cv::Mat(resultROI.height, resultROI.width, CV_8UC1, cv::Scalar(0));

	std::vector<int> vCornerIdx(4, 0);
	vCornerIdx[1] = 1;
	vCornerIdx[2] = gridDim.x + 1;
	vCornerIdx[3] = vCornerIdx[2] + 1;
	std::vector<cv::Point2d> vCornerPt(4), vGridPt(4, cv::Point2d(0, 0));
	vGridPt[1] = cv::Point2d(gridSize.width, 0);
	vGridPt[2] = cv::Point2d(0, gridSize.height);
	vGridPt[3] = cv::Point2d(gridSize.width, gridSize.height);
	int XShift = gridDim.x * gridSize.width;

	cv::Point cornerTl, cornerBr;
	for (int i = 0; i < gridDim.y; i++)
	{
		for (int j = 0; j < gridDim.x; j++)
		{
			for (size_t m = 0; m < 4; m++)
				vCornerPt[m] = vVertices[vCornerIdx[m]];

			GridMapping(vCornerPt, vGridPt, vertShift, src_, dst, srcMask_, mask, true);
			/*
			//Analysis the grid quadrilateral
			std::vector<cv::Vec3d> vCornerPt_(4);
			for (size_t m = 0; m < 4; m++)
			vCornerPt_[m] = cv::Vec3d(vCornerPt[m].x, vCornerPt[m].y, 1);

			//Get the cross point of diagonal line
			cv::Vec3d line03 = vCornerPt_[0].cross(vCornerPt_[3]);
			cv::Vec3d line12 = vCornerPt_[1].cross(vCornerPt_[2]);
			cv::Vec3d crossPt_ = line03.cross(line12);
			bool valid = std::abs(crossPt_.dot(crossPt_)) > 1e-16;

			if (valid)
			{
			cv::Point2d crossPt(crossPt_[0] / crossPt_[2], crossPt_[1] / crossPt_[2]);

			//Get the delata vector from grid corner to the cross point
			std::vector<cv::Vec2d> vDelta(4);
			for (size_t m = 0; m < 4; m++)
			vDelta[m] = cv::Vec2d(crossPt.x - vCornerPt[m].x, crossPt.y - vCornerPt[m].y);

			//decide whether the cross point on the diagonal lines
			double proof03 = vDelta[0].dot(vDelta[3]);
			double proof12 = vDelta[1].dot(vDelta[2]);

			if (proof03 < 0 && proof12 < 0)
			{
			cornerTl = vCornerPt[0];
			cornerBr = cornerTl;
			for (size_t m = 0; m < 4; m++)
			{

			cv::Point2d &pt = vCornerPt[m];
			if (cornerTl.x > pt.x)cornerTl.x = floor(pt.x);
			if (cornerTl.y > pt.y)cornerTl.y = floor(pt.y);
			if (cornerBr.x < pt.x)cornerBr.x = ceil(pt.x);
			if (cornerBr.y < pt.y)cornerBr.y = ceil(pt.y);
			}

			std::vector<uchar> inlierMask(4, 1);
			cv::Mat gridHinv = cv::findHomography(vCornerPt, vGridPt, inlierMask);
			//cv::Mat gridHinv = cv::getAffineTransform(&vCornerPt[0], &vGridPt[0]);

			cv::Rect srcGridROI(vGridPt[0], vGridPt[3]), dstGridROI(cornerTl, cornerBr),
			dstGridROI_(cornerTl + vertShift, cornerBr + vertShift);
			if (srcGridROI.x < 0 || srcGridROI.y < 0)
			{
			std::cout << i << " " << j << " " << srcGridROI << std::endl;
			}
			HomographMapping(gridHinv, srcGridROI, dstGridROI, src_(srcGridROI), dst(dstGridROI_), srcMask_(srcGridROI), mask(dstGridROI_));
			}
			else if (proof03 > 0 && proof12 > 0)
			{
			//The quadrangle is twisty
			valid = false;
			std::cout << "twisty situation" << std::endl;
			//continue;
			}
			else if (proof03 == 0 && proof12 == 0)
			{
			//Only choose one bigest triangle for warp
			int maxIdx1 = 0, maxIdx2 = 0, minIdx = 0;
			double max1 = 0, max2 = 0, min = vDelta[0].dot(vDelta[0]);
			for (int m = 0; m < 4; m++)
			{
			double distant2 = vDelta[m].dot(vDelta[m]);
			if (distant2 > max1)
			{
			max2 = max1;
			maxIdx2 = maxIdx1;
			max1 = distant2;
			maxIdx1 = m;
			}
			else if (distant2 > max2)
			{
			max2 = distant2;
			maxIdx2 = m;
			}

			if (distant2 < min)
			{
			min = distant2;
			minIdx = m;
			}
			}

			cv::Point2f vTrianglePt[3], vTriangleGridPt[3];
			vTrianglePt[0] = crossPt;
			vTrianglePt[1] = vCornerPt[maxIdx1];
			vTrianglePt[2] = vCornerPt[maxIdx2];

			vTriangleGridPt[0] = vGridPt[minIdx];
			vTriangleGridPt[1] = vGridPt[maxIdx1];
			vTriangleGridPt[2] = vGridPt[maxIdx2];

			cornerTl.x = floor(std::min(crossPt.x, std::min(vCornerPt[maxIdx1].x, vCornerPt[maxIdx2].x)));
			cornerTl.y = floor(std::min(crossPt.y, std::min(vCornerPt[maxIdx1].y, vCornerPt[maxIdx2].y)));
			cornerBr.x = ceil(std::max(crossPt.x, std::max(vCornerPt[maxIdx1].x, vCornerPt[maxIdx2].x)));
			cornerBr.y = ceil(std::max(crossPt.y, std::max(vCornerPt[maxIdx1].y, vCornerPt[maxIdx2].y)));

			cv::Mat warpH = cv::getAffineTransform(vTrianglePt, vTriangleGridPt);

			cv::Rect srcGridROI(vGridPt[0], vGridPt[3]), dstGridROI(cornerTl, cornerBr),
			dstGridROI_(cornerTl + vertShift, cornerBr + vertShift);
			HomographMapping(warpH, srcGridROI, dstGridROI, src_(srcGridROI), dst(dstGridROI_), srcMask_(srcGridROI), mask(dstGridROI_));
			}
			else
			{
			std::vector<int> vIdx1 = { 0, 3 }, vIdx2 = { 1, 2 };

			std::vector<int> &digIdx = proof03 < 0 ? vIdx2 : vIdx1;
			std::vector<int> &angIdx = proof03 < 0 ? vIdx1 : vIdx2;

			//std::cout << "proof03 = " << proof03 << "    proof12 = " << proof12 << std::endl;

			for (size_t m = 0; m < 2; m++)
			{
			std::vector<int> tIdx = { digIdx[0], digIdx[1], angIdx[m] };
			cv::Point2f vTrianglePt[3], vTriangleGridPt[3];
			cornerTl = vCornerPt[tIdx[0]];
			cornerBr = cornerTl;
			for (size_t n = 0; n < 3; n++)
			{
			int &idxTmp = tIdx[n];
			vTrianglePt[n] = vCornerPt[idxTmp];
			vTriangleGridPt[n] = vGridPt[idxTmp];

			cornerTl.x = std::min(cornerTl.x, int(floor(vTrianglePt[n].x)));
			cornerTl.y = std::min(cornerTl.y, int(floor(vTrianglePt[n].y)));
			cornerBr.x = std::max(cornerBr.x, int(ceil(vTrianglePt[n].x)));
			cornerBr.y = std::max(cornerBr.y, int(ceil(vTrianglePt[n].y)));
			}

			cv::Mat warpAffine = cv::getAffineTransform(vTrianglePt, vTriangleGridPt);
			cv::Mat warpH = cv::Mat::eye(3, 3, CV_64FC1);
			warpAffine.copyTo(warpH(cv::Rect(0, 0, 3, 2)));

			cv::Rect srcGridROI(vGridPt[0], vGridPt[3]), dstGridROI(cornerTl, cornerBr),
			dstGridROI_(cornerTl + vertShift, cornerBr + vertShift);
			HomographMapping(warpH, srcGridROI, dstGridROI, src_(srcGridROI), dst(dstGridROI_), srcMask_(srcGridROI), mask(dstGridROI_));
			}
			}
			}*/


			for (size_t m = 0; m < 4; m++)
			{
				vCornerIdx[m]++;
				vGridPt[m].x += gridSize.width;
			}
		}

		for (size_t m = 0; m < 4; m++)
		{
			vCornerIdx[m] += 1;
			vGridPt[m].y += gridSize.height;
			vGridPt[m].x -= XShift;
		}

	}

}

//The function for computing the map for ROI region by using the Homography matrix:
//dstMap : the mapping matrix from dst to src image
//srcMap : the mapping matrix from src to dst image
//other parameters are similar to GridMapping function
inline void BuildHomographMap(cv::Mat Hinv, cv::Rect srcROI, cv::Rect dstROI,
							  const cv::Mat &src, const cv::Mat &srcMask,
							  cv::Mat &dstMap, cv::Mat &srcMap, cv::Mat &dstMask,
							  bool IsComputeSrcMap = false)
{
	assert(src.rows == srcROI.height && src.cols == srcROI.width);
	assert(srcMap.rows == srcROI.height && srcMap.cols == srcROI.width);
	assert(dstMap.rows == dstROI.height && dstMap.cols == dstROI.width);
	assert(srcMask.rows == srcROI.height && srcMask.cols == srcROI.width);
	assert(dstMask.rows == dstROI.height && dstMask.cols == dstROI.width);
	assert(src.type() == CV_8UC3 && dstMap.type() == CV_32FC2 && srcMap.type() == CV_32FC2);
	assert(srcMask.type() == CV_8UC1 && dstMask.type() == CV_8UC1);
	cv::Point2d srcPt, dstPt;
	cv::Point2d srcGridPt, dstGridPt;
	cv::Size srcSize(src.size()), dstSize(dstMap.size());
	for (int i = 0; i < dstROI.height; i++)
	{
		cv::Vec2f *dstMapRow = reinterpret_cast<cv::Vec2f *>(dstMap.ptr(i));
		uchar *dstMaskRow = dstMask.ptr(i);
		for (int j = 0; j < dstROI.width; j++)
		{
			dstPt.x = dstROI.x + j;
			dstPt.y = dstROI.y + i;
			PointHTransform(dstPt, Hinv, srcPt);
			srcGridPt.x = round(srcPt.x - srcROI.x);
			srcGridPt.y = round(srcPt.y - srcROI.y);
			if (CheckInSize(srcGridPt, srcSize) && srcMask.at<uchar>(srcGridPt.y, srcGridPt.x))
			{
				dstMapRow[j] = cv::Vec2f(srcPt.x, srcPt.y);
				dstMaskRow[j] = 255;
			}
		}
	}

	if (IsComputeSrcMap)
	{
		cv::Mat H = Hinv.inv();
		for (int i = 0; i < srcROI.height; i++)
		{
			cv::Vec2f *srcMapRow = reinterpret_cast<cv::Vec2f *>(srcMap.ptr(i));
			for (int j = 0; j < srcROI.width; j++)
			{
				srcPt.x = srcROI.x + j;
				srcPt.y = srcROI.y + i;
				PointHTransform(srcPt, H, dstPt);
				dstGridPt.x = round(dstPt.x - dstROI.x);
				dstGridPt.y = round(dstPt.y - dstROI.y);
				if (CheckInSize(dstGridPt, dstSize) && dstMask.at<uchar>(dstGridPt.y, dstGridPt.x))
				{
					srcMapRow[j] = cv::Vec2f(dstPt.x, dstPt.y);
				}
			}
		}
	}

}

//The function for computing the map for the grid region:
//dstMap : the mapping matrix from dst to src image
//srcMap : the mapping matrix from src to dst image
//other parameters are similar to GridMapping function
inline bool BuildGridMap(const std::vector<cv::Point2d> &vCornerPt, const std::vector<cv::Point2d> &vGridPt,
						 cv::Point dstROIShift, const cv::Mat &src, const cv::Mat &srcMask,
						 cv::Mat &dstMap, cv::Mat &srcMap, cv::Mat &dstMask,
						 bool IsComputeSrcMap = false, bool processTwist = false)
{

	//Check the vector size
	if (vCornerPt.size() != 4 || vGridPt.size() != 4)
		HL_CERR("The input vCornerPt and vGridPt must only contains 4 elements");

	//Check the validity of vCornerPt and vGridPt
	cv::Size srcSize = src.size(), dstSize = dstMap.size();
	bool ptValid = true;
	for (size_t m = 0; m < 4; m++)
	{
		ptValid &= CheckInSize(vGridPt[m], srcSize);
		ptValid &= CheckInSize(vCornerPt[m] + cv::Point2d(dstROIShift), dstSize);
	}

	if (!ptValid)
		HL_CERR("The input point exceed the image size");

	//Analysis the grid quadrilateral
	std::vector<cv::Vec3d> vCornerPt_(4);
	for (size_t m = 0; m < 4; m++)
		vCornerPt_[m] = cv::Vec3d(vCornerPt[m].x, vCornerPt[m].y, 1);

	//Get the cross point of diagonal line
	cv::Vec3d line03 = vCornerPt_[0].cross(vCornerPt_[3]);
	cv::Vec3d line12 = vCornerPt_[1].cross(vCornerPt_[2]);
	cv::Vec3d crossPt_ = line03.cross(line12);
	if (std::abs(crossPt_.dot(crossPt_)) <= 1e-16)
	{
		//std::cout << "std::abs(crossPt_.dot(crossPt_)) <= 1e-16" << std::endl;
		return false;
	}

	cv::Point2d crossPt(crossPt_[0] / crossPt_[2], crossPt_[1] / crossPt_[2]);

	//Get the delata vector from grid corner to the cross point
	std::vector<cv::Vec2d> vDelta(4);
	for (size_t m = 0; m < 4; m++)
		vDelta[m] = cv::Vec2d(crossPt.x - vCornerPt[m].x, crossPt.y - vCornerPt[m].y);

	//decide whether the cross point on the diagonal lines
	double proof03 = vDelta[0].dot(vDelta[3]);
	double proof12 = vDelta[1].dot(vDelta[2]);
	cv::Point cornerTl, cornerBr;
	if (proof03 < 0 && proof12 < 0)
	{
		cornerTl = vCornerPt[0];
		cornerBr = cornerTl;
		for (size_t m = 0; m < 4; m++)
		{
			const cv::Point2d &pt = vCornerPt[m];
			if (cornerTl.x > pt.x)cornerTl.x = round(pt.x);
			if (cornerTl.y > pt.y)cornerTl.y = round(pt.y);
			if (cornerBr.x < pt.x)cornerBr.x = round(pt.x);
			if (cornerBr.y < pt.y)cornerBr.y = round(pt.y);
		}

		std::vector<uchar> inlierMask(4, 1);
		cv::Mat gridHinv = cv::findHomography(vCornerPt, vGridPt, inlierMask);

		cv::Rect srcGridROI(vGridPt[0], vGridPt[3]), dstGridROI(cornerTl, cornerBr),
			dstGridROI_(cornerTl + dstROIShift, cornerBr + dstROIShift);

		BuildHomographMap(gridHinv, srcGridROI, dstGridROI, src(srcGridROI), srcMask(srcGridROI), dstMap(dstGridROI_), srcMap(srcGridROI), dstMask(dstGridROI_), IsComputeSrcMap);
	}
	else if (proof03 > 0 && proof12 > 0)
	{
		//The quadrangle is twisty, then we change the order of vCornerPt
		//and mapping the new grid vCornerTwistyPt which just change the vCornerPt's 3 and 1 pos
		//std::cout << "twisty situation" << std::endl;
		if (processTwist)
		{

			bool find_mapping = false;
			for (size_t m = 1; m < 4; m++)
			{
				std::vector<cv::Point2d> vCornerTwistyPt = vCornerPt;
				vCornerTwistyPt[0] += vCornerTwistyPt[m];
				vCornerTwistyPt[m] = vCornerTwistyPt[0] - vCornerTwistyPt[m];
				vCornerTwistyPt[0] = vCornerTwistyPt[0] - vCornerTwistyPt[m];

				//TODO: Maybe there is a more reasonable strategy
				if ((find_mapping = BuildGridMap(vCornerTwistyPt, vGridPt, dstROIShift, src, srcMask, dstMap, srcMap, dstMask)))
					break;
			}
			return find_mapping;
		}
		else
			return false;
		//continue;
	}
	else if (proof03 == 0 && proof12 == 0)
	{
		//Only choose one bigest triangle for warp
		//std::cout << "Only choose one bigest triangle for warp" << std::endl;
		int maxIdx1 = 0, maxIdx2 = 0, minIdx = 0;
		double max1 = 0, max2 = 0, min = vDelta[0].dot(vDelta[0]);
		for (int m = 0; m < 4; m++)
		{
			double distant2 = vDelta[m].dot(vDelta[m]);
			if (distant2 > max1)
			{
				max2 = max1;
				maxIdx2 = maxIdx1;
				max1 = distant2;
				maxIdx1 = m;
			}
			else if (distant2 > max2)
			{
				max2 = distant2;
				maxIdx2 = m;
			}

			if (distant2 < min)
			{
				min = distant2;
				minIdx = m;
			}
		}

		cv::Point2f vTrianglePt[3], vTriangleGridPt[3];
		vTrianglePt[0] = crossPt;
		vTrianglePt[1] = vCornerPt[maxIdx1];
		vTrianglePt[2] = vCornerPt[maxIdx2];

		vTriangleGridPt[0] = vGridPt[minIdx];
		vTriangleGridPt[1] = vGridPt[maxIdx1];
		vTriangleGridPt[2] = vGridPt[maxIdx2];

		cornerTl.x = floor(std::min(crossPt.x, std::min(vCornerPt[maxIdx1].x, vCornerPt[maxIdx2].x)));
		cornerTl.y = floor(std::min(crossPt.y, std::min(vCornerPt[maxIdx1].y, vCornerPt[maxIdx2].y)));
		cornerBr.x = ceil(std::max(crossPt.x, std::max(vCornerPt[maxIdx1].x, vCornerPt[maxIdx2].x)));
		cornerBr.y = ceil(std::max(crossPt.y, std::max(vCornerPt[maxIdx1].y, vCornerPt[maxIdx2].y)));

		cv::Mat warpAffine = cv::getAffineTransform(vTrianglePt, vTriangleGridPt);
		cv::Mat warpH = cv::Mat::eye(3, 3, CV_64FC1);
		warpAffine.copyTo(warpH(cv::Rect(0, 0, 3, 2)));

		cv::Rect srcGridROI(vGridPt[0], vGridPt[3]), dstGridROI(cornerTl, cornerBr),
			dstGridROI_(cornerTl + dstROIShift, cornerBr + dstROIShift);
		BuildHomographMap(warpH, srcGridROI, dstGridROI, src(srcGridROI), srcMask(srcGridROI), dstMap(dstGridROI_), srcMap(srcGridROI), dstMask(dstGridROI_), IsComputeSrcMap);
	}
	else
	{
		std::vector<int> vIdx1 = { 0, 3 }, vIdx2 = { 1, 2 };

		std::vector<int> &digIdx = proof03 < 0 ? vIdx2 : vIdx1;
		std::vector<int> &angIdx = proof03 < 0 ? vIdx1 : vIdx2;

		//std::cout << "proof03 = " << proof03 << "    proof12 = " << proof12 << std::endl;

		for (size_t m = 0; m < 2; m++)
		{
			std::vector<int> tIdx = { digIdx[0], digIdx[1], angIdx[m] };
			cv::Point2f vTrianglePt[3], vTriangleGridPt[3];
			cornerTl = vCornerPt[tIdx[0]];
			cornerBr = cornerTl;
			for (size_t n = 0; n < 3; n++)
			{
				int &idxTmp = tIdx[n];
				vTrianglePt[n] = vCornerPt[idxTmp];
				vTriangleGridPt[n] = vGridPt[idxTmp];

				cornerTl.x = std::min(cornerTl.x, int(floor(vTrianglePt[n].x)));
				cornerTl.y = std::min(cornerTl.y, int(floor(vTrianglePt[n].y)));
				cornerBr.x = std::max(cornerBr.x, int(ceil(vTrianglePt[n].x)));
				cornerBr.y = std::max(cornerBr.y, int(ceil(vTrianglePt[n].y)));
			}

			cv::Mat warpAffine = cv::getAffineTransform(vTrianglePt, vTriangleGridPt);
			cv::Mat warpH = cv::Mat::eye(3, 3, CV_64FC1);
			warpAffine.copyTo(warpH(cv::Rect(0, 0, 3, 2)));

			cv::Rect srcGridROI(vGridPt[0], vGridPt[3]), dstGridROI(cornerTl, cornerBr),
				dstGridROI_(cornerTl + dstROIShift, cornerBr + dstROIShift);

			BuildHomographMap(warpH, srcGridROI, dstGridROI, src(srcGridROI), srcMask(srcGridROI), dstMap(dstGridROI_), srcMap(srcGridROI), dstMask(dstGridROI_), IsComputeSrcMap);
		}
	}
	return true;
}

//The function for computing the maps:
//dstMap : the mapping matrix from dst to src image
//srcMap : the mapping matrix from src to dst image
//other parameters are similar to GridWarping function
inline void BuildGridWarpMap(const cv::Mat& src, const cv::Mat& srcMask, const cv::Point& gridDim,
							 const cv::Size &gridSize, const std::vector<cv::Point2d> &vVertices,
							 cv::Mat &dstMap, cv::Mat &srcMap, cv::Mat &dstMask, cv::Rect &resultROI,
							 bool IsComputeSrcMap = false)
{
	assert(src.type() == CV_8UC3);
	assert(srcMask.type() == CV_8UC1 || srcMask.empty());
	assert(vVertices.size() == (gridDim.x + 1) * (gridDim.y + 1));

	cv::Size srcSize_(gridDim.x*gridSize.width + 1, gridDim.y*gridSize.height + 1);
	srcSize_.width = src.cols > srcSize_.width ? src.cols : srcSize_.width;
	srcSize_.height = src.rows > srcSize_.height ? src.rows : srcSize_.height;
	cv::Mat src_(srcSize_, CV_8UC3, cv::Scalar(0));
	cv::Mat srcMask_(srcSize_, CV_8UC1, cv::Scalar(0));
	cv::Rect srcROI(0, 0, src.cols, src.rows);
	src.copyTo(src_(srcROI));
	if (!srcMask.empty() && srcMask.size() == src.size())
		srcMask.copyTo(srcMask_(srcROI));
	else
		cv::rectangle(srcMask_, srcROI, cv::Scalar(255), -1);

	cv::Point gridTl(vVertices[0]), gridBr = gridTl;
	for (int i = 0; i < vVertices.size(); i++)
	{
		const cv::Point2d &pt = vVertices[i];
		if (gridTl.x > pt.x)gridTl.x = floor(pt.x);
		if (gridTl.y > pt.y)gridTl.y = floor(pt.y);
		if (gridBr.x < pt.x)gridBr.x = ceil(pt.x);
		if (gridBr.y < pt.y)gridBr.y = ceil(pt.y);
	}

	resultROI = cv::Rect(gridTl, gridBr);
	cv::Point vertShift = -resultROI.tl();

	//To avoid the size check;
	resultROI.width++;
	resultROI.height++;

	dstMap = cv::Mat(resultROI.height, resultROI.width, CV_32FC2, cv::Scalar(-1.0, -1.0));
	srcMap = cv::Mat(srcSize_.height, srcSize_.width, CV_32FC2, cv::Scalar(-1.0, -1.0));
	dstMask = cv::Mat(resultROI.height, resultROI.width, CV_8UC1, cv::Scalar(0));

	std::vector<int> vCornerIdx(4, 0);
	vCornerIdx[1] = 1;
	vCornerIdx[2] = gridDim.x + 1;
	vCornerIdx[3] = vCornerIdx[2] + 1;
	std::vector<cv::Point2d> vCornerPt(4), vGridPt(4, cv::Point2d(0, 0));
	vGridPt[1] = cv::Point2d(gridSize.width, 0);
	vGridPt[2] = cv::Point2d(0, gridSize.height);
	vGridPt[3] = cv::Point2d(gridSize.width, gridSize.height);
	int XShift = gridDim.x * gridSize.width;

	cv::Point cornerTl, cornerBr;
	for (int i = 0; i < gridDim.y; i++)
	{
		for (int j = 0; j < gridDim.x; j++)
		{
			for (size_t m = 0; m < 4; m++)
				vCornerPt[m] = vVertices[vCornerIdx[m]];

			BuildGridMap(vCornerPt, vGridPt, vertShift, src_, srcMask_, dstMap, srcMap, dstMask, IsComputeSrcMap, true);

			for (size_t m = 0; m < 4; m++)
			{
				vCornerIdx[m]++;
				vGridPt[m].x += gridSize.width;
			}
		}

		for (size_t m = 0; m < 4; m++)
		{
			vCornerIdx[m] += 1;
			vGridPt[m].y += gridSize.height;
			vGridPt[m].x -= XShift;
		}

	}

	srcMap = srcMap(srcROI);
}

//simplest average merging function
//Return the result ROI;
inline cv::Rect AverageMerge(const std::vector<cv::Mat> &vImage, const std::vector<cv::Mat> &vMask,
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

	cv::threshold(resultMask, resultMask, 0, 255, cv::THRESH_BINARY);
	return resultROI;
}

#define BUILDPROBLEMDEBUG 1
//Get the vertice warp problem's sparse matrix elements vTriplet and b
//vPresetVert is the preset of vertices , it is always set according the global Homography Matrix
//gamma : control the weight of point-correspond constraint
//alpha : control the weight of preset vertices constraint
//beta  : control the weight of shape constraint which is inroduced from content-preserving warp
//forceAlpha : force the preset vertices constraint which will not be ignored by pair points
inline void buildGridWarpProblem(const PairInfo &pair, const std::vector<cv::Point2d> &vPresetVert,
								 const std::vector<cv::Mat> &images, cv::Point gridDim, cv::Size gridSize,
								 std::vector<Eigen::Triplet<double>> &vTriplet, Eigen::VectorXd &b,
								 double gamma = 1, double alpha = 1e-4, double beta = 1e-5, bool forceAlpha = false)
{
	int verticeNum = (gridDim.x + 1) * (gridDim.y + 1), paramNum = verticeNum * 2;
	b.resize(paramNum);
	b.setZero();

	std::vector<std::vector<int>> cellFCount(gridDim.y, std::vector<int>(gridDim.x, 0));
	std::vector<std::vector<double>> denseMatrix(paramNum, std::vector<double>(paramNum, 0));
	std::vector<double> nSumFCount(verticeNum, 0);
	//add local alignment term
	assert(gamma >= 0.0);
	if (gamma > 0.0)
	{
		for (size_t k = 0; k < pair.pairs_num; k++)
		{
			if (!pair.mask[k]) continue;

			cv::Point2f pt1 = pair.points1[k], pt2 = pair.points2[k];
			int c = pt1.x / gridSize.width, r = pt1.y / gridSize.height;
			++cellFCount[r][c];

			int vertIdxs[4];
			vertIdxs[0] = r * (gridDim.x + 1) + c, vertIdxs[1] = vertIdxs[0] + 1;
			vertIdxs[2] = vertIdxs[0] + (gridDim.x + 1), vertIdxs[3] = vertIdxs[2] + 1;

			double coeffs[4];
			cv::Point2f localCoord(pt1.x - c * gridSize.width, pt1.y - r * gridSize.height);
			double a1 = 1 - (localCoord.x / gridSize.width), a2 = 1 - (localCoord.y / gridSize.height);
			coeffs[0] = a1 * a2;
			coeffs[1] = (1 - a1) * a2;
			coeffs[2] = a1 * (1 - a2);
			coeffs[3] = (1 - a1)*(1 - a2);

			for (int i = 0; i < 4; i++)
			{
				int rIdx = vertIdxs[i] * 2;
				++nSumFCount[vertIdxs[i]];
				for (int j = 0; j < 4; j++)
				{
					int cIdx = vertIdxs[j] * 2;
					double tmpCoeff = coeffs[i] * coeffs[j] * gamma;
					denseMatrix[rIdx][cIdx] += tmpCoeff;
					denseMatrix[rIdx + 1][cIdx + 1] += tmpCoeff;
				}

				b[rIdx] += coeffs[i] * pt2.x * gamma;
				b[rIdx + 1] += coeffs[i] * pt2.y * gamma;
			}
		}
	}

	//add global alignment term
	assert(alpha >= 0.0);
	if (alpha > 0.0)
	{
		for (int i = 0, rIdx = 0; i < verticeNum; i++, rIdx += 2)
		{
			if (forceAlpha || !nSumFCount[i])
			{
				denseMatrix[rIdx][rIdx] += alpha;
				denseMatrix[rIdx + 1][rIdx + 1] += alpha;
				b[rIdx] += alpha * vPresetVert[i].x;
				b[rIdx + 1] += alpha * vPresetVert[i].y;
			}
		}
	}


	//calculate the variance of triangles
	//add smoothness term
	assert(beta >= 0.0);
	if (beta > 0.0)
	{
		cv::Mat squareMask(gridSize, CV_8UC1, cv::Scalar(0));
		cv::line(squareMask, cv::Point(0, 0), cv::Point(squareMask.cols - 1, squareMask.rows - 1), cv::Scalar(1));
		cv::line(squareMask, cv::Point(squareMask.cols - 1, 0), cv::Point(0, squareMask.rows - 1), cv::Scalar(2));

		const cv::Mat &image1 = images[pair.index1];
		cv::Rect image1Roi(0, 0, image1.cols, image1.rows);

		cv::Size extSize(gridSize.width * gridDim.x, gridSize.height * gridDim.y);
		extSize.width = image1Roi.width > extSize.width ? image1Roi.width : extSize.width;
		extSize.height = image1Roi.height > extSize.height ? image1Roi.height : extSize.height;

		cv::Mat extMask(extSize, CV_8UC1, cv::Scalar(0));
		cv::Mat extImg(extSize, CV_8UC3, cv::Scalar(0));

		cv::rectangle(extMask, image1Roi, cv::Scalar(255), -1);
		image1.copyTo(extImg(image1Roi));

		//Notice that the order of quadrangle in here is 0 1 2 3 followed clockwise.
		//But in other place , the clockwise index order is 0 1 3 2
		const int vTriangles[8][3] = {
			{ 0, 2, 3 },
			{ 2, 3, 0 },
			{ 0, 1, 2 },
			{ 2, 0, 1 },
			{ 1, 3, 0 },
			{ 3, 0, 1 },
			{ 1, 2, 3 },
			{ 3, 1, 2 }
		};

		for (size_t i = 0; i < gridDim.y; i++)
		{
			for (size_t j = 0; j < gridDim.x; j++)
			{
				cv::Rect cellRoi(j * gridSize.width, i * gridSize.height, gridSize.width, gridSize.height);
				std::vector<cv::Vec3d> vSumColor(4, cv::Vec3d(0, 0, 0)), vAvgColor(4);
				std::vector<int> vCount(4, 1);
				cv::Mat cellMat = extImg(cellRoi), cellMask = extMask(cellRoi);

				//calculate the sum of different triangle regions
				for (size_t m = 0; m < gridSize.height; m++)
				{
					cv::Vec3b *pRowImg = reinterpret_cast<cv::Vec3b *>(cellMat.ptr(m));
					uchar *pRowMask = reinterpret_cast<uchar *>(cellMask.ptr(m));
					uchar *pRowSquareMask = reinterpret_cast<uchar *>(squareMask.ptr(m));
					bool across1 = false, across2 = false;
					for (size_t n = 0; n < gridSize.width; n++)
					{
						if (pRowSquareMask[n] == 1)across1 = true;
						if (pRowSquareMask[n] == 2)across2 = true;
						if (pRowMask[n])
						{
							int triangleIdx01 = across1 ? 1 : 0;
							int triangleIdx23 = across2 ? 3 : 2;
							vSumColor[triangleIdx01] += pRowImg[n];
							vSumColor[triangleIdx23] += pRowImg[n];
							vCount[triangleIdx01] += 1;
							vCount[triangleIdx23] += 1;
						}
					}
				}

				//calculate the average color vectors
				for (size_t m = 0; m < 4; m++)
					vAvgColor[m] = vSumColor[m] / vCount[m];

				//calculate the covariance matrix and its L2 norm
				std::vector<cv::Mat> vCovMat(4, cv::Mat(3, 3, CV_64FC1, cv::Scalar(0)));
				std::vector<double> vCovL2Value(4, 0);
				for (size_t m = 0; m < gridSize.height; m++)
				{
					cv::Vec3b *pRowImg = reinterpret_cast<cv::Vec3b *>(cellMat.ptr(m));
					uchar *pRowMask = reinterpret_cast<uchar *>(cellMask.ptr(m));
					uchar *pRowSquareMask = reinterpret_cast<uchar *>(squareMask.ptr(m));
					bool across1 = false, across2 = false;
					for (size_t n = 0; n < gridSize.width; n++)
					{
						if (pRowSquareMask[n] == 1)across1 = true;
						if (pRowSquareMask[n] == 2)across2 = true;
						if (pRowMask[n])
						{
							int triangleIdx01 = across1 ? 1 : 0;
							int triangleIdx23 = across2 ? 3 : 2;
							cv::Vec3d residual01_ = cv::Vec3d(pRowImg[n]) - vAvgColor[triangleIdx01];
							cv::Vec3d residual23_ = cv::Vec3d(pRowImg[n]) - vAvgColor[triangleIdx23];
							cv::Mat residual01(residual01_), residual23(residual23_);
							vCovMat[triangleIdx01] += (residual01 * residual01.t());
							vCovMat[triangleIdx23] += (residual23 * residual23.t());
						}
					}
				}

				for (size_t m = 0; m < 4; m++)
					vCovL2Value[m] = cv::norm(vCovMat[m] / vCount[m]);

				std::vector<int> vVertIdx(4);
				vVertIdx[0] = i * (gridDim.x + 1) + j; vVertIdx[1] = vVertIdx[0] + 1;
				vVertIdx[3] = vVertIdx[0] + (gridDim.x + 1); vVertIdx[2] = vVertIdx[3] + 1;

				for (int m = 0; m < 8; m++)
				{
					int idx1 = vVertIdx[vTriangles[m][0]], idx2 = vVertIdx[vTriangles[m][1]], idx3 = vVertIdx[vTriangles[m][2]];
					const cv::Point2f &pt1 = vPresetVert[idx1], &pt2 = vPresetVert[idx2], &pt3 = vPresetVert[idx3];
					cv::Point2f pt21 = pt1 - pt2;
					cv::Point2f pt23 = pt3 - pt2;
					double u = pt21.dot(pt23) / pt23.dot(pt23);

					cv::Point2f R90pt23(pt23.y, -pt23.x);
					double v = pt21.dot(R90pt23) / R90pt23.dot(R90pt23);

					double ws = std::max(vCovL2Value[m / 2], 100.0);

					std::vector<int> vParamIdx(5, 0);
					std::vector<double>  vCoeff(5, 0);
					vParamIdx[0] = idx1 * 2; vCoeff[0] = 1;
					vParamIdx[1] = idx2 * 2; vCoeff[1] = u - 1;
					vParamIdx[2] = idx3 * 2; vCoeff[2] = -u;
					vParamIdx[3] = vParamIdx[2] + 1; vCoeff[3] = -v;
					vParamIdx[4] = vParamIdx[1] + 1; vCoeff[4] = v;

					for (int n = 0; n < 5; n++)
					{
						int &rIdx = vParamIdx[n];
						for (int l = 0; l < 5; l++)
						{
							int &cIdx = vParamIdx[l];
							denseMatrix[rIdx][cIdx] += (beta * ws * vCoeff[n] * vCoeff[l]);
						}
					}

					vParamIdx[3] = vParamIdx[1];
					vParamIdx[4] = vParamIdx[2];
					vParamIdx[0] += 1;
					vParamIdx[1] += 1;
					vParamIdx[2] += 1;

					for (int n = 0; n < 5; n++)
					{
						int &rIdx = vParamIdx[n];
						for (int l = 0; l < 5; l++)
						{
							int &cIdx = vParamIdx[l];
							denseMatrix[rIdx][cIdx] += (beta * ws * vCoeff[n] * vCoeff[l]);
						}
					}

				}

			}
		}
	}

	if (!vTriplet.empty())vTriplet.clear();
	for (size_t i = 0; i < paramNum; i++)
		for (size_t j = 0; j < paramNum; j++)
			if (denseMatrix[i][j])
				vTriplet.push_back(Eigen::Triplet<double>(i, j, denseMatrix[i][j]));

#if defined(BUILDPROBLEMDEBUG) && (defined(_DEBUG) || defined(DEBUG))
	cv::Mat AImage(paramNum, paramNum, CV_32FC1, cv::Scalar(0));
	for (size_t i = 0; i < vTriplet.size(); i++)
	{
		AImage.at<float>(vTriplet[i].row(), vTriplet[i].col()) = vTriplet[i].value();
	}

	cv::Mat bImage(paramNum, 1, CV_32FC1, cv::Scalar(0));
	for (size_t i = 0; i < paramNum; i++)
	{
		bImage.at<float>(i, 0) = b[i];
	}
#endif // BUILDPROBLEMDEBUG

}


//Estimate the warped vertices of the grid
//gamma : control the weight of point-correspond constraint
//alpha : control the weight of preset vertices constraint
//beta  : control the weight of shape constraint which is inroduced from content-preserving warp
//forceAlpha : force the preset vertices constraint which will not be ignored by pair points
inline void EstimateGridVertices(const PairInfo &pair, const cv::Mat &presetH, cv::Point gridDim,
								 cv::Size gridSize, const std::vector<cv::Mat> &images,
								 std::vector<cv::Point2d> &vVertices, double gamma = 1, double alpha = 1e-4,
								 double beta = 1e-5, bool forceAlpha = false)
{
	int verticeNum = (gridDim.x + 1) * (gridDim.y + 1), paramNum = verticeNum * 2;
	std::vector<Eigen::Triplet<double>> vTriplet;
	Eigen::VectorXd b(paramNum), x(paramNum);
	 
	std::vector<cv::Point2d> vPresetVert(verticeNum);
	for (size_t i = 0; i < verticeNum; i++)
	{
		int r = i / (gridDim.x + 1), c = i - r * (gridDim.x + 1);
		cv::Point2d vertPt(c * gridSize.width, r * gridSize.height);
		if (!PointHTransform(vertPt, presetH, vPresetVert[i]))
			HL_CERR("Failed to Transform point by Homograph Matrix");
	}

	buildGridWarpProblem(pair, vPresetVert, images, gridDim, gridSize, vTriplet, b, gamma, alpha, beta, forceAlpha);


	Eigen::SparseMatrix<double> A(paramNum, paramNum);
	A.setFromTriplets(vTriplet.begin(), vTriplet.end());

	std::cout << "Matrix = (" << paramNum << "*" << paramNum << ")" << std::endl;

	HL_INTERVAL_START;


	Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
	solver.compute(A);
	if (solver.info() != Eigen::Success)
		HL_CERR("Failed to compute sparse Matrix");

	x = solver.solve(b);
	if (solver.info() != Eigen::Success)
		HL_CERR("Failed to solve the result");

	HL_INTERVAL_ENDSTR("Solve LLT")

	vVertices.resize(verticeNum);
	for (int i = 0, pIdx = 0; i < verticeNum; i++, pIdx += 2)
	{
		vVertices[i].x = x[pIdx];
		vVertices[i].y = x[pIdx + 1];
		if (isnan(x[pIdx]) || isnan(x[pIdx + 1]))
			HL_CERR("x " << pIdx << " is not valid");
	}

}


//Estimate the warped vertices of the grid with Preset Points and fixation mask
//vPresetVert £º the preset position of grid poings
//vFixMask : the mask indicate which points will be fixed to the preset position during the Estimation,
//           'True' means that the corresponding point is fixed.
//gamma : control the weight of point-correspond constraint
//alpha : control the weight of preset vertices constraint
//beta  : control the weight of shape constraint which is inroduced from content-preserving warp
//forceAlpha : force the preset vertices constraint which will not be ignored by pair points
inline void EstimateGridVertices(const PairInfo &pair, const std::vector<cv::Point2d> &vPresetVert,
								 const std::vector<bool> &vFixMask, cv::Point gridDim,
								 cv::Size gridSize, const std::vector<cv::Mat> &images,
								 std::vector<cv::Point2d> &vVertices, double gamma = 1, double alpha = 1e-4,
								 double beta = 1e-5, bool forceAlpha = false)
{
	int verticeNum = (gridDim.x + 1) * (gridDim.y + 1), paramNum = verticeNum * 2;

	if (verticeNum != vFixMask.size() || verticeNum != vPresetVert.size())
		HL_CERR("The verticeNum(" << verticeNum << ") is not equal to the size of vFixMask("
				<< vFixMask.size() << ") or vPresetVert(" << vPresetVert.size() << ")");

	std::vector<Eigen::Triplet<double>> vTriplet;
	Eigen::VectorXd b(paramNum);
	buildGridWarpProblem(pair, vPresetVert, images, gridDim, gridSize, vTriplet, b, gamma, alpha, beta, forceAlpha);

	int freePtNum = 0;
	std::for_each(vFixMask.begin(), vFixMask.end(), [&](const bool &isFixed) {freePtNum += (isFixed ? 0 : 1); });

	int freeParamNum = freePtNum * 2;
	Eigen::VectorXd freeB(freeParamNum), freeX(freeParamNum);
	std::vector<int> vAuxIndex(paramNum);
	for (size_t i = 0, pIdx = 0, fIdx = 0; i < verticeNum; i++, pIdx += 2)
	{
		if (vFixMask[i])
		{
			vAuxIndex[pIdx] = -1;
			vAuxIndex[pIdx + 1] = -1;
		}
		else
		{
			vAuxIndex[pIdx] = fIdx;
			vAuxIndex[pIdx + 1] = fIdx + 1;

			freeB[fIdx] = b[pIdx];
			freeB[fIdx + 1] = b[pIdx + 1];
			fIdx += 2;
		}
	}

	Eigen::SparseMatrix<double> freeA(freeParamNum, freeParamNum);

	std::vector<Eigen::Triplet<double>> vFreeTriplet;
	std::for_each(vTriplet.begin(), vTriplet.end(), [&](Eigen::Triplet<double> &triplet) {
		int r = triplet.row(), c = triplet.col();
		double v = triplet.value();

		int ptR = r / 2, ptC = c / 2;

		if (vFixMask[ptR] || vFixMask[ptC])
		{
			if (!vFixMask[ptR])
			{
				int fIdx = vAuxIndex[r];
				assert(fIdx != -1);
				double presetV = c % 2 == 0 ? vPresetVert[ptC].x : vPresetVert[ptC].y;
				freeB[fIdx] -= (v*presetV);
			}
		}
		else
		{
			Eigen::Triplet<double> triplet_ = triplet;
			int fIdxR = vAuxIndex[r], fIdxC = vAuxIndex[c];
			if (fIdxC == -1 || fIdxR == -1)
				HL_CERR("Error : Invalid fidx row(" << fIdxR << ") and col(" << fIdxC << ")");

			vFreeTriplet.push_back(Eigen::Triplet<double>(fIdxR, fIdxC, v));
		}

	});

	freeA.setFromTriplets(vFreeTriplet.begin(), vFreeTriplet.end());

	Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
	solver.compute(freeA);
	if (solver.info() != Eigen::Success)
		HL_CERR("Failed to compute sparse Matrix");

	freeX = solver.solve(freeB);
	if (solver.info() != Eigen::Success)
		HL_CERR("Failed to solve the result");

	vVertices.resize(verticeNum);
	for (int i = 0, pIdx = 0; i < verticeNum; i++)
	{
		if (!vFixMask[i])
		{
			vVertices[i].x = freeX[pIdx];
			vVertices[i].y = freeX[pIdx + 1];
			if (isnan(freeX[pIdx]) || isnan(freeX[pIdx + 1]))
				HL_CERR("x " << pIdx << " is not valid");
			pIdx += 2;
		}
		else
			vVertices[i] = vPresetVert[i];
	}

}


//Stitching the two images use the global Homography Matrix
inline void GlobalHStitching(const cv::Mat &src1, const cv::Mat &mask1, const cv::Mat &src2, const cv::Mat &mask2,
							 const cv::Mat &H, cv::Mat &dst, cv::Mat &maskDst)
{
	cv::Point gridDim(1, 1);
	cv::Size gridSize(src1.cols, src1.rows);
	int verticeNum = (gridDim.x + 1) * (gridDim.y + 1);
	std::vector<cv::Point2d> vVertices(verticeNum);

	for (size_t i = 0; i < verticeNum; i++)
	{
		int r = i / (gridDim.x + 1), c = i - r * (gridDim.x + 1);
		cv::Point2d vertPt(c * gridSize.width, r * gridSize.height);
		if (!PointHTransform(vertPt, H, vVertices[i]))
			HL_CERR("Failed to Transform point by Homograph Matrix");
	}

	/*cv::Mat gridResult = src2.clone() * 0.6;
	DrawGridVertices(gridResult, vVertices, gridDim, 2, 3);
	cv::imwrite("gridResultGlobal.jpg", gridResult);*/

	cv::Mat warpedResult, warpedMask;
	cv::Rect warpedROI;
	GridWarping(src1, mask1, gridDim, gridSize, vVertices, warpedResult, warpedMask, warpedROI);
	std::vector<cv::Mat> vPreparedImg, vPreparedMask;
	std::vector<cv::Rect> vROI;

	vPreparedImg.push_back(src2);
	vPreparedMask.push_back(mask2);
	vROI.push_back(cv::Rect(0, 0, src2.cols, src2.rows));

	vPreparedImg.push_back(warpedResult);
	vPreparedMask.push_back(warpedMask);
	vROI.push_back(warpedROI);

	AverageMerge(vPreparedImg, vPreparedMask, vROI, dst, maskDst);
}
