#define MAIN_FILE
#include <commonMacro.h>

#include <OpencvCommon.h>
#include "../common/SequenceMatcher.h"
#include "../common/stitchingCommonFuc.h"

void BlockStitching(const cv::Mat &src, cv::Mat &dst, cv::Rect srcRoi, cv::Rect dstRoi, 
					cv::Mat &dstMask, const cv::Mat &H)
{
	//decide the size of final image
	std::vector<cv::Point2f> corners(4);
	corners[0] = srcRoi.tl();
	corners[1] = srcRoi.br();
	corners[2] = cv::Point2f(corners[0].x, corners[1].y);
	corners[3] = cv::Point2f(corners[1].x, corners[0].y);

	cv::Point2f tl(H_FLOAT_MAX, H_FLOAT_MAX);
	cv::Point2f br(H_FLOAT_MIN, H_FLOAT_MIN);
	for (auto &corner : corners)
	{
		cv::Point2f tcorner;
		if (!PointHTransform(corner, H, tcorner))
		{
			HL_CERR("The Homography Point Translate is illegal");
		}
		if (tcorner.x < tl.x) { tl.x = tcorner.x; }
		if (tcorner.y < tl.y) { tl.y = tcorner.y; }
		if (tcorner.x > br.x) { br.x = tcorner.x; }
		if (tcorner.y > br.y) { br.y = tcorner.y; }
	}
	/*tl = cv::Point2f(0, 0);
	br = cv::Point2f(dst.cols, dst.rows);*/
	/*tl = cv::Point2f(-dst.cols, -dst.rows);
	br = cv::Point2f(dst.cols * 2, dst.rows * 2);*/

	cv::Rect srcRoi_(tl, br);
	bool isOverlap = GetOverlapRoi(srcRoi_, dstRoi, srcRoi_);
	if (!isOverlap) { return; }
	cv::Mat Hinv = H.inv();
	cv::Point tl_ = srcRoi_.tl();
	cv::Point br_ = srcRoi_.br();

	for (int yi = tl_.y; yi < br_.y; yi++)
	{
		cv::Vec3b *rowptr = reinterpret_cast<cv::Vec3b *>(dst.ptr(yi - dstRoi.y));
		uchar *rowMaskptr = dstMask.ptr(yi - dstRoi.y);
		for (int xi = tl_.x; xi < br_.x; xi++)
		{
			cv::Point2f curPt(xi, yi), srcPt;
			PointHTransform(curPt, Hinv, srcPt);
			cv::Point srcPt_;
			srcPt_.x = srcPt.x - srcRoi.x;
			srcPt_.y = srcPt.y - srcRoi.y;
			if (srcPt_.x >= 0 && srcPt_.x < src.cols && srcPt_.y >= 0 && srcPt_.y < src.rows)
			{
				cv::Vec3b srcPixel = src.at<cv::Vec3b>(srcPt_.y, srcPt_.x);
				int tempx = xi - dstRoi.x;
				switch (rowMaskptr[tempx])
				{
				case 255:
					rowptr[tempx] = 0.5*rowptr[tempx] + 0.5*srcPixel;
					rowMaskptr[tempx] = 254;
					break;
				case 254:
					break;
				default:
					rowptr[tempx] = srcPixel;
					break;
				}
			}
		}
	}
}

//using the nearest cluster model
void NNMovingDLTStitching(const cv::Mat &src1, const cv::Mat &src2, PairInfo &pairinfo, cv::Mat &dst,
						   cv::Point cellGrid, cv::Size cellSize)
{
	//get the origin A matrix
	std::vector<cv::Point2f> srcPts1(pairinfo.inliers_num), srcPts2(pairinfo.inliers_num);
	for (size_t i = 0, j = 0; i < pairinfo.pairs_num; i++)
	{
		if (pairinfo.mask[i] == 0) { continue; }
		srcPts1[j] = pairinfo.points1[i];
		srcPts2[j] = pairinfo.points2[i];
		j++;
	}

	std::vector<cv::Point2f> pts1, pts2;
	cv::Mat T1inv, T2inv;
	GetRegularizedPoints(srcPts1, pts1, T1inv);
	GetRegularizedPoints(srcPts2, pts2, T2inv);

	//T1 and T2inv used for compute origin H = T2inv*H'*T1
	cv::Mat T1 = T1inv.inv();

	cv::Mat A(pairinfo.inliers_num * 2, 9, CV_64F, cv::Scalar(0));
	double *Aptr = reinterpret_cast<double *>(A.data);
	for (size_t i = 0; i < pairinfo.inliers_num; i++, Aptr += 18)
	{
		//H'*pt1 = pt2
		cv::Point2f pt1 = pts1[i];
		cv::Point2f pt2 = pts2[i];
							 
		Aptr[3] = -pt1.x;
		Aptr[4] = -pt1.y;
		Aptr[5] = -1;
		Aptr[6] = pt2.y * pt1.x;
		Aptr[7] = pt2.y * pt1.y;
		Aptr[8] = pt2.y;

		Aptr[9] = pt1.x;
		Aptr[10] = pt1.y;
		Aptr[11] = 1;
		Aptr[15] = -pt2.x * pt1.x;
		Aptr[16] = -pt2.x * pt1.y;
		Aptr[17] = -pt2.x;
	}

	cv::Mat W, U, VT;
	cv::SVD::compute(A, W, U, VT);
	cv::Mat h = VT.row(8);
	cv::Mat globalH = h.reshape(1, 3);
	globalH = T2inv*globalH*T1;
	//GlobalHStitching(src1, src2, globalH, dst);

	//cv::Size cellSize(50, 50);
	int width = src1.cols, height = src1.rows;
	int cellCol = cellGrid.x;
	int cellRow = cellGrid.y;

	double gamma = 0.0025;
	//100 pixel region
	double radius2 = std::max(width, height);
	radius2 *= 0.25;
	radius2 *= radius2;
	double sigma2 = -radius2 / log(gamma);

	std::vector<cv::Mat> HArray(cellCol*cellRow);


	std::cout << "-------------Start to Moving DLT--------------" << std::endl;
	
	for (size_t i = 0, yIdx = cellSize.height / 2, Hidx = 0; i < cellRow; i++, yIdx += cellSize.height)
	{

		for (size_t j = 0, xIdx = cellSize.width /2; j < cellCol; j++, xIdx += cellSize.width, Hidx++)
		{
			int changeCount = 0;
			cv::Mat Atemp = A.clone();

			//change the center to the nearest point
			double minDistance2 = H_DOUBLE_MAX;
			int nearestIdx = 0;
			for (size_t k = 0; k < pairinfo.inliers_num; k++)
			{
				float dx = srcPts1[k].x - xIdx;
				float dy = srcPts1[k].y - yIdx;
				double distant2 = dx*dx + dy*dy;
				if (distant2 < minDistance2)
				{
					minDistance2 = distant2;
					nearestIdx = k;
				}
			}
			cv::Point2f nearestPt = srcPts1[nearestIdx];
			//

			for (size_t k = 0; k < pairinfo.inliers_num; k++)
			{
				float dx = srcPts1[k].x - nearestPt.x;
				float dy = srcPts1[k].y - nearestPt.y;
				double distant2 = dx*dx + dy*dy;
				if (distant2 <= radius2)
				{
					double wi = exp(-distant2 / sigma2);
					wi /= gamma;
					cv::Mat &rowTwo = Atemp.rowRange(k * 2, k * 2 + 2);
					rowTwo *= wi;
					changeCount++;
				}
			}
			if (changeCount > 0)
			{
				cv::Mat W_, U_, VT_;
				cv::SVD::compute(Atemp, W_, U_, VT_);
				cv::Mat h_ = VT_.row(8);
				cv::Mat H_ = h_.reshape(1, 3);
				HArray[Hidx] = T2inv*H_*T1;
			}
			else
			{
				HArray[Hidx] = globalH;
			}
		}
	}

	std::cout << "--------------End Moving DLT---------------" << std::endl;

	//decide the size of final image
	std::vector<cv::Point2f> corners(4);
	corners[0] = cv::Point2f(0, 0);
	corners[1] = cv::Point2f(src1.cols, 0);
	corners[2] = cv::Point2f(0, src1.rows);
	corners[3] = cv::Point2f(src1.cols, src1.rows);

	cv::Point2f tl(H_FLOAT_MAX, H_FLOAT_MAX);
	cv::Point2f br(H_FLOAT_MIN, H_FLOAT_MIN);
	for (auto &corner : corners)
	{
		cv::Point2f tcorner;
		int colIdx = corner.x / cellSize.width;
		int rowIdx = corner.y / cellSize.height;
		if (colIdx == cellCol)colIdx--;
		if (rowIdx == cellRow) rowIdx--;
		PointHTransform(corner, HArray[colIdx + rowIdx*cellCol], tcorner);
		if (tcorner.x < tl.x) { tl.x = tcorner.x; }
		if (tcorner.y < tl.y) { tl.y = tcorner.y; }
		if (tcorner.x > br.x) { br.x = tcorner.x; }
		if (tcorner.y > br.y) { br.y = tcorner.y; }
	}

	cv::Rect roi2(0, 0, src2.cols, src2.rows), roi1(tl, br);
	cv::Rect dstRoi = GetUnionRoi(roi1, roi2);

	roi2.x -= dstRoi.x;
	roi2.y -= dstRoi.y;

	dst = cv::Mat(dstRoi.size(), CV_8UC3);
	src2.copyTo(dst(roi2));

	cv::Mat dstMask(dstRoi.size(), CV_8U, cv::Scalar(0));
	cv::Mat src2Mask(src2.size(), CV_8U, cv::Scalar(255));
	src2Mask.copyTo(dstMask(roi2));

	//int edgeExtend = 0.5 * std::min(cellSize.height, cellSize.width);
	int edgeExtend = 0;
	for (size_t i = 0, yIdx = 0, Hidx = 0; i < cellRow; i++, yIdx += cellSize.height)
	{
		for (size_t j = 0, xIdx = 0; j < cellCol; j++, xIdx += cellSize.width, Hidx++)
		{
			//not transform the cut edge
			int cellx = xIdx, celly = yIdx, cellw = cellSize.width, cellh = cellSize.height;
			//if (j != 0)cellx -= edgeExtend;
			if (j != cellCol - 1)cellw += (1 * edgeExtend);
			//if (i != 0)celly -= edgeExtend;
			if (i != cellRow - 1) cellh += (1 * edgeExtend);
			cv::Rect cellRoi(cellx, celly, cellw, cellh);
			cv::Mat cellImg = src1(cellRoi);
			BlockStitching(cellImg, dst, cellRoi, dstRoi, dstMask, HArray[Hidx]);
		}
	}

	//for (size_t i = 0, yBound = cellSize.height, HRowIdx = 0, Hidx = 0; i < height; i++)
	//{
	//	const cv::Vec3b *rowptr = reinterpret_cast<const cv::Vec3b *>(src1.ptr(i));
	//	for (size_t j = 0, xBound = cellSize.width, HColIdx = 0; j < width; j++)
	//	{
	//		if (j == xBound)
	//		{
	//			HColIdx++;
	//			xBound += cellSize.width;
	//			if (HColIdx != cellCol)Hidx++;
	//		}
	//		cv::Mat &Htemp = HArray[Hidx];
	//		cv::Point2f dstPt;
	//		PointHTransform(cv::Point2f(j, i), Htemp, dstPt);
	//		cv::Point dstnPt;
	//		dstnPt.x = int(dstPt.x - dstRoi.x);
	//		dstnPt.y = int(dstPt.y - dstRoi.y);
	//		if (dstnPt.x >= 0 && dstnPt.x < dst.cols && dstnPt.y >= 0 && dstnPt.y < dst.rows)
	//		{
	//			int dstIdx = dstnPt.x + dstnPt.y * dst.cols;
	//			cv::Vec3b *dstPtr = reinterpret_cast<cv::Vec3b *>(dst.data) + dstIdx;
	//			uchar *maskPtr = dstMask.data + dstIdx;
	//
	//			if (*maskPtr == 255)
	//			{
	//				(*dstPtr) = 0.5*rowptr[j] + 0.5*(*dstPtr);
	//			}
	//			else
	//			{
	//				(*dstPtr) = rowptr[j];
	//			}
	//			if (dstnPt.y < dst.rows - 1 || dstnPt.x < dst.cols - 4)
	//			{
	//				//?????????????????????????????
	//				//(*(dstPtr + 1)) = (*dstPtr);
	//				//(*(dstPtr + 2)) = (*dstPtr);
	//				//(*(dstPtr + 3)) = (*dstPtr);
	//			}
	//		}
	//		//rowPtr[j][0] = dstPt.x;
	//		//rowPtr[j][1] = dstPt.y;
	//	}
	//	Hidx = Hidx - cellCol + 1;
	//	if (i == yBound)
	//	{
	//		HRowIdx++;
	//		yBound += cellSize.height;
	//		if (HRowIdx != cellRow)Hidx++;
	//	}
	//}
}

void RawMovingDLTStitching(const cv::Mat &src1, const cv::Mat &src2, PairInfo &pairinfo, cv::Mat &dst,
						   cv::Point cellGrid, cv::Size cellSize)
{
	//get the origin A matrix

	std::vector<cv::Point2f> srcPts1(pairinfo.inliers_num), srcPts2(pairinfo.inliers_num);
	for (size_t i = 0, j = 0; i < pairinfo.pairs_num; i++)
	{
		if (pairinfo.mask[i] == 0) { continue; }
		srcPts1[j] = pairinfo.points1[i];
		srcPts2[j] = pairinfo.points2[i];
		j++;
	}

	std::vector<cv::Point2f> pts1, pts2;
	cv::Mat T1inv, T2inv;
	GetRegularizedPoints(srcPts1, pts1, T1inv);
	GetRegularizedPoints(srcPts2, pts2, T2inv);
	/*pts1 = srcPts1;
	pts2 = srcPts2;*/

	//T1 and T2inv used for compute origin H = T2inv*H'*T1
	cv::Mat T1 = T1inv.inv();


	cv::Mat A(pairinfo.inliers_num * 2, 9, CV_64F, cv::Scalar(0));
	double *Aptr = reinterpret_cast<double *>(A.data);
	for (size_t i = 0; i < pairinfo.inliers_num; i++, Aptr += 18)
	{
		//H'*pt1 = pt2
		cv::Point2f pt1 = pts1[i];
		cv::Point2f pt2 = pts2[i];

		Aptr[3] = -pt1.x;
		Aptr[4] = -pt1.y;
		Aptr[5] = -1;
		Aptr[6] = pt2.y * pt1.x;
		Aptr[7] = pt2.y * pt1.y;
		Aptr[8] = pt2.y;

		Aptr[9] = pt1.x;
		Aptr[10] = pt1.y;
		Aptr[11] = 1;
		Aptr[15] = -pt2.x * pt1.x;
		Aptr[16] = -pt2.x * pt1.y;
		Aptr[17] = -pt2.x;
	}

	cv::Mat W, U, VT;
	cv::SVD::compute(A, W, U, VT);
	cv::Mat h = VT.row(8);
	cv::Mat globalH = h.reshape(1, 3);
	globalH = T2inv*globalH*T1;
	cv::Mat globalDst2;

	//cv::Size cellSize(50, 50);
	int width = src1.cols, height = src1.rows;
	int cellCol = cellGrid.x;
	int cellRow = cellGrid.y;

	double gamma = 0.01, sigma2 = 10000;
	//100 pixel region
	//double radius = -2 * sigma2 * log(gamma);
	double radius = 0.15 * std::min(src1.rows, src1.cols);
	radius *= radius;
	sigma2 = -radius / (2 * log(gamma));
	std::cout << "impact radius = " << sqrt(radius) << std::endl;
	std::cout << "cut gamma = " << gamma << std::endl;
	std::cout << "sigma2 = " << sigma2 << std::endl;

	std::vector<cv::Mat> HArray(cellCol*cellRow);


	std::cout << "-------------Start to Moving DLT--------------" << std::endl;
	cv::Mat showTest = src1.clone();
	DrawGrid(showTest, cellGrid, cellSize, 1, 1, false);

	for (size_t i = 0; i < pairinfo.inliers_num; i++)
	{
		cv::Scalar color = RandomColor();
		cv::circle(showTest, srcPts1[i], 5, color, -1);
	}


	for (size_t i = 0, yIdx = cellSize.height / 2, Hidx = 0; i < cellRow; i++, yIdx += cellSize.height)
	{

		for (size_t j = 0, xIdx = cellSize.width / 2; j < cellCol; j++, xIdx += cellSize.width, Hidx++)
		{

			std::vector<cv::Point2d> vCorners(4);

			vCorners[0] = cv::Point(xIdx - cellSize.width / 2, yIdx - cellSize.height / 2);
			vCorners[1] = cv::Point(xIdx + cellSize.width / 2, yIdx - cellSize.height / 2);
			vCorners[2] = cv::Point(xIdx - cellSize.width / 2, yIdx + cellSize.height / 2);
			vCorners[3] = cv::Point(xIdx + cellSize.width / 2, yIdx + cellSize.height / 2);

			cv::Mat showTmp = showTest.clone();
			cv::circle(showTmp, cv::Point(xIdx, yIdx), sqrt(radius), cv::Scalar(0, 0, 255), 2);
			cv::circle(showTmp, cv::Point(xIdx, yIdx), 10, cv::Scalar(255, 0, 0), -1);
			
			int changeCount = 0;
			cv::Mat Atemp = A.clone();

			for (size_t k = 0; k < pairinfo.inliers_num; k++)
			{
				float dx = srcPts1[k].x - xIdx;
				float dy = srcPts1[k].y - yIdx;
				double distant = dx*dx + dy*dy;
				if (distant <= radius)
				{
					double wi = exp(-distant / (2 * sigma2));
					wi /= gamma;
					cv::Mat &rowTwo = Atemp.rowRange(k * 2, k * 2 + 2);
					rowTwo *= wi;
					changeCount++;

					//cv::line(showTmp, cv::Point(xIdx, yIdx), srcPts1[k], cv::Scalar(0, 0, 255), 2);
					cv::circle(showTmp, srcPts1[k], 10, cv::Scalar(0, 255, 0), -1);
					
				}
			}
			if (changeCount > 0)
			{
				cv::Mat W_, U_, VT_;
				cv::SVD::compute(Atemp, W_, U_, VT_);
				cv::Mat h_ = VT_.row(8);
				cv::Mat H_ = h_.reshape(1, 3);
				HArray[Hidx] = T2inv*H_*T1;
				//HArray[Hidx] = H_;

				double HDist = 0;
				for (size_t m = 0; m < 4; m++)
				{
					cv::Point2d localPt, globalPt, distPt;
					PointHTransform(vCorners[m], globalH, globalPt);
					PointHTransform(vCorners[m], HArray[Hidx], localPt);
					distPt = localPt - globalPt;
					HDist = std::max(sqrt(distPt.dot(distPt)), HDist);
				}

				//HDist /= 4;
				if (HDist > 0.4 * std::min(src1.rows, src1.cols))
				{
					HArray[Hidx] = globalH;
				}

				//cv::imshow("showTmp", showTmp);
				//cv::waitKey(1);
			}
			else
			{
				HArray[Hidx] = globalH;
			}

			
		}
	}

	std::cout << "--------------End Moving DLT---------------" << std::endl;

	//decide the size of final image
	std::vector<cv::Point2f> corners(4);
	corners[0] = cv::Point2f(0, 0);
	corners[1] = cv::Point2f(src1.cols, 0);
	corners[2] = cv::Point2f(0, src1.rows);
	corners[3] = cv::Point2f(src1.cols, src1.rows);

	cv::Point2f tl(H_FLOAT_MAX, H_FLOAT_MAX);
	cv::Point2f br(H_FLOAT_MIN, H_FLOAT_MIN);
	for (auto &corner : corners)
	{
		cv::Point2f tcorner;
		int colIdx = corner.x / cellSize.width;
		int rowIdx = corner.y / cellSize.height;
		if (colIdx == cellCol)colIdx--;
		if (rowIdx == cellRow) rowIdx--;
		PointHTransform(corner, HArray[colIdx + rowIdx*cellCol], tcorner);
		if (tcorner.x < tl.x) { tl.x = tcorner.x; }
		if (tcorner.y < tl.y) { tl.y = tcorner.y; }
		if (tcorner.x > br.x) { br.x = tcorner.x; }
		if (tcorner.y > br.y) { br.y = tcorner.y; }
	}

	cv::Rect roi2(0, 0, src2.cols, src2.rows), roi1(tl, br);
	cv::Rect dstRoi = GetUnionRoi(roi1, roi2);

	roi2.x -= dstRoi.x;
	roi2.y -= dstRoi.y;

	dst = cv::Mat(dstRoi.size(), CV_8UC3);
	src2.copyTo(dst(roi2));

	cv::Mat dstMask(dstRoi.size(), CV_8U, cv::Scalar(0));
	cv::Mat src2Mask(src2.size(), CV_8U, cv::Scalar(255));
	src2Mask.copyTo(dstMask(roi2));

	int edgeExtend = 0.5 * std::min(cellSize.height, cellSize.width);
	//int edgeExtend = 0;
	for (size_t i = 0, yIdx = 0, Hidx = 0; i < cellRow; i++, yIdx += cellSize.height)
	{
		for (size_t j = 0, xIdx = 0; j < cellCol; j++, xIdx += cellSize.width, Hidx++)
		{
			//not transform the cut edge
			int cellx = xIdx, celly = yIdx, cellw = cellSize.width, cellh = cellSize.height;
			//if (j != 0)cellx -= edgeExtend;
			if (j != cellCol - 1)cellw += (1 * edgeExtend);
			//if (i != 0)celly -= edgeExtend;
			if (i != cellRow - 1) cellh += (1 * edgeExtend);
			cv::Rect cellRoi(cellx, celly, cellw, cellh);
			cv::Mat cellImg = src1(cellRoi);
			BlockStitching(cellImg, dst, cellRoi, dstRoi, dstMask, HArray[Hidx]);
		}
	}

	
}

//Compute the moving DLT grid result
//the result store the grid center coordinate and warped center corrdinate in resultPair
void GetMovingDLTGridCenter(const PairInfo &pair, const std::vector<cv::Mat> &images,
					  cv::Point gridDim, cv::Size gridSize, PairInfo &resultPair,
					  double gamma = 0.01, double radiusRatio = 0.15)
{
	//Get the valid points
	std::vector<cv::Point2f> srcPts1(pair.inliers_num), srcPts2(pair.inliers_num);
	for (size_t i = 0, j = 0; i < pair.pairs_num; i++)
	{
		if (pair.mask[i] == 0) { continue; }
		srcPts1[j] = pair.points1[i];
		srcPts2[j] = pair.points2[i];
		j++;
	}

	//Regularize the points
	std::vector<cv::Point2f> pts1, pts2;
	cv::Mat T1inv, T2inv;
	GetRegularizedPoints(srcPts1, pts1, T1inv);
	GetRegularizedPoints(srcPts2, pts2, T2inv);

	//T1 and T2inv used for compute origin H = T2inv*H'*T1
	cv::Mat T1 = T1inv.inv();

	//get the origin A matrix
	cv::Mat A(pair.inliers_num * 2, 9, CV_64F, cv::Scalar(0));
	double *Aptr = reinterpret_cast<double *>(A.data);
	for (size_t i = 0; i < pair.inliers_num; i++, Aptr += 18)
	{
		//H'*pt1 = pt2
		cv::Point2f pt1 = pts1[i];
		cv::Point2f pt2 = pts2[i];

		Aptr[3] = -pt1.x;
		Aptr[4] = -pt1.y;
		Aptr[5] = -1;
		Aptr[6] = pt2.y * pt1.x;
		Aptr[7] = pt2.y * pt1.y;
		Aptr[8] = pt2.y;

		Aptr[9] = pt1.x;
		Aptr[10] = pt1.y;
		Aptr[11] = 1;
		Aptr[15] = -pt2.x * pt1.x;
		Aptr[16] = -pt2.x * pt1.y;
		Aptr[17] = -pt2.x;
	}

	//Get the global Homograph matrix
	cv::Mat W, U, VT;
	cv::SVD::compute(A, W, U, VT);
	cv::Mat h = VT.row(8);
	cv::Mat globalH = h.reshape(1, 3);
	globalH = T2inv*globalH*T1;

	//Just for test the regular method which applied to the correspondence points
	cv::Mat globalResult, globalResultMask;
	cv::Mat mask1(images[0].size(), CV_8UC1, cv::Scalar(255));
	cv::Mat mask2(images[1].size(), CV_8UC1, cv::Scalar(255));
	GlobalHStitching(images[0], mask1, images[1], mask2, globalH, globalResult, globalResultMask);
	cv::imwrite("GlobalHStitchingNorm.jpg", globalResult);

	const cv::Mat &image1 = images[pair.index1];

	//double radius = -2 * sigma2 * log(gamma);
	double radius = radiusRatio * std::min(image1.cols, image1.rows);
	radius *= radius;
	double sigma2 = -radius / (2 * log(gamma));

	std::cout << "impact radius = " << sqrt(radius) << std::endl;
	std::cout << "cut gamma = " << gamma << std::endl;
	std::cout << "sigma2 = " << sigma2 << std::endl;

	std::vector<cv::Point2d> vCornerPt(4), vGridPt(4, cv::Point2d(0, 0));
	vGridPt[1] = cv::Point2d(gridSize.width, 0);
	vGridPt[2] = cv::Point2d(0, gridSize.height);
	vGridPt[3] = cv::Point2d(gridSize.width, gridSize.height);
	int XShift = gridDim.x * gridSize.width;


	//Get the pair information of Moving DLT which stores the origin gridCenter and warped gridCenter
	resultPair.pairs_num = resultPair.inliers_num = gridDim.x * gridDim.y;
	resultPair.mask.resize(resultPair.pairs_num, 1);
	resultPair.points1.resize(resultPair.pairs_num);
	resultPair.points2.resize(resultPair.pairs_num);
	for (int i = 0, yIdx = gridSize.height / 2, gridIdx = 0; i < gridDim.y; i++, yIdx += gridSize.height)
	{
		for (int j = 0, xIdx = gridSize.width / 2; j < gridDim.x; j++, xIdx += gridSize.width, gridIdx++)
		{
			int changeCount = 0;
			cv::Mat Atemp = A.clone();
			for (size_t k = 0; k < pair.inliers_num; k++)
			{
				float dx = srcPts1[k].x - xIdx;
				float dy = srcPts1[k].y - yIdx;
				double distant = dx*dx + dy*dy;
				if (distant <= radius)
				{
					double wi = exp(-distant / (2 * sigma2));
					wi /= gamma;
					cv::Mat &rowTwo = Atemp.rowRange(k * 2, k * 2 + 2);
					rowTwo *= wi;
					changeCount++;
				}
			}

			cv::Mat H_ = globalH;
			if (changeCount > 0)
			{
				cv::Mat W_, U_, VT_;
				cv::SVD::compute(Atemp, W_, U_, VT_);
				cv::Mat h_ = VT_.row(8);
				H_ = h_.reshape(1, 3);
				H_ = T2inv*H_*T1;

				double HMaxDist = 0;
				for (size_t m = 0; m < 4; m++)
				{
					cv::Point2d localPt, globalPt, distPt;
					PointHTransform(vGridPt[m], globalH, globalPt);
					PointHTransform(vGridPt[m], H_, localPt);
					distPt = localPt - globalPt;
					HMaxDist = std::max(sqrt(distPt.dot(distPt)), HMaxDist);
				}

				if (HMaxDist > 0.4 * std::min(image1.rows, image1.cols))
					H_ = globalH;
			}

			cv::Point2f gridCenter(xIdx, yIdx), warpedCenter;
			PointHTransform(gridCenter, H_, warpedCenter);
			resultPair.points1[gridIdx] = gridCenter;
			resultPair.points2[gridIdx] = warpedCenter;

			for (size_t m = 0; m < 4; m++)
				vGridPt[m].x += gridSize.width;
		}

		for (size_t m = 0; m < 4; m++)
		{
			vGridPt[m].y += gridSize.height;
			vGridPt[m].x -= XShift;
		}
	}
}


int main(int argc, char *argv[])
{
	std::vector<cv::Mat> images;
	std::string dir = "test1";
	if (argc == 2)
		dir = std::string(argv[1]);
	if (!LoadSameSizeImages(images, dir)) return -1;

	images.resize(2);
	int height = images[0].rows, width = images[0].cols;

	SequenceMatcher smatcher(SequenceMatcher::F_SIFT);
	std::list<PairInfo> pairinfos;
	smatcher.process(images, pairinfos);

	//Just get the first pairinfo
	PairInfo &firstPair = *(pairinfos.begin());
	double threshold = std::min(width, height) * 0.04;
	std::cout << "Homography Ransac threshold = " << threshold << std::endl;
	cv::Mat globalH = firstPair.findHomography(cv::RANSAC, threshold);
	
	cv::Point gridDim(50, 50);
	cv::Size gridSize(std::ceil(width / double(gridDim.x)), std::ceil(height / double(gridDim.y)));
	cv::Size extSize1(gridDim.x * gridSize.width, gridDim.y * gridSize.height);
	cv::Mat ext1(extSize1, CV_8UC3, cv::Scalar(0));
	images[0].copyTo(ext1(cv::Rect(0, 0, images[0].cols, images[0].rows)));
	images[0] = ext1;

	//DrawGrid(images[0], gridDim, gridSize, 2);

	cv::Mat globalResult, globalResultMask;
	cv::Mat mask1(images[0].size(), CV_8UC1, cv::Scalar(255));
	cv::Mat mask2(images[1].size(), CV_8UC1, cv::Scalar(255));
	GlobalHStitching(images[0], mask1, images[1], mask2, globalH, globalResult, globalResultMask);
	cv::imwrite("GlobalHStitching.jpg", globalResult);

	/*cv::Mat dstAPAP;
	RawMovingDLTStitching(images[0], images[1], firstPair, dstAPAP, gridDim, gridSize);
	cv::imwrite("APAPStitching.jpg", dstAPAP);*/

	PairInfo gridCenterPair;
	GetMovingDLTGridCenter(firstPair, images, gridDim, gridSize, gridCenterPair);
	std::vector<cv::Point2d> vVertices;
	EstimateGridVertices(gridCenterPair, globalH, gridDim, gridSize, images, vVertices, 10.0);

	{
		cv::Mat showTest = images[1].clone();
		DrawGrid(showTest, gridDim, gridSize, 1, 3);

		cv::Mat resultTest = images[1].clone() * 0.6;
		DrawGridVertices(resultTest, cv::Rect(0, 0, resultTest.cols, resultTest.rows), vVertices, gridDim, 1, 2);

		cv::imwrite("resultTest.jpg", resultTest);
		cv::imwrite("showTest.jpg", showTest);
	}

	cv::Mat warpedResult, warpedMask;
	cv::Rect warpedROI;
	GridWarping(images[0], cv::Mat(), gridDim, gridSize, vVertices, warpedResult, warpedMask, warpedROI);
	
	cv::imwrite("warpedResult.jpg", warpedResult);
	cv::imwrite("warpedMask.jpg", warpedMask);

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
	
	return 0;
}