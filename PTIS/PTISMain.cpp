#include <OpencvCommon.h>
#include "SequenceMatcher.h"
#include <Eigen/Sparse>
#include <sstream>

int dotR = 2, lineW = 2;

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
		PointHTransform(corner, H, tcorner);
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

void GlobalHStitching(const cv::Mat &src1, const cv::Mat &src2, const cv::Mat &H, cv::Mat &dst)
{
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
		PointHTransform(corner, H, tcorner);
		if (tcorner.x < tl.x) { tl.x = tcorner.x; }
		if (tcorner.y < tl.y) { tl.y = tcorner.y; }
		if (tcorner.x > br.x) { br.x = tcorner.x; }
		if (tcorner.y > br.y) { br.y = tcorner.y; }
	}
	/*for (size_t i = 0; i < src1.rows; i++)
	{
	for (size_t j = 0; j < src1.cols; j++)
	{
	if (i > 0 && i < src1.rows - 1 && j > 0 && j < src1.cols - 1)continue;
	cv::Point2f boundaryPt(j, i);
	cv::Point2f tcorner;
	PointHTransform(boundaryPt, H, tcorner);
	if (tcorner.x < tl.x) { tl.x = tcorner.x; }
	if (tcorner.y < tl.y) { tl.y = tcorner.y; }
	if (tcorner.x > br.x) { br.x = tcorner.x; }
	if (tcorner.y > br.y) { br.y = tcorner.y; }
	}
	}*/
	/*tl = cv::Point2f(-src2.cols, -src2.rows);
	br = cv::Point2f(src2.cols*2, src2.rows*2);*/


	cv::Rect roi2(0, 0, src2.cols, src2.rows), roi1(tl, br);
	cv::Rect dstRoi = GetUnionRoi(roi1, roi2);

	roi2.x -= dstRoi.x;
	roi2.y -= dstRoi.y;

	dst = cv::Mat(dstRoi.size(), CV_8UC3);
	src2.copyTo(dst(roi2));

	cv::Mat dstMask(dstRoi.size(), CV_8U, cv::Scalar(0));
	cv::Mat src2Mask(src2.size(), CV_8U, cv::Scalar(255));
	src2Mask.copyTo(dstMask(roi2));

	BlockStitching(src1, dst, cv::Rect(0, 0, src1.cols, src1.rows), dstRoi, dstMask, H);
}

const int vTriangles[8][3] = {
	{0, 2, 3},
	{2, 3, 0},
	{0, 1, 2},
	{2, 0, 1},
	{1, 3, 0},
	{3, 0, 1},
	{1, 2, 3},
	{3, 1, 2}
};

void buildProblemTest(const PairInfo &pair, const cv::Mat &H, cv::Size cellNum, cv::Size cellSize,
				  std::vector<Eigen::Triplet<double>> &vTriplet, Eigen::VectorXd &b, std::vector<cv::Mat> &images)
{

	int verticeNum = (cellNum.width + 1) * (cellNum.height + 1), paramNum = verticeNum * 2;
	b.resize(paramNum);
	b.setZero();
	std::vector<std::vector<int>> cellFCount(cellNum.height, std::vector<int>(cellNum.width, 0));
	std::vector<std::vector<double>> denseMatrix(paramNum, std::vector<double>(paramNum, 0));
	std::vector<double> nSumFCount(verticeNum, 0);
	//add local alignment term
	for (size_t k = 0; k < pair.pairs_num; k++)
	{
		if (!pair.mask[k]) continue;

		cv::Point2f pt1 = pair.points1[k], pt2 = pair.points2[k];
		int c = pt1.x / cellSize.width, r = pt1.y / cellSize.height;
		++cellFCount[r][c];

		int vertIdxs[4];
		vertIdxs[0] = r * (cellNum.width + 1) + c, vertIdxs[1] = vertIdxs[0] + 1;
		vertIdxs[2] = vertIdxs[0] + (cellNum.width + 1), vertIdxs[3] = vertIdxs[2] + 1;

		double coeffs[4];
		cv::Point2f localCoord(pt1.x - c * cellSize.width, pt1.y - r * cellSize.height);
		double a1 =  1 - (localCoord.x / cellSize.width), a2 = 1 - (localCoord.y / cellSize.height);
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
				double tmpCoeff = coeffs[i] * coeffs[j];
				denseMatrix[rIdx][cIdx] += tmpCoeff;
				denseMatrix[rIdx + 1][cIdx + 1] += tmpCoeff;
			}

			b[rIdx] += coeffs[i] * pt2.x;
			b[rIdx + 1] += coeffs[i] * pt2.y;
		}
	}

	//calculate the pre-warped vertices
	std::vector<cv::Point2f> vPreWarpedPt(verticeNum);
	for (size_t i = 0; i < verticeNum; i++)
	{
		int r = i / (cellNum.width + 1), c = i - r * (cellNum.width + 1);
		cv::Point2f vertPt(c * cellSize.width, r * cellSize.height);
		if (!PointHTransform(vertPt, H, vPreWarpedPt[i]))
		{
			std::cout << "Failed to Transform point by Homograph Matrix" << std::endl;
			exit(-1);
		}
	}

	//add global alignment term
	double alpha = 1e-4;
	for (int i = 0; i < verticeNum; i++)
	{
		//if (1)
		if (!nSumFCount[i])
		{
			int rIdx = i * 2;
			denseMatrix[rIdx][rIdx] += alpha;
			denseMatrix[rIdx + 1][rIdx + 1] += alpha;
			b[rIdx] += alpha * vPreWarpedPt[i].x;
			b[rIdx + 1] += alpha * vPreWarpedPt[i].y;
		}
	}

	//calculate the variance of triangles
	//and add smoothness term
	cv::Mat squareMask(cellSize, CV_8UC1, cv::Scalar(0));
	cv::line(squareMask, cv::Point(0, 0), cv::Point(squareMask.cols - 1, squareMask.rows - 1), cv::Scalar(1));
	cv::line(squareMask, cv::Point(squareMask.cols - 1, 0), cv::Point(0, squareMask.rows - 1), cv::Scalar(2));
	cv::Size extSize(cellSize.width * cellNum.width, cellSize.height * cellNum.height);
	cv::Mat extMask(extSize, CV_8UC1, cv::Scalar(0));
	cv::Mat extImg(extSize, CV_8UC3, cv::Scalar(0));
	int index1 = pair.index1;
	cv::Rect originRoi(0, 0, images[index1].cols, images[index1].rows);
	cv::rectangle(extMask, originRoi, cv::Scalar(255), -1);
	images[index1].copyTo(extImg(originRoi));

	double beta = 0.001, beta2 = 0.01;
	
	for (size_t i = 0; i < cellNum.height; i++)
	{
		for (size_t j = 0; j < cellNum.width; j++)
		{
			cv::Rect cellRoi(j * cellSize.width, i * cellSize.height, cellSize.width, cellSize.height);
			std::vector<cv::Vec3d> vSumColor(4, cv::Vec3d(0,0,0)) , vAvgColor(4);
			std::vector<int> vCount(4, 1);
			cv::Mat cellMat = extImg(cellRoi), cellMask = extMask(cellRoi);

			//calculate the sum of different triangle regions
			for (size_t m = 0; m < cellSize.height; m++)
			{
				cv::Vec3b *pRowImg = reinterpret_cast<cv::Vec3b *>(cellMat.ptr(m));
				uchar *pRowMask = reinterpret_cast<uchar *>(cellMask.ptr(m));
				uchar *pRowSquareMask = reinterpret_cast<uchar *>(squareMask.ptr(m));
				bool across1 = false, across2 = false;
				for (size_t n = 0; n < cellSize.width; n++)
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
			for (size_t m = 0; m < cellSize.height; m++)
			{
				cv::Vec3b *pRowImg = reinterpret_cast<cv::Vec3b *>(cellMat.ptr(m));
				uchar *pRowMask = reinterpret_cast<uchar *>(cellMask.ptr(m));
				uchar *pRowSquareMask = reinterpret_cast<uchar *>(squareMask.ptr(m));
				bool across1 = false, across2 = false;
				for (size_t n = 0; n < cellSize.width; n++)
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
			vVertIdx[0] = i * (cellNum.width + 1) + j; vVertIdx[1] = vVertIdx[0] + 1;
			vVertIdx[3] = vVertIdx[0] + (cellNum.width + 1); vVertIdx[2] = vVertIdx[3] + 1;

			for (int m = 0; m < 8; m++)
			{
				int idx1 = vVertIdx[vTriangles[m][0]], idx2 = vVertIdx[vTriangles[m][1]], idx3 = vVertIdx[vTriangles[m][2]];
				cv::Point2f &pt1 = vPreWarpedPt[idx1], &pt2 = vPreWarpedPt[idx2], &pt3 = vPreWarpedPt[idx3];
				cv::Point2f pt21 = pt1 - pt2;
				cv::Point2f pt23 = pt3 - pt2;
				double u = pt21.dot(pt23) / pt23.dot(pt23);

				cv::Point2f R90pt23(pt23.y, -pt23.x);
				double v = pt21.dot(R90pt23) / R90pt23.dot(R90pt23);

				double ws = vCovL2Value[m / 2] * beta2 + 0.5;
				if (std::isnan(ws))
					std::cout << "ws = " << ws << std::endl;

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

	for (size_t i = 0; i < paramNum; i++)
		for (size_t j = 0; j < paramNum; j++)
			if (denseMatrix[i][j])
				vTriplet.push_back(Eigen::Triplet<double>(i, j, denseMatrix[i][j]));

	assert(images.size() == 2);
	cv::Mat img1 = images[pair.index1].clone();
	
	for (size_t i = 0; i < pair.pairs_num; i++)
	{
		if (!pair.mask[i])continue;
		cv::Scalar color = RandomColor();
		cv::circle(img1, pair.points1[i], dotR, color, -1);
	}

	for (size_t i = 0; i < cellFCount.size(); i++)
	{
		std::vector<int> &rowFCount = cellFCount[i];
		for (size_t j = 0; j < rowFCount.size(); j++)
		{
			cv::Point tl(j*cellSize.width, i*cellSize.height);
			cv::Point br(tl.x + cellSize.width, tl.y + cellSize.height);
			cv::Scalar color(0, 255, 0);
			if (rowFCount[j])
				color = cv::Scalar(0, 0, 255);
			//cv::rectangle(img1, cv::Rect(), color, lineW);
			cv::Rect cellRoi(tl, br);
			cv::Mat colorRect(cellRoi.height, cellRoi.width, CV_8UC3, color);
			double gamma = 0.1;
			img1(cellRoi) = img1(cellRoi)*(1 - gamma) + img1(cellRoi).mul(colorRect) * gamma;
			
			std::stringstream ioStr;
			ioStr << i*(cellNum.width + 1) + j << ":" << rowFCount[j];

			cv::Point org(tl.x + cellSize.width * 0.1, br.y - cellSize.height * 0.5);
			//cv::putText(img1, ioStr.str(), org, cv::HersheyFonts::FONT_HERSHEY_COMPLEX, 0.5, color, 1);
		}
	}

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

	cv::imwrite("gridDetail.jpg", img1);
}

int main(int argc, char *argv[])
{
	std::vector<cv::Mat> images;
	LoadSameSizeImages(images, "test3");
	images.resize(2);
	int height = images[0].rows, width = images[0].cols;

	SequenceMatcher smatcher(SequenceMatcher::F_SIFT);
	std::list<PairInfo> pairinfos;
	smatcher.process(images, pairinfos);

	PairInfo &firstPair = *(pairinfos.begin());
	double threshold = std::min(width, height) * 0.05;
	cv::Mat globalH = cv::findHomography(firstPair.points1, firstPair.points2, firstPair.mask, cv::RANSAC, threshold);
	firstPair.inliers_num = 0;
	for (auto &mask : firstPair.mask)
		if (mask != 0) firstPair.inliers_num++;

	cv::Size cellNum(50, 50), cellSize(std::ceil(width / double(cellNum.width)), std::ceil(height / double(cellNum.height)));
	int verticeNum = (cellNum.width + 1) * (cellNum.height + 1), paramNum = verticeNum * 2;
	Eigen::VectorXd b(paramNum), result;
	std::vector<Eigen::Triplet<double>> vTriplet;

	buildProblemTest(firstPair, globalH, cellNum, cellSize, vTriplet, b, images);
	//buildProblem(firstPair, globalH, cellNum, cellSize, vTriplet, b);
	
	Eigen::SparseMatrix<double> A(paramNum, paramNum);
	A.setFromTriplets(vTriplet.begin(), vTriplet.end());
	
	Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
	solver.compute(A);
	if (solver.info() != Eigen::Success)
	{
		std::cout << "Failed to compute sparse Matrix" << std::endl;
		return -1;
	}

	result = solver.solve(b);
	if (solver.info() != Eigen::Success)
	{
		std::cout << "Failed to solve the result" << std::endl;
		return -1;
	}

	cv::Mat showTest(cellNum.height * cellSize.height, cellNum.width * cellSize.width, CV_8UC3, cv::Scalar(0));
	cv::Rect oriRoi(0, 0, width, height);
	images[1].copyTo(showTest(oriRoi));

	cv::Point resTl(result[0], result[1]), resBr = resTl;
	for (int i = 0; i < paramNum; i += 2)
	{
		if (std::isnan(result[i]) || std::isnan(result[i + 1]))
		{
			std::cout << "Invalid Result" << std::endl;
			exit(-1);
		}
		if (resTl.x > result[i])resTl.x = result[i];
		if (resTl.y > result[i + 1])resTl.y = result[i + 1];
		if (resBr.x < result[i])resBr.x = result[i];
		if (resBr.y < result[i + 1])resBr.y = result[i + 1];
	}

	cv::Rect resultRoi(resTl, resBr);
	oriRoi.width = showTest.cols;
	oriRoi.height = showTest.rows;
	resultRoi = GetUnionRoi(resultRoi, oriRoi);
	cv::Mat resultTest(resultRoi.height, resultRoi.width, CV_8UC3, cv::Scalar(0));
	cv::Point offset = -resultRoi.tl();
	showTest.copyTo(resultTest(cv::Rect(oriRoi.x + offset.x, oriRoi.y + offset.y, oriRoi.width, oriRoi.height)));
	
	resultTest *= 0.6;
	for (int i = 0; i < verticeNum; i++)
	{
		int r = i / (cellNum.width + 1), c = i - r * (cellNum.width + 1);
		cv::Point vertPt(c * cellSize.width, r * cellSize.height);
		//cv::circle(showTest, vertPt, 4, cv::Scalar(0, 255, 0), -1);
		cv::Point downPt(vertPt.x, vertPt.y + cellSize.height);
		cv::Point rightPt(vertPt.x + cellSize.width, vertPt.y);
		cv::line(showTest, vertPt, downPt, cv::Scalar(0, 255, 255), lineW);
		cv::line(showTest, vertPt, rightPt, cv::Scalar(0, 255, 255), lineW);

		vertPt.x = result[i * 2];
		vertPt.y = result[i * 2 + 1];
		
		//cv::circle(resultTest, vertPt + offset, 4, cv::Scalar(255, 0, 0), -1);

		if (c < cellNum.width)
		{
			int j = i + 1;
			rightPt.x = result[j * 2];
			rightPt.y = result[j * 2 + 1];
			cv::line(resultTest, vertPt + offset, rightPt + offset, cv::Scalar(255, 0, 255), lineW);
		}

		if(r < cellNum.height)
		{
			int j = i + cellNum.width + 1;
			downPt.x = result[j * 2];
			downPt.y = result[j * 2 + 1];
			cv::line(resultTest, vertPt + offset, downPt + offset, cv::Scalar(0, 255, 255), lineW);
		}
	}

	/*for (int i = 0; i < verticeNum; i++)
	{
		int r = i / (cellNum.width + 1), c = i - r * (cellNum.width + 1);
		cv::Point vertPt(c * cellSize.width, r * cellSize.height);
		cv::circle(showTest, vertPt, dotR, cv::Scalar(0, 255, 0), -1);

		vertPt.x = result[i * 2];
		vertPt.y = result[i * 2 + 1];

		cv::circle(resultTest, vertPt + offset, dotR, cv::Scalar(255, 0, 0), -1);
	}*/

	cv::imwrite("resultTest.jpg", resultTest);
	cv::imwrite("showTest.jpg", showTest);
	

	cv::Mat dstGlobal;
	GlobalHStitching(images[0], images[1], globalH, dstGlobal);
	cv::imwrite("GlobalHStitching.jpg", dstGlobal);

	DrawPairInfos(images, pairinfos, true);

	return 0;
}