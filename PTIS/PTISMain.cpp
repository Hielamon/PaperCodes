#include <OpencvCommon.h>
#include "SequenceMatcher.h"
#include <Eigen/Sparse>
#include <sstream>

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


void buildProblem(const PairInfo &pair, const cv::Mat &H, cv::Size cellNum, cv::Size cellSize,
				  std::vector<Eigen::Triplet<double>> &vTriplet, Eigen::VectorXd &b)
{
	
	int verticeNum = (cellNum.width + 1) * (cellNum.height + 1), paramNum = verticeNum * 2;
	b.resize(paramNum);
	b.setZero();
	std::vector<std::vector<int>> cellFCount(cellNum.height, std::vector<int>(cellNum.width, 0));
	std::vector<std::vector<double>> denseMatrix(paramNum, std::vector<double>(paramNum, 0));
	std::vector<double> nSumFCount(verticeNum, 0);
	//add local alignment term
	for (size_t i = 0; i < pair.pairs_num; i++)
	{
		if (!pair.mask[i]) continue;

		cv::Point2f pt1 = pair.points1[i], pt2 = pair.points2[i];
		int c = pt1.x / cellSize.width, r = pt1.y / cellSize.height;
		++cellFCount[r][c];

		int vertIdxs[4];
		vertIdxs[0] = r * (cellNum.width + 1) + c, vertIdxs[1] = vertIdxs[0] + 1;
		vertIdxs[2] = vertIdxs[0] + (cellNum.width + 1), vertIdxs[3] = vertIdxs[2] + 1;

		double coeffs[4];
		cv::Point2f localCoord(pt1.x - c * cellSize.width, pt1.y - r * cellSize.height);
		double a1 = localCoord.x / cellSize.width, a2 = localCoord.y / cellSize.height;
		coeffs[0] = a1 * a2;
		coeffs[1] = (1 - a1) * a2;
		coeffs[2] = a1 * (1 - a2);
		coeffs[3] = (1 - a1)*(1 - a2);

		for (int i = 0; i < 4; i++)
		{
			int rIdx = vertIdxs[i] * 2;
			++nSumFCount[vertIdxs[i]];
			for (size_t j = 0; j < 4; j++)
			{
				int cIdx = vertIdxs[j] * 2;
				double tmpCoeff = coeffs[i] * coeffs[j];
				denseMatrix[rIdx][cIdx] += tmpCoeff;
				denseMatrix[rIdx + 1][cIdx + 1] += tmpCoeff;
				b[rIdx] = coeffs[i] * pt2.x;
				b[rIdx + 1] = coeffs[i] * pt2.y;
			}
		}
	}

	//add global alignment term
	double alpha = 0.01;
	for (int i = 0; i < verticeNum; i++)
	{
		if (!nSumFCount[i])
		{
			int r = i / (cellNum.width + 1), c = i - r * (cellNum.width + 1);
			cv::Point2f vertPt(c * cellSize.width, r * cellSize.height), prePt;
			PointHTransform(vertPt, H, prePt);
			int rIdx = i * 2;
			denseMatrix[rIdx][rIdx] += alpha;
			denseMatrix[rIdx + 1][rIdx + 1] += alpha;
			b[rIdx] += alpha * prePt.x;
			b[rIdx + 1] += alpha * prePt.y;
		}
	}

	for (size_t i = 0; i < paramNum; i++)
	{
		for (size_t j = 0; j < paramNum; j++)
		{
			if (denseMatrix[i][j])
				vTriplet.push_back(Eigen::Triplet<double>(i, j, denseMatrix[i][j]));
		}
	}
}

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

	//add global alignment term
	double alpha = 0.01;
	for (int i = 0; i < verticeNum; i++)
	{
		//if (1)
		if (!nSumFCount[i])
		{
			int r = i / (cellNum.width + 1), c = i - r * (cellNum.width + 1);
			cv::Point2f vertPt(c * cellSize.width, r * cellSize.height), prePt;
			PointHTransform(vertPt, H, prePt);
			int rIdx = i * 2;
			denseMatrix[rIdx][rIdx] += alpha;
			denseMatrix[rIdx + 1][rIdx + 1] += alpha;
			b[rIdx] += alpha * prePt.x;
			b[rIdx + 1] += alpha * prePt.y;
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
		cv::circle(img1, pair.points1[i], 4, color, -1);
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
			cv::rectangle(img1, cv::Rect(tl, br), color, 2);
			std::stringstream ioStr;
			ioStr << i*(cellNum.width + 1) + j << ":" << rowFCount[j];

			cv::Point org(tl.x + cellSize.width * 0.1, br.y - cellSize.height * 0.5);
			cv::putText(img1, ioStr.str(), org, cv::HersheyFonts::FONT_HERSHEY_COMPLEX, 0.5, color, 1);
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
	double threshold = std::min(width, height) * 0.08;
	cv::Mat globalH = cv::findHomography(firstPair.points1, firstPair.points2, firstPair.mask, cv::RANSAC, threshold);
	firstPair.inliers_num = 0;
	for (auto &mask : firstPair.mask)
		if (mask != 0) firstPair.inliers_num++;

	cv::Size cellNum(40, 40), cellSize(std::ceil(width / double(cellNum.width)), std::ceil(height / double(cellNum.height)));
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
	
	for (int i = 0; i < verticeNum; i++)
	{
		int r = i / (cellNum.width + 1), c = i - r * (cellNum.width + 1);
		cv::Point vertPt(c * cellSize.width, r * cellSize.height);
		//cv::circle(showTest, vertPt, 4, cv::Scalar(0, 255, 0), -1);
		cv::Point downPt(vertPt.x, vertPt.y + cellSize.height);
		cv::Point rightPt(vertPt.x + cellSize.width, vertPt.y);
		cv::line(showTest, vertPt, downPt, cv::Scalar(0, 255, 255), 2);
		cv::line(showTest, vertPt, rightPt, cv::Scalar(0, 255, 255), 2);

		vertPt.x = result[i * 2];
		vertPt.y = result[i * 2 + 1];
		
		//cv::circle(resultTest, vertPt + offset, 4, cv::Scalar(255, 0, 0), -1);

		if (c < cellNum.width)
		{
			int j = i + 1;
			rightPt.x = result[j * 2];
			rightPt.y = result[j * 2 + 1];
			cv::line(resultTest, vertPt + offset, rightPt + offset, cv::Scalar(255, 255, 0), 2);
		}

		if(r < cellNum.height)
		{
			int j = i + cellNum.width + 1;
			downPt.x = result[j * 2];
			downPt.y = result[j * 2 + 1];
			cv::line(resultTest, vertPt + offset, downPt + offset, cv::Scalar(255, 255, 0), 2);
		}
	}

	for (int i = 0; i < verticeNum; i++)
	{
		int r = i / (cellNum.width + 1), c = i - r * (cellNum.width + 1);
		cv::Point vertPt(c * cellSize.width, r * cellSize.height);
		cv::circle(showTest, vertPt, 4, cv::Scalar(0, 255, 0), -1);

		vertPt.x = result[i * 2];
		vertPt.y = result[i * 2 + 1];

		cv::circle(resultTest, vertPt + offset, 4, cv::Scalar(255, 0, 0), -1);
	}

	cv::imwrite("resultTest.jpg", resultTest);
	cv::imwrite("showTest.jpg", showTest);
	

	cv::Mat dstGlobal;
	GlobalHStitching(images[0], images[1], globalH, dstGlobal);
	cv::imwrite("GlobalHStitching.jpg", dstGlobal);

	DrawPairInfos(images, pairinfos, true);

	return 0;
}