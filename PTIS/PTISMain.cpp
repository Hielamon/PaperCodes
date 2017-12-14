#define MAIN_FILE
#include <commonMacro.h>

#include <OpencvCommon.h>
#include "../common/SequenceMatcher.h"
#include <Eigen/Sparse>
#include <sstream>
#include "../common/stitchingCommonFuc.h"

int dotR = 2, lineW = 2;

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
	double gamma = 1;
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
				double tmpCoeff = coeffs[i] * coeffs[j] * gamma;
				denseMatrix[rIdx][cIdx] += tmpCoeff;
				denseMatrix[rIdx + 1][cIdx + 1] += tmpCoeff;
			}

			b[rIdx] += coeffs[i] * pt2.x * gamma;
			b[rIdx + 1] += coeffs[i] * pt2.y * gamma;
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
		if (1)
		//if (!nSumFCount[i])
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
	
	cv::Size resultSize(cellNum.width * cellSize.width, cellNum.height*cellSize.height);
	cv::Rect originROI(0, 0, images[pair.index1].cols, images[pair.index1].rows);
	cv::Mat img1(resultSize, CV_8UC3, cv::Scalar(0));
	images[pair.index1].copyTo(img1(originROI));
	
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
			cv::Rect cellRoi(tl, br);

			cv::rectangle(img1, cellRoi, color, lineW);
			//cv::Mat colorRect(cellRoi.height, cellRoi.width, CV_8UC3, color);

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
	std::string dir = "test5";
	if (argc == 2)
		dir = std::string(argv[1]);
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
	std::vector<cv::Point2d> vVertices;

	bool forTest = false;
	if(forTest)
	{
		int verticeNum = (cellNum.width + 1) * (cellNum.height + 1), paramNum = verticeNum * 2;
		Eigen::VectorXd b(paramNum), result;
		std::vector<Eigen::Triplet<double>> vTriplet;

		buildProblemTest(firstPair, globalH, cellNum, gridSize, vTriplet, b, images);

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


		vVertices.resize(verticeNum);
		for (int i = 0, pIdx = 0; i < verticeNum; i++, pIdx += 2)
		{
			vVertices[i].x = result[pIdx];
			vVertices[i].y = result[pIdx + 1];
			if (isnan(result[pIdx]) || isnan(result[pIdx + 1]))
				HL_CERR("result " << pIdx << " is not valid");
		}
	}
	else
	{
		EstimateGridVertices(firstPair, globalH, gridDim, gridSize, images, vVertices, 3);
	}

	{
		cv::Mat showTest = images[1].clone();
		DrawGrid(showTest, gridDim, gridSize, 1, 3);

		cv::Mat resultTest = images[1].clone() * 0.6;
		DrawGridVertices(resultTest, cv::Rect(0, 0, resultTest.cols, resultTest.rows), vVertices, gridDim, 2, 3);

		cv::imwrite("resultTest.jpg", resultTest);
		cv::imwrite("showTest.jpg", showTest);
	}

	cv::Mat globalResult, globalResultMask;
	cv::Mat mask1(images[0].size(), CV_8UC1, cv::Scalar(255));
	cv::Mat mask2(images[1].size(), CV_8UC1, cv::Scalar(255));
	GlobalHStitching(images[0], mask1, images[1], mask2, globalH, globalResult, globalResultMask);
	cv::imwrite("GlobalHStitching.jpg", globalResult);

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

	return 0;
}