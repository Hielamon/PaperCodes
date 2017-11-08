#pragma once
#include <OpencvCommon.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Sparse>
#include <commonMacro.h>

inline void GridMapping(cv::Mat Hinv, cv::Rect srcROI, cv::Rect dstROI, 
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

//the size of src should be equal to the gridSize * gridDim
inline void GridWarping(const cv::Mat& src, const cv::Mat& srcMask, const cv::Point& gridDim, const cv::Size &gridSize,
						const std::vector<cv::Point2d> &vVertices, 
						cv::Mat &dst, cv::Mat &mask, cv::Rect &resultROI)
{
	assert(src.type() == CV_8UC3);
	assert(srcMask.type() == CV_8UC1);
	assert(vVertices.size() == (gridDim.x + 1) * (gridDim.y + 1));
	assert(src.rows == gridDim.y*gridSize.height && src.cols == gridDim.x*gridSize.width);
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
			cornerTl = vVertices[vCornerIdx[0]];
			cornerBr = cornerTl;
			for (size_t m = 0; m < 4; m++)
			{
				vCornerPt[m] = vVertices[vCornerIdx[m]];

				cv::Point2d &pt = vCornerPt[m];
				if (cornerTl.x > pt.x)cornerTl.x = floor(pt.x);
				if (cornerTl.y > pt.y)cornerTl.y = floor(pt.y);
				if (cornerBr.x < pt.x)cornerBr.x = ceil(pt.x);
				if (cornerBr.y < pt.y)cornerBr.y = ceil(pt.y);
			}

			std::vector<uchar> inlierMask(4, 1);
			cv::Mat gridHinv = cv::findHomography(vCornerPt, vGridPt, inlierMask, cv::LMEDS);

			cv::Rect srcGridROI(vGridPt[0], vGridPt[3]), dstGridROI(cornerTl, cornerBr),
				dstGridROI_(cornerTl + vertShift, cornerBr + vertShift);
			GridMapping(gridHinv, srcGridROI, dstGridROI, src(srcGridROI), dst(dstGridROI_), srcMask(srcGridROI), mask(dstGridROI_));

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

//simplest average merging function
inline void AverageMerge(const std::vector<cv::Mat> &vImage, const std::vector<cv::Mat> &vMask,
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
}

//Get the vertice warp problem's sparse matrix elements vTriplet and b
//vPresetVert is the preset of vertices , it is always set according the global Homography Matrix
//gamma : control the weight of point-correspond constraint
//alpha : control the weight of preset vertices constraint
//beta  : control the weight of shape constraint which is inroduced from content-preserving warp
inline void buildProblem(const PairInfo &pair, const std::vector<cv::Point2d> &vPresetVert,
				  const std::vector<cv::Mat> &images, cv::Point gridDim, cv::Size gridSize,
				  std::vector<Eigen::Triplet<double>> &vTriplet, Eigen::VectorXd &b,
				  double gamma = 1, double alpha = 1e-4, double beta = 1e-5)
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
			if (!nSumFCount[i])
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
		
		cv::Size extSize(gridSize.width * gridDim.x, gridSize.height * gridDim.y);
		cv::Mat extMask(extSize, CV_8UC1, cv::Scalar(0));
		cv::Mat extImg(extSize, CV_8UC3, cv::Scalar(0));

		const cv::Mat &image1 = images[pair.index1];
		cv::Rect image1Roi(0, 0, image1.cols, image1.rows);
		cv::rectangle(extMask, image1Roi, cv::Scalar(255), -1);
		image1.copyTo(extImg(image1Roi));

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

					double ws = vCovL2Value[m / 2] + 50;

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
}


//Estimate the warped vertices of the grid
//gamma, alpha, and beta are same with buildProblem
inline void EstimateGridVertices(const PairInfo &pair, const cv::Mat &presetH, cv::Point gridDim,
						  cv::Size gridSize, const std::vector<cv::Mat> &images,
						  std::vector<cv::Point2d> &vVertices, double gamma = 1, double alpha = 1e-4, double beta = 1e-5)
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

	buildProblem(pair, vPresetVert, images, gridDim, gridSize, vTriplet, b, gamma, alpha, beta);

	Eigen::SparseMatrix<double> A(paramNum, paramNum);
	A.setFromTriplets(vTriplet.begin(), vTriplet.end());

	Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
	solver.compute(A);
	if (solver.info() != Eigen::Success)
		HL_CERR("Failed to compute sparse Matrix");

	x = solver.solve(b);
	if (solver.info() != Eigen::Success)
		HL_CERR("Failed to solve the result");

	vVertices.resize(verticeNum);
	for (int i = 0, pIdx = 0; i < verticeNum; i++, pIdx += 2)
	{
		vVertices[i].x = x[pIdx];
		vVertices[i].y = x[pIdx + 1];
		if (isnan(x[pIdx]) || isnan(x[pIdx + 1]))
			HL_CERR("x " << pIdx << " is not valid");
	}

}

