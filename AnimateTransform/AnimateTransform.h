#pragma once
#include "../common/stitchingCommonFuc.h"
#include "../common/SequenceMatcher.h"
#include <sstream>
#include <iomanip>
#include <direct.h>
#include <io.h>

inline cv::Mat Get2DRotation(double theta)
{
	cv::Mat R2D = cv::Mat::eye(2, 2, CV_64FC1);
	double s = sin(theta), c = cos(theta);
	R2D.at<double>(0, 0) = c;
	R2D.at<double>(1, 0) = -s;
	R2D.at<double>(0, 1) = s;
	R2D.at<double>(1, 1) = c;
	return R2D;
}

inline void GetROICorners(const cv::Rect& ROI, std::vector<cv::Point2d>& vCorner)
{
	vCorner.resize(4);
	vCorner[0] = cv::Point2d(ROI.x, ROI.y);
	vCorner[1] = cv::Point2d(ROI.x + ROI.width, ROI.y);
	vCorner[2] = cv::Point2d(ROI.x, ROI.y + ROI.height);
	vCorner[3] = cv::Point2d(ROI.x + ROI.width, ROI.y + ROI.height);
}

class WarperBase
{
public:
	WarperBase(cv::Point gridDim = cv::Point(1, 1), cv::Size gridSize = cv::Size(0, 0))
		: mGridDim(gridDim), mGridSize(gridSize)
	{}
	~WarperBase() {}

	virtual void setWarpRatio(double tRatio) = 0;
	virtual void warp(const cv::Mat &src, const cv::Mat &srcMask, const cv::Rect &srcROI,
					  cv::Mat &dst, cv::Mat &dstMask, cv::Rect &dstROI) = 0;

	virtual void drawWarpedGrid(cv::Mat &src, cv::Rect &srcROI, int lineW = 2)
	{
		if (mvVertices.size() == 0 || mvVertices.size() != (mGridDim.x + 1) * (mGridDim.y + 1)) return;

		srcROI = DrawGridVertices(src, srcROI, mvVertices, mGridDim, lineW);
	}
protected:
	//The value indicate the percent of the warp effect
	//Which is a [0~1] number
	//double mTRatio;

	std::vector<cv::Point2d> mvVertices;

	cv::Point mGridDim;
	cv::Size mGridSize;
};

class HomographWarper : public WarperBase
{
public:
	HomographWarper(double theta, double phi, double sx, double sy, double t1, double t2,
					double v1, double v2) :
		mtheta(theta), mphi(phi), msx(sx), msy(sy), mt1(t1), mt2(t2), mv1(v1), mv2(v2)
	{
	};

	~HomographWarper() {};

	virtual void setWarpRatio(double tRatio)
	{
		mH = _homographTransform(mtheta, mphi, msx, msy, mt1, mt2, mv1, mv2, tRatio);
	}

	virtual void warp(const cv::Mat &src, const cv::Mat &srcMask, const cv::Rect &srcROI,
					  cv::Mat &dst, cv::Mat &dstMask, cv::Rect &dstROI)
	{
		std::vector<cv::Point2d> vSrcCorner(4), vSrcWarpedCorner(4);
		mGridSize = src.size();

		GetROICorners(srcROI, vSrcCorner);
		PointSetHTransform(vSrcCorner, mH, vSrcWarpedCorner);
		mvVertices = vSrcWarpedCorner;

		GridWarping(src, srcMask, cv::Point(1, 1), cv::Size(src.size()),
					vSrcWarpedCorner, dst, dstMask, dstROI);

		//dstROI = DrawGridVertices(dst, dstROI, vSrcWarpedCorner, cv::Point(1, 1), 10, 0, false);
	}

private:
	//the reasonable parameters of Homograph matrix
	double mtheta, mphi, msx, msy, mt1, mt2, mv1, mv2;

	//The percent Homograph Matrix
	cv::Mat mH;

	cv::Mat _homographTransform(double theta, double phi, double sx, double sy, double t1, double t2, double v1, double v2, double tRatio)
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

};

class PTISWarper : public WarperBase
{
public:
	//mGammaRange: the range of gamma parameter, mGammaRange[0] ~ mGammaRange[1]
	//mAlphaRange: the range of gamma parameter, mAlphaRange[0] ~ mAlphaRange[1]
	//mBetaRange: the range of gamma parameter, mBetaRange[0] ~ mBetaRange[1]
	//Other parameters are similar to buildProblem function in stitchingCommonFuc.h file
	PTISWarper(const PairInfo &pair, const std::vector<cv::Mat> &vImages,
			   const cv::Mat &presetH, cv::Point gridDim, cv::Size gridSize, 
			   cv::Vec2d gammaRange, cv::Vec2d alphaRange, cv::Vec2d betaRange
	) : WarperBase(gridDim, gridSize), mpair(pair), mvImages(vImages),
		mpresetH(presetH), mGammaRange(gammaRange), mAlphaRange(alphaRange),
		mBetaRange(betaRange)
	{ }

	~PTISWarper() {}

	virtual void setWarpRatio(double tRatio)
	{
		double gamma = tRatio * (mGammaRange[1] - mGammaRange[0]) + mGammaRange[0];
		double alpha = tRatio * (mAlphaRange[1] - mAlphaRange[0]) + mAlphaRange[0];
		double beta = tRatio * (mBetaRange[1] - mBetaRange[0]) + mBetaRange[0];

		EstimateGridVertices(mpair, mpresetH, mGridDim, mGridSize, mvImages, mvVertices,
							 gamma, alpha, beta);
	}

	virtual void warp(const cv::Mat &src, const cv::Mat &srcMask, const cv::Rect &srcROI,
					  cv::Mat &dst, cv::Mat &dstMask, cv::Rect &dstROI)
	{
		GridWarping(src, srcMask, mGridDim, mGridSize,
					mvVertices, dst, dstMask, dstROI);

		//dstROI = DrawOuterContour(dst, dstROI, mvVertices, mGridDim, 3);
		//dstROI = DrawGridVertices(dst, dstROI, mvVertices, mGridDim, 2);
	}


private:
	

	const PairInfo &mpair;
	const std::vector<cv::Mat> &mvImages;
	const cv::Mat &mpresetH;

	cv::Vec2d mGammaRange, mAlphaRange, mBetaRange;
};


class AnimateTransform
{
public:
	AnimateTransform(const std::shared_ptr<WarperBase> pWarper, int fraction, bool bOrder = false)
		: mpWarper(pWarper), mfraction(fraction), mbOrder(bOrder)
	{}

	~AnimateTransform() {}

	void run(const cv::Mat &src1, const cv::Mat &src1Mask, const cv::Rect &ROI1,
			 const cv::Mat &src2, const cv::Mat &src2Mask, const cv::Rect &ROI2,
			 std::string saveFolderName)
	{
		mvResult.clear();
		mvResultROI.clear();

		runCurrentAnimate(src1, src1Mask, ROI1, src2, src2Mask, ROI2);

		normalizeROI();

		saveResults(saveFolderName, true);
	}

	void resetAnimateParam(const std::shared_ptr<WarperBase> pWarper, int fraction, bool bOrder = false)
	{
		mpWarper = pWarper;
		mfraction = fraction;
		mbOrder = bOrder;
	}

	void runCurrentAnimate(const cv::Mat &src1, const cv::Mat &src1Mask, const cv::Rect &ROI1,
						   const cv::Mat &src2, const cv::Mat &src2Mask, const cv::Rect &ROI2)
	{
		for (size_t i = 0; i <= mfraction; i++)
		{
			double tRatio = mfraction ? double(mbOrder ? i : mfraction - i) / mfraction : 0;
			mpWarper->setWarpRatio(tRatio);
			cv::Mat src2Warped, src2MaskWarped;
			cv::Rect ROI2Warped;
			mpWarper->warp(src2, src2Mask, ROI2, src2Warped, src2MaskWarped, ROI2Warped);

			//mpWarper->drawWarpedGrid(src2Warped, ROI2Warped, 2);

			std::vector<cv::Mat> vPreparedImg, vPreparedMask;
			std::vector<cv::Rect> vROI;

			vPreparedImg.push_back(src1);
			vPreparedMask.push_back(src1Mask);
			vROI.push_back(ROI1);

			vPreparedImg.push_back(src2Warped);
			vPreparedMask.push_back(src2MaskWarped);
			vROI.push_back(ROI2Warped);

			cv::Mat currResultImg, currResultImgMask;
			cv::Rect currResultROI;
			currResultROI = AverageMerge(vPreparedImg, vPreparedMask, vROI, currResultImg, currResultImgMask);

			mpWarper->drawWarpedGrid(currResultImg, currResultROI, 3);

			mvResultROI.push_back(currResultROI);
			mvResult.push_back(currResultImg);
		}
	}

	void normalizeROI()
	{
		if (mvResultROI.size() == 0)return;
		cv::Rect finalROI = mvResultROI[0];
		for (size_t i = 1; i < mvResultROI.size(); i++)
			finalROI = GetUnionRoi(finalROI, mvResultROI[i]);

		for (size_t i = 0; i < mvResult.size(); i++)
		{
			cv::Mat tmp(finalROI.height, finalROI.width, CV_8UC3, cv::Scalar(0));
			cv::Rect tmpROI = mvResultROI[i];
			tmpROI.x -= finalROI.x;
			tmpROI.y -= finalROI.y;
			mvResult[i].copyTo(tmp(tmpROI));
			mvResult[i] = tmp;
		}
	}

	void saveResults(std::string saveFolderName, bool bUserShow = true)
	{
		//Check and Create the folder
		if (_access(saveFolderName.c_str(), 0) == -1)
		{
			_mkdir(saveFolderName.c_str());
		}

		std::stringstream ioStr;
		for (size_t i = 0; i < mvResult.size(); i++)
		{
			ioStr.str("");

			ioStr << saveFolderName << "/result_" << std::setw(4) << std::setfill('0') << i << ".jpg";
			cv::imwrite(ioStr.str(), mvResult[i]);

			if (bUserShow)
			{
				resizeShow("test", mvResult[i], true);
				cv::waitKey(0);
			}
		}
	}

	

private:
	std::shared_ptr<WarperBase> mpWarper;
	int mfraction;
	bool mbOrder;

	//The vector to preserving the result
	//The normalizeROI function can be used to unify the result vector
	//to a common coordinate.
	std::vector<cv::Mat> mvResult;
	std::vector<cv::Rect> mvResultROI;
};


