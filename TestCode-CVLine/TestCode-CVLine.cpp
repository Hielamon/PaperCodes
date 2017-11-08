#include <OpencvCommon.h>
#define MAIN_FILE
#include <commonMacro.h>

int main(int argc, char *argv[])
{
	cv::Mat square(50, 50, CV_8UC1, cv::Scalar(0));
	cv::line(square, cv::Point(0, 0), cv::Point(square.cols - 1, square.rows - 1), cv::Scalar(100));
	cv::line(square, cv::Point(square.cols - 1, 0), cv::Point(0, square.rows - 1), cv::Scalar(200));

	cv::Rect roiTest(0, 0, 5, 5);
	cv::Mat roiMat = square(roiTest);


	double norm_[9] = { 1, 1, 2 };
	cv::Mat normTest(3, 3, CV_64FC1, norm_);
	std::cout << "L2 norm = " << cv::norm(normTest) << std::endl;

	cv::Vec3d vec_( 1, 1, 2 );
	
	cv::Mat vec(vec_);
	cv::Size vecSize = vec.size();
	cv::Size normTestSize = normTest.size();
	std::cout << "vec size = " << vec.size() << std::endl;

	cv::Mat testImg(100, 100, CV_8UC3, cv::Scalar(0));
	cv::rectangle(testImg, cv::Rect(0, 0, 50, 50), cv::Scalar(0, 255, 0, 0.4), -1);

	normTest = vec * vec.t();

	int64 iterNum1 = 1e8, iterNum2 = 1e4;
	std::vector<double> result(iterNum1);
	int testTimes = 2;
	for (size_t m = 0; m < testTimes; m++)
	{
		HL_INTERVAL_START;
		for (int64 i = 0, k = 0; i < iterNum2; i++)
		{
			for (int64 j = 0; j < iterNum2; j++, k++)
			{
				result[k] = sin(k)*k;
			}
		}
		
		HL_INTERVAL_END;

		HL_INTERVAL_START;
		for (int64 i = 0; i < iterNum1; i++)
		{
			result[i] = sin(i)*i;
		}
		HL_INTERVAL_END;
	}

	HL_CERR("Error Log Test");
	return 0;
}
