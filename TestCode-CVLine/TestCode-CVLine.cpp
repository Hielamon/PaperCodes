#include <OpencvCommon.h>

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
	normTest = vec * vec.t();
	return 0;
}
