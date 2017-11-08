#include "SequenceMatcher.h"
#include <Monitor.h>

SequenceMatcher::SequenceMatcher(const FeatureType &featuretype, float scale)
{
	m_featuretype = featuretype;
	m_scale = scale;
	switch (m_featuretype)
	{
	case F_ORB:	m_featurator = cv::ORB::create(2000);
		m_descriptorMatcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
		break;
	case F_SIFT:m_featurator = cv::xfeatures2d::SIFT::create();
		m_descriptorMatcher = new cv::FlannBasedMatcher;
		break;
	case F_SURF:m_featurator = cv::xfeatures2d::SURF::create();
		m_descriptorMatcher = new cv::FlannBasedMatcher;
		break;
	default:
		break;
	}
}

SequenceMatcher::~SequenceMatcher() {}

//ʹ������ƥ��ķ����ҵ�һ�����ʵ�ͼƬƥ�����У����������ƥ�����е���Ϣ
void SequenceMatcher::process(const std::vector<cv::Mat>&images, std::list<PairInfo> &pairinfos)
{
	int num_images = images.size();
	if (num_images < 2)return;

	m_keypoints_arr.resize(num_images);
	m_descriptors_arr.resize(num_images);

	_getFeatures(images, m_keypoints_arr, m_descriptors_arr);
	int height = images[0].rows;
	int q_height = 1.0 * height * m_scale;

	if (!pairinfos.empty())pairinfos.clear();
	if (!m_all_pairinfos.empty())m_all_pairinfos.clear();
	m_all_pairinfos.resize(num_images - 1);
	std::vector<bool> is_finds(num_images, false);
	int left_index = -1, right_index = -1, max_num = 0;
	for (size_t i = 0; i < num_images - 1; i++)
	{
		for (size_t j = i + 1; j < num_images; j++)
		{
			std::vector<cv::DMatch> matches;
			_getCorresponds(m_descriptors_arr[i], m_descriptors_arr[j], matches, 0.7);
			PairInfo pair_temp;
			pair_temp.pairs_num = matches.size();
			pair_temp.inliers_num = pair_temp.pairs_num;
			pair_temp.mask.resize(matches.size(), 1);
			pair_temp.index1 = i;
			pair_temp.index2 = j;
			std::vector<cv::KeyPoint> &keypoints1 = m_keypoints_arr[i];
			std::vector<cv::KeyPoint> &keypoints2 = m_keypoints_arr[j];
			for (size_t i = 0; i < matches.size(); i++)
			{
				cv::Point2d point1 = keypoints1[matches[i].queryIdx].pt;
				cv::Point2d point2 = keypoints2[matches[i].trainIdx].pt;
				if (point1.y > q_height || point2.y > q_height)continue;
				(pair_temp.points1).push_back(point1);
				(pair_temp.points2).push_back(point2);
			}

			m_all_pairinfos[i].push_back(pair_temp);

			if (pair_temp.pairs_num > max_num)
			{
				left_index = i;
				right_index = j;
				max_num = pair_temp.pairs_num;
			}
		}
	}

	pairinfos.push_back(m_all_pairinfos[left_index][right_index - left_index - 1]);
	is_finds[left_index] = is_finds[right_index] = true;

	while (1)
	{
		int left_max_num = 0, left_index_temp = -1;
		for (size_t i = 0; i < num_images; i++)
		{
			if (is_finds[i])continue;
			assert(i != left_index);
			int num_i_left;
			if (i > left_index)
				num_i_left = m_all_pairinfos[left_index][i - 1 - left_index].pairs_num;
			else
				num_i_left = m_all_pairinfos[i][left_index - 1 - i].pairs_num;
			if (num_i_left > left_max_num)
			{
				left_max_num = num_i_left;
				left_index_temp = i;
			}
		}

		int right_max_num = 0, right_index_temp = -1;
		for (size_t i = 0; i < num_images; i++)
		{
			if (is_finds[i])continue;
			assert(i != right_index);
			int num_i_right;
			if (i > right_index)
				num_i_right = m_all_pairinfos[right_index][i - 1 - right_index].pairs_num;
			else
				num_i_right = m_all_pairinfos[i][right_index - 1 - i].pairs_num;
			if (num_i_right > right_max_num)
			{
				right_max_num = num_i_right;
				right_index_temp = i;
			}
		}

		if (left_index_temp == -1 && right_index_temp == -1)break;
		if (left_max_num > right_max_num)
		{
			if (left_index_temp > left_index)
			{
				PairInfo pair_temp = m_all_pairinfos[left_index][left_index_temp - left_index - 1];
				int temp = pair_temp.index1;
				pair_temp.index1 = pair_temp.index2;
				pair_temp.index2 = temp;
				std::vector<cv::Point2f> temp_pps = pair_temp.points1;
				pair_temp.points1 = pair_temp.points2;
				pair_temp.points2 = temp_pps;
				pairinfos.push_front(pair_temp);
			}
			else pairinfos.push_front(m_all_pairinfos[left_index_temp][left_index - 1 - left_index_temp]);

			is_finds[left_index_temp] = true;
			left_index = left_index_temp;
		}
		else
		{
			if (right_index_temp < right_index)
			{
				PairInfo pair_temp = m_all_pairinfos[right_index_temp][right_index - 1 - right_index_temp];
				int temp = pair_temp.index1;
				pair_temp.index1 = pair_temp.index2;
				pair_temp.index2 = temp;
				std::vector<cv::Point2f> temp_pps = pair_temp.points1;
				pair_temp.points1 = pair_temp.points2;
				pair_temp.points2 = temp_pps;
				pairinfos.push_back(pair_temp);
			}
			else pairinfos.push_back(m_all_pairinfos[right_index][right_index_temp - 1 - right_index]);

			is_finds[right_index_temp] = true;
			right_index = right_index_temp;
		}
	}

}

//ʹ������ƥ��ķ����ҵ�һ�����ʵ�ͼƬƥ�����У����������ƥ�����е���Ϣ
void SequenceMatcher::processTest(const std::vector<cv::Mat>&images, std::list<PairInfo> &pairinfos)
{
	int num_images = images.size();
	if (num_images < 2)return;

	m_keypoints_arr.resize(num_images);
	m_descriptors_arr.resize(num_images);

	_getFeatures(images, m_keypoints_arr, m_descriptors_arr);
	int height = images[0].rows;

	if (!pairinfos.empty())pairinfos.clear();
	for (size_t i = 0; i < num_images - 1; i++)
	{
		size_t j = i + 1;
		std::vector<cv::DMatch> matches;
		_getCorresponds(m_descriptors_arr[i], m_descriptors_arr[j], matches, 0.8);
		PairInfo pair_temp;
		pair_temp.pairs_num = matches.size();
		pair_temp.inliers_num = pair_temp.pairs_num;
		pair_temp.mask.resize(matches.size(), 1);
		pair_temp.index1 = i;
		pair_temp.index2 = j;
		std::vector<cv::KeyPoint> &keypoints1 = m_keypoints_arr[i];
		std::vector<cv::KeyPoint> &keypoints2 = m_keypoints_arr[j];
		for (size_t i = 0; i < matches.size(); i++)
		{
			cv::Point2d point1 = keypoints1[matches[i].queryIdx].pt;
			cv::Point2d point2 = keypoints2[matches[i].trainIdx].pt;
			(pair_temp.points1).push_back(point1);
			(pair_temp.points2).push_back(point2);
		}
		pairinfos.push_back(pair_temp);
	}

}


//��pairinfosƥ�����еĵ�һ��ͼ�����һ��ͼ��ƥ����Ϣ��push_back ��pairinfos��
void SequenceMatcher::getRingPair(std::list<PairInfo> &pairinfos)
{
	assert(pairinfos.size() >= 2 || !m_all_pairinfos.empty());
	int first_index = pairinfos.begin()->index1;
	int second_index = first_index;
	//std::list<PairInfo>::iterator result_iter;

	for (std::list<PairInfo>::iterator iter = pairinfos.begin(); iter != pairinfos.end(); iter++)
	{
		second_index = iter->index2;
		//result_iter = iter;
	}


	assert(first_index != second_index);
	assert(m_all_pairinfos.size() >= first_index && m_all_pairinfos.size() >= second_index);

	if (first_index < second_index)
		pairinfos.push_back(m_all_pairinfos[first_index][second_index - first_index - 1]);
	else
	{
		PairInfo pair_temp = m_all_pairinfos[second_index][first_index - 1 - second_index];
		int temp = pair_temp.index1;
		pair_temp.index1 = pair_temp.index2;
		pair_temp.index2 = temp;
		std::vector<cv::Point2f> temp_pps = pair_temp.points1;
		pair_temp.points1 = pair_temp.points2;
		pair_temp.points2 = temp_pps;
		pairinfos.push_back(pair_temp);
	}

	//return result_iter;
}

//��pairinfosƥ�����еĵ�һ��ͼ�����һ��ͼ��ƥ����Ϣ��push_back ��pairinfos��
void SequenceMatcher::getRingPairTest(std::list<PairInfo> &pairinfos)
{
	assert(pairinfos.size() >= 2);

	{
		size_t i = 0, j = m_descriptors_arr.size() - 1;
		std::vector<cv::DMatch> matches;
		_getCorresponds(m_descriptors_arr[i], m_descriptors_arr[j], matches, 0.8);
		PairInfo pair_temp;
		pair_temp.pairs_num = matches.size();
		pair_temp.inliers_num = pair_temp.pairs_num;
		pair_temp.mask.resize(matches.size(), 1);
		pair_temp.index1 = i;
		pair_temp.index2 = j;
		std::vector<cv::KeyPoint> &keypoints1 = m_keypoints_arr[i];
		std::vector<cv::KeyPoint> &keypoints2 = m_keypoints_arr[j];
		for (size_t i = 0; i < matches.size(); i++)
		{
			cv::Point2d point1 = keypoints1[matches[i].queryIdx].pt;
			cv::Point2d point2 = keypoints2[matches[i].trainIdx].pt;
			(pair_temp.points1).push_back(point1);
			(pair_temp.points2).push_back(point2);
		}
		pairinfos.push_back(pair_temp);
	}

	//return result_iter;
}

//��ȡimages��ÿ��ͼƬ���������Լ���������
void SequenceMatcher::_getFeatures(const std::vector<cv::Mat>&images, std::vector<std::vector<cv::KeyPoint>> &keypoints_arr, std::vector<cv::Mat> &descriptors_arr)
{
	for (size_t i = 0; i < images.size(); i++)
	{
		m_featurator->detectAndCompute(images[i], cv::Mat(), keypoints_arr[i], descriptors_arr[i]);
		for (size_t j = 0; j < keypoints_arr[i].size(); j++)
		{
			keypoints_arr[i][j].pt *= m_scale;
		}
	}
}

//�ҵ�des1 �� descriptors2�е�����ڶ�Ӧ�� des2��������des2 �� descriptors1�е�����ڶ�Ӧ��Ϊdes1
bool SequenceMatcher::_getCorresponds(const cv::Mat &descriptors1, const cv::Mat &descriptors2, std::vector<cv::DMatch>& matches, double ratio)
{

	if (!matches.empty())matches.clear();
	if (m_featuretype == FeatureType::F_ORB)
	{
		std::vector<cv::DMatch> matches_1t2, matches_2t1;

		m_descriptorMatcher->match(descriptors1, descriptors2, matches_1t2);
		m_descriptorMatcher->match(descriptors2, descriptors1, matches_2t1);
		for (size_t i = 0; i < matches_1t2.size(); i++)
		{
			if (matches_2t1[matches_1t2[i].trainIdx].trainIdx == i)
			{
				assert(matches_1t2[i].queryIdx == i);
				matches.push_back(matches_1t2[i]);
			}
		}
	}
	else
	{
		assert(m_featuretype == FeatureType::F_SIFT || m_featuretype == FeatureType::F_SURF);
		std::vector<std::vector<cv::DMatch>> two_matches_1t2, two_matches_2t1;
		m_descriptorMatcher->knnMatch(descriptors1, descriptors2, two_matches_1t2, 2);
		m_descriptorMatcher->knnMatch(descriptors2, descriptors1, two_matches_2t1, 2);

		//float ratio = 0.50;

		for (size_t i = 0; i < two_matches_1t2.size(); i++)
		{
			std::vector<cv::DMatch> &match_tmp = two_matches_1t2[i];
			if (match_tmp.size() < 2 || match_tmp[0].distance / match_tmp[1].distance > ratio)match_tmp.clear();
		}

		for (size_t i = 0; i < two_matches_2t1.size(); i++)
		{
			std::vector<cv::DMatch> &match_tmp = two_matches_2t1[i];
			if (match_tmp.size() < 2 || match_tmp[0].distance / match_tmp[1].distance > ratio)match_tmp.clear();
		}

		for (size_t i = 0; i < two_matches_1t2.size(); i++)
		{
			if (two_matches_1t2[i].size() == 2 && two_matches_2t1[two_matches_1t2[i][0].trainIdx].size() == 2
				&& two_matches_2t1[two_matches_1t2[i][0].trainIdx][0].trainIdx == i)
			{
				assert(two_matches_1t2[i][0].queryIdx == i);
				matches.push_back(two_matches_1t2[i][0]);
			}
		}
	}

	if (matches.size() < 4)return false;
	return true;
}

//�õ��ڵ����
float SequenceMatcher::_getInlierRate(const std::vector<char>& inliers)
{
	int n = 0;
	for (size_t i = 0; i < inliers.size(); i++)
	{
		if (inliers[i] != 0)n++;
	}
	return n / (float)inliers.size();
}

void DrawPairInfos(std::vector<cv::Mat> &images, std::list<PairInfo> &pairinfos, bool onlyPoints, double scale)
{
	size_t pair_num = pairinfos.size();
	std::string name = "pairs_";
	std::stringstream ss;
	int index = 0;

	for (std::list<PairInfo>::iterator iter = pairinfos.begin(); iter != pairinfos.end(); iter++)
	{
		cv::Mat result;
		cv::Mat left, right;
		images[iter->index1].copyTo(left);
		images[iter->index2].copyTo(right);
		cv::hconcat(left, right, result);

		for (size_t i = 0; i < iter->pairs_num; i++)
		{
			if (iter->mask[i] != 1)continue;
			uchar r = rand() % 255;
			uchar g = rand() % 255;
			uchar b = rand() % 255;
			cv::Scalar color(b, g, r);
			cv::circle(result, iter->points1[i] * scale, 6, color, -1);
			cv::Point2d pt2 = iter->points2[i] * scale;
			pt2.x += left.cols;
			if(!onlyPoints)cv::line(result, iter->points1[i] * scale, pt2, color, 3);
			cv::circle(result, pt2, 6, color, -1);
			//cv::imshow("showtemp", result);
			//cv::waitKey(0);
		}
		ss << index;
		cv::imwrite(name + ss.str() + ".jpg", result);
		ss.str("");
		index++;
	}
}

void DrawPairInfoHomo(const std::vector<cv::Mat> &images, const PairInfo &pairinfos, const cv::Mat &H)
{
	int index0 = pairinfos.index1, index1 = pairinfos.index2;
	cv::Size imgSize0(images[index0].size()), imgSize1(images[index1].size());
	cv::Size resultSize(imgSize0.width + imgSize1.width, std::max(imgSize0.height, imgSize1.height));

	double miniScale = FitSizeToScreen(resultSize.width, resultSize.height);
	cv::Size miniSize0(imgSize0.width*miniScale, imgSize0.height*miniScale);
	cv::Size miniSize1(imgSize1.width*miniScale, imgSize1.height*miniScale);
	cv::Mat resizedImg0, resizedImg1, showImg;
	cv::resize(images[pairinfos.index1], resizedImg0, miniSize0);
	cv::resize(images[pairinfos.index2], resizedImg1, miniSize1);
	cv::hconcat(resizedImg0, resizedImg1, showImg);
	cv::Mat tempShow = showImg.clone();
	for (size_t i = 0; i < pairinfos.pairs_num; i++)
	{
		if (pairinfos.mask[i] != 1)continue;
		uchar r = rand() % 255;
		uchar g = rand() % 255;
		uchar b = rand() % 255;
		cv::Scalar color(b, g, r);

		cv::circle(showImg, pairinfos.points1[i] * miniScale, 6, color, -1);
		cv::Point2f pt2 = pairinfos.points2[i] * miniScale;
		pt2.x += resizedImg0.cols;
		cv::line(showImg, pairinfos.points1[i] * miniScale, pt2, color, 2);
		cv::circle(showImg, pt2, 3, color, -1);
		cv::Point2f tempPt;
		PointHTransform(pairinfos.points1[i], H, tempPt);
		tempPt *= miniScale;
		tempPt.x += resizedImg0.cols;
		cv::line(showImg, pt2, tempPt, cv::Scalar(0, 0, 255), 1);
		cv::Rect rect(tempPt - cv::Point2f(3, 3), tempPt + cv::Point2f(3, 3));
		cv::rectangle(showImg, rect, cv::Scalar(0, 0, 255), 1);

		cv::imshow("showImg", showImg);
		showImg = tempShow.clone();
		cv::waitKey(0);
	}
}
