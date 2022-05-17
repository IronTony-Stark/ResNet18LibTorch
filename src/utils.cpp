#include "utils.hpp"

#include <fstream>

void centerCrop(const cv::Mat& src, cv::Mat& dst, const cv::Size& size)
{
	cv::Rect roi;
	roi.x = src.size().width / 2 - size.width / 2;
	roi.width = size.width;
	roi.y = src.size().height / 2 - size.height / 2;
	roi.height = size.height;

	cv::Mat croppedRef(src, roi);
	croppedRef.copyTo(dst);
}

std::vector<std::string> readClasses(const char* filepath)
{
	std::vector<std::string> categories;

	std::ifstream file(filepath);
	std::string str;
	while (std::getline(file, str))
		categories.push_back(str);

	return categories;
}
