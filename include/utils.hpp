//	+------------------------------------------------------------------------
//	|	Copyright 2021 by Avid Technology, Inc.
//	+------------------------------------------------------------------------
#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

void centerCrop(const cv::Mat& src, cv::Mat& dst, const cv::Size& size);
std::vector<std::string> readClasses(const char* filepath);

#endif // __UTILS_HPP__
