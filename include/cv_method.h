#pragma once

#include <iostream>
#include <string>
#include <vector>


#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include "ORBextractor.h"

cv::Mat imshow_parallel(const cv::Mat &image1, const cv::Mat &image2);

cv::Mat calculate_H(const cv::Mat &image1, const cv::Mat &image2 , double thres = 20);

cv::Mat picmerge(const cv::Mat &image1, const cv::Mat &image2, cv::Mat H ,cv::Mat &Trans);

