#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void convolute(cv::Mat input_image, cv::Mat output_image, cv::Mat kernel);

void sobel(cv::Mat input, cv::Mat sobelX, cv::Mat sobelY, cv::Mat sobelMag, cv::Mat sobelDir, bool debug_mode);

void thresholdX(cv::Mat input, cv::Mat output, int T);

std::vector<cv::Vec3f> hough(cv::Mat grad_mag, cv::Mat grad_orient, int threshold, cv::Mat org, bool debug_mode);

std::vector<cv::Vec3f> houghCircleCalculation(cv::Mat input, int minRadius, int maxRadius, bool debug_mode);

// just for debugging
std::string type2str(int type);
