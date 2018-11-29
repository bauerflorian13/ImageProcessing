#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <opencv2/core.hpp>        //you may need to
#include <opencv2/highgui.hpp>   //adjust import locations
#include <opencv2/imgproc.hpp>    //depending on your machine setup
#include <opencv2/imgproc/imgproc.hpp>

void convolute(cv::Mat input_image, cv::Mat output_image, cv::Mat kernel);

void sobel(cv::Mat input, cv::Mat sobelX, cv::Mat sobelY, cv::Mat sobelMag, cv::Mat sobelDir);

void thresholdX(cv::Mat input, cv::Mat output, int T);

std::vector<cv::Vec3f> hough(cv::Mat grad_mag, cv::Mat grad_orient, int threshold, cv::Mat org);

std::vector<cv::Vec3f> houghCircleCalculation(cv::Mat input, int minDist, int minRadius, int maxRadius);

// just for debugging
std::string type2str(int type);
