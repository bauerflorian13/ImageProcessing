// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <string>
#include "newhough.h"

using namespace std;
using namespace cv;

std::vector<cv::Rect> detectDartboards( Mat frame );

std::vector<cv::Rect> selectDartboards(std::vector<cv::Rect> dartboard, std::vector<cv::Vec3f> circles);

void drawDartboards(std::vector<cv::Rect> dartboards, cv::Mat image);

void drawDebugDartboards(vector<Rect> dartboards, vector<Vec3f>circles, Mat image);

String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;

int main(int argc, char** argv) {
    cout << "Hello Circle Detector" << endl;

    // input image
    String input_filename = argv[1];
    Mat image = imread(input_filename, 1);
    cout << "Loaded image '" << input_filename << "' as input file." << endl;

    // Convert to gray scale
    Mat gray_image;
    cvtColor(image, gray_image, CV_BGR2GRAY);
    imwrite("gray_img.jpg", gray_image);
    cout << "Converted image to gray scale image." << endl;

    // Sobel filter
    cout << "Begin Sobel filter calculation..." << endl;
    Mat sobelY = Mat(gray_image.size(), CV_8U);
    Mat sobelX = Mat(gray_image.size(), CV_8U);
    Mat sobelMag = Mat(gray_image.size(), CV_8U);
    Mat sobelDir = Mat(gray_image.size(), CV_8U);
    sobel(gray_image, sobelX, sobelY, sobelMag, sobelDir);
    cout << "Finished Sobel calculation!" << endl;

    cout << "Begin thresholding sobelMag image..." << endl;
    Mat thresholdSobelMag = Mat(gray_image.size(), CV_8U);
    thresholdX(thresholdSobelMag, thresholdSobelMag, 100);
    cout << "Finished thresholding sobelMag image!" << endl;

    // detect circles
    cout << "Begin hough transformation..." << endl;
    vector<Vec3f> circles;
    circles = hough(sobelMag, sobelDir, 100, image);
    cout << "Finished hough transformation!" << endl;

	// load the Strong Classifier in a structure called `Cascade'
    cout << "Start loading classifier structure..." << endl;
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    cout << "Finished loading classifier structure!" << endl;

    // detect dartboards
    cout << "Start dartboard detection...!" << endl;
	vector<Rect> dartboards;
    dartboards = detectDartboards(image);
    cout << "Finished dartboard detection!" << endl;

    // remove false dartboards
    cout << "Match detected dartboards with detected circles...!" << endl;
    vector<Rect> dartboards2;
    dartboards2 = dartboards;
    dartboards = selectDartboards(dartboards, circles);
    cout << "Finished dartboard selection!" << endl;

    // draw dartboards on the image
    cout << "Start drawing dartboards on the image...!" << endl;
    Mat image2 = image.clone();
    drawDartboards(dartboards, image);
    drawDebugDartboards(dartboards2, circles, image2);
    cout << "Finished drawing dartboards!" << endl;    

	// Save Result Image
	string prefix = "detected_";
	string filename = argv[1];
	imwrite( (prefix + filename), image);


	// Save Result Image
	prefix = "detected2_";
	filename = argv[1];
	imwrite( (prefix + filename), image2);

    // show the detected dartboards in the end
	imshow("Detected dartboards", image);
    waitKey(0);

    // just for debugging
	//imshow("Debugging dartboards", image2);
    //waitKey(0);

    return 0;
}

std::vector<Rect> detectDartboards( Mat frame )
{
	std::vector<Rect> dartboards;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, dartboards, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

    std::cout << "[INFO]: Found " << dartboards.size() << " dartboards in the image." << std::endl;

    return dartboards;
}


std::vector<cv::Rect> selectDartboards(std::vector<cv::Rect> dartboards, std::vector<cv::Vec3f> circles){
    vector<Rect> matchedDartboards;

    for(int i = 0; i < dartboards.size(); i++ )
	{
        /**
         *      c
         *   _ _ _
         *  |     |
         * a|  .  | b 
         *  |_ _ _|
         *      d
         * 
         **/
		bool detected = false;
        for(int j = 0; j < circles.size(); j++){
            bool a = dartboards[i].x < circles[j][0]; 
            bool b = (dartboards[i].x + dartboards[i].width) > circles[j][0];
            bool c = dartboards[i].y < circles[j][1];
            bool d = (dartboards[i].y + dartboards[i].height) > circles[j][1];
            cout << "--------------------" << endl;
            cout << "a: " << a << "| b: " << b << "| c: " << c << "| d: " << d << endl;
            cout << "dartboards[i].x: " << dartboards[i].x << " | dartboards[i].y: " << dartboards[i].y << endl;
            cout << "dartboards[i].width: " << dartboards[i].width << " | dartboards[i].height: " << dartboards[i].height << endl;
            cout << "circles[j][0]: " << circles[j][0] << " | circles[j][1]: " << circles[j][1] << endl;
         
            if (a && b && c && d){
                detected = true;
                //cout << "Found matching dartboard!" << endl;
            }
        }
        if (detected){
            cout << "Detected!" << endl;
            matchedDartboards.push_back(dartboards[i]);
        }
    }

    std::cout << "[INFO]: Found " << matchedDartboards.size() << " matching dartboards in the image!" << std::endl;

    return matchedDartboards;
}

void drawDartboards(vector<Rect> dartboards, Mat image){
	for(int i = 0; i < dartboards.size(); i++ )
	{
		rectangle(image, Point(dartboards[i].x, dartboards[i].y), Point(dartboards[i].x + dartboards[i].width, dartboards[i].y + dartboards[i].height), Scalar( 0, 255, 0 ), 2);
	}

}

void drawDebugDartboards(vector<Rect> dartboards, vector<Vec3f>circles, Mat image){
	for(int i = 0; i < dartboards.size(); i++ )
	{
		rectangle(image, Point(dartboards[i].x, dartboards[i].y), Point(dartboards[i].x + dartboards[i].width, dartboards[i].y + dartboards[i].height), Scalar( 0, 255, 0 ), 2);
	}

    for( size_t i = 0; i < circles.size(); i++ ) {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        // circle center
        circle( image, center, 3, Scalar(255,0,0), -1, 8, 0 );
        // circle outline
        circle( image, center, radius, Scalar(0,0,255), 2, 8, 0 );
    }

}