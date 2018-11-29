// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <string>
#include "houghcircles.h"
#include <algorithm>

using namespace std;
using namespace cv;

std::vector<cv::Rect> detectDartboards(Mat frame);

std::vector<cv::Rect> selectDartboards(std::vector<cv::Rect> dartboards, std::vector<cv::Vec3f> circles);

std::vector<cv::Rect> mergeDartboards(std::vector<cv::Rect> dartboards);

void drawDartboards(std::vector<cv::Rect> dartboards, cv::Mat image);

void drawDebugDartboards(vector<Rect> dartboards, vector<Vec3f>circles, Mat image);

String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;

int main(int argc, char** argv) {
    cout << "Hello Circle Detector" << endl;

    // input image
    String input_filename = argv[1];
    input_filename = "input_images/" + input_filename;
    Mat image = imread(input_filename, 1);
    cout << "Loaded image '" << input_filename << "' from input_images directory as input file." << endl;

    // Convert to gray scale
    Mat gray_image;
    cvtColor(image, gray_image, CV_BGR2GRAY);
    imwrite("workdir/gray_img.jpg", gray_image);
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
    thresholdX(sobelMag, thresholdSobelMag, 100);
    cout << "Finished thresholding sobelMag image!" << endl;

    // detect circles
    cout << "Begin hough transformation..." << endl;
    vector<Vec3f> circles;
    circles = hough(thresholdSobelMag, sobelDir, 100, image);
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

    // merge overlapping dartboards together
    dartboards =  mergeDartboards(dartboards);
    
    // draw dartboards on the image
    cout << "Start drawing dartboards on the image...!" << endl;
    Mat image2 = image.clone();
    drawDartboards(dartboards, image);
    drawDebugDartboards(dartboards2, circles, image2);
    cout << "Finished drawing dartboards!" << endl;    

	// Save Result Image
	string prefix = "output_images/detected_";
	string filename = argv[1];
	imwrite( (prefix + filename), image);


	// Save Result Image
	prefix = "output_images/detected2_";
	imwrite( (prefix + filename), image2);

    // show the detected dartboards in the end
	imshow("Detected dartboards", image);
    waitKey(0);

    // just for debugging
	imshow("Debugging dartboards", image2);
    waitKey(0);

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

bool rectContainsPoint(Rect db, Point p, bool debug){
    bool a1 = db.x < p.x; 
    bool a2 = (db.x + db.width) > p.x;
    bool a3 = db.y < p.y;
    bool a4 = (db.y + db.height) > p.y;
    bool a = a1 && a2 && a3 && a4;
    if (debug){
    cout << "--------------------" << endl;
    cout << "a1: " << a1 << "| a2: " << a2 << "| a3: " << a3 << "| a4: " << a4 << endl;
    cout << "dartboards[i].x: " << db.x << " | dartboards[i].y: " << db.y << endl;
    cout << "dartboards[i].width: " << db.width << " | dartboards[i].height: " << db.height << endl;
    cout << "circles[j][0]: " << p.x << " | circles[j][1]: " << p.y << endl;
    }
    return a;
}

std::vector<cv::Rect> selectDartboards(std::vector<cv::Rect> dartboards, std::vector<cv::Vec3f> circles){
    vector<Rect> matchedDartboards;
    bool debug = 0;
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
            bool a = rectContainsPoint(dartboards[i], Point(circles[j][0], circles[j][1]), false);
            bool b = rectContainsPoint(dartboards[i], Point(circles[j][0]-circles[j][2], circles[j][1]), false);
            bool c = rectContainsPoint(dartboards[i], Point(circles[j][0]+circles[j][2], circles[j][1]), false);
            bool d = rectContainsPoint(dartboards[i], Point(circles[j][0], circles[j][1]-circles[j][2]), false);
            bool e = rectContainsPoint(dartboards[i], Point(circles[j][0], circles[j][1]+circles[j][2]), false);
            if (a && b && c && d && e){
                detected = true;
                break;
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

//TODO: implement this or fix the code below
std::vector<cv::Rect> mergeDartboards(std::vector<cv::Rect> dartboards){
    /*vector<Rect> newDartboards;
    vector<Rect> mergedDartboards;
    bool anychange = false;
    for(int i = 0; i < dartboards.size(); i++ )
	{
		for(int k = 0; k < dartboards.size(); k++ )
	    {
            if (i == k){
                continue;
            }
            bool acontb = dartboards[i].contains(Point(dartboards[k].x, dartboards[k].y));
            //bool bconta = dartboards[k].contains(Point(dartboards[i].x, dartboards[i].y));
            if (acontb){
                Rect r;
                r = Rect((int) dartboards[i].x, (int) dartboards[i].y, 
                (int) dartboards[k].x + dartboards[k].width - dartboards[i].x, 
                (int) dartboards[k].y + dartboards[k].height - dartboards[i].y);

                newDartboards.push_back(r);
                // cout << "HELLLLOOO!" << endl;
                anychange = true;
                mergedDartboards.push_back(dartboards[i]);
                mergedDartboards.push_back(dartboards[k]);
            }
        }
    }

    for(int i = 0; i < dartboards.size(); i++){
        if(!count(mergedDartboards.begin(), mergedDartboards.end(), dartboards[i])){
            newDartboards.push_back(dartboards[i]);
        }
    }
    if (anychange){
        cout << newDartboards.size() << endl;
        return mergeDartboards(newDartboards);
    }*/
    return dartboards;
    //return newDartboards;
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