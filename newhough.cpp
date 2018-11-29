#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <opencv2/core.hpp>        //you may need to
#include <opencv2/highgui.hpp>   //adjust import locations
#include <opencv2/imgproc.hpp>    //depending on your machine setup
#include <opencv2/imgproc/imgproc.hpp>
#include "newhough.h"

using namespace cv;
using namespace std;


int main2() {
    cout << "Hello Circle Detector" << endl;

    // input image
    String input_filename = "dart13.jpg";
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

    cout << "Begin hough transformation..." << endl;
    hough(sobelMag, sobelDir, 100, image);
    cout << "Finished hough transformation!" << endl;

    sobelX.deallocate();
    sobelY.deallocate();
    sobelMag.deallocate();
    sobelDir.deallocate();
    thresholdSobelMag.deallocate();

    return 0;
}

void convolute(Mat input, Mat output, Mat kernel){
    // at the moment use the opencv convolution, maybe implement it later by yourself
     filter2D(input, output,-1, kernel);
}

// method for debugging
string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

void sobel(Mat input, Mat sobelX, Mat sobelY, Mat sobelMag, Mat sobelDir){
    // deriative in x direction
    Mat kernelX(3, 3, CV_32F);
    kernelX.at<float>(0,0) = 1.0f;
    kernelX.at<float>(0,1) = 0.0f;
    kernelX.at<float>(0,2) = -1.0f;
    kernelX.at<float>(1,0) = 2.0f;
    kernelX.at<float>(1,1) = 0.0f;
    kernelX.at<float>(1,2) = -2.0f;
    kernelX.at<float>(2,0) = 1.0f;
    kernelX.at<float>(2,1) = 0.0f;
    kernelX.at<float>(2,2) = -1.0f;

    convolute(input,sobelX,kernelX);

    // and in y direction
    Mat kernelY(3, 3, CV_32F);
    kernelY.at<float>(0,0) = 1.0f;
    kernelY.at<float>(0,1) = 2.0f;
    kernelY.at<float>(0,2) = 1.0f;
    kernelY.at<float>(1,0) = 0.0f;
    kernelY.at<float>(1,1) = 0.0f;
    kernelY.at<float>(1,2) = 0.0f;
    kernelY.at<float>(2,0) = -1.0f;
    kernelY.at<float>(2,1) = -2.0f;
    kernelY.at<float>(2,2) = -1.0f;

    convolute(input,sobelY,kernelY);

    for(int y = 0; y < input.rows; y++){
        for(int x = 0; x < input.cols; x++){
            float gx = abs(sobelX.at<float>(y,x));
            float gy = abs(sobelY.at<float>(y,x));
            float g = (gx + gy);
            sobelMag.at<float>(y,x) = (float) g;
        }
    }
    cout << "Type of sobelX: " << type2str(sobelX.type()) << endl;
    cout << "Type of sobelY: " << type2str(sobelY.type()) << endl;
    cout << "Type of sobelMag: " << type2str(sobelMag.type()) << endl;

    imwrite("sobelGradientMagnitude.jpg", sobelMag);
    // calculate the direction of the gradient
    // the orientation O = arctan(G_y / G_x)
    for(int y = 0; y < input.rows; y++){
        for(int x = 0; x < input.cols; x++){
            float gx = sobelX.at<float>(y,x);
            float gy = sobelY.at<float>(y,x);
            float orient = (float) atan(gy / gx);

            sobelDir.at<float>(y,x) = orient;
        }
    }

    // save all images
    imwrite("sobelX.jpg", sobelX);
    imwrite("sobelY.jpg", sobelY);
    imwrite("sobelGradientDirection.jpg", sobelDir);
}

void thresholdX(Mat input, Mat output, int T){
    // simple threshold calculation
    for(int y = 0; y < input.rows; y++){
        for(int x = 0; x < input.cols; x++){
            uchar pixel = input.at<uchar>(y,x);
            if(pixel >= T){
                output.at<uchar>(y,x) = 255;
            }else{
                output.at<uchar>(y,x) = 0;
            }
        }
    }

    // save the threshold image
    imwrite("threshold.jpg", output);
}


vector<Vec3f> hough(Mat grad_mag, Mat grad_orient, int threshold, Mat org){

    Mat src = org;
    vector<Vec3f> circles;
    // Apply the Hough Transform to find the circles
    circles = houghCircleCalculation( grad_mag, grad_mag.rows/8, 30, 80 );

    // Draw the circles detected
    /*for( size_t i = 0; i < circles.size(); i++ ) {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);

        cout << "Radius is: " << radius << endl;
        // circle center
        circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
        // circle outline
        circle( src, center, radius, Scalar(0,0,255), 3, 8, 0 );
    }*/
    // Show your results
    cout << "[INFO]: Found " << circles.size() << " circles in the image!" << endl;
    //namedWindow( "Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE );
    //imshow( "Hough Circle Transform Demo", src );
    imwrite("hough.jpg", src);
    waitKey(0);
    return circles;
}

vector<Vec3f> houghCircleCalculation(Mat input, int minDist, int minRadius, int maxRadius){
    vector<Vec3f> output;
    // reimplement this
    //HoughCircles(input, output, CV_HOUGH_GRADIENT, 1, input.rows/8, 200, 100, 0, 0 );
    //return output;

    // some parameters to increase performance and other adjustments
    // IMPORTANT: DO NOT CHANGE ANY OF THESE PARAMS IF YOU DO NOT KNOW EXACTLY WHAT YOU ARE DOING THERE!!!!
    int x_step_size = 1;
    int y_step_size = 1;
    int theta_step_size = 1;
    int r_step_size = 5;

    int t1 = 200;
    int t = 80; // this is the threshold for detecting a center of a cricle as a center!
        t = t/ (y_step_size * x_step_size * theta_step_size);
    int debug = 0;

    cout << "minDist: " << minDist << endl;
    cout << "minRadius: " << minRadius << endl;
    cout << "minRadius: " << maxRadius << endl;
    
    cout << "Checkpoint 01: inited params" << endl;

    // init houghspace H
    int H[input.cols][input.rows];
    Mat houghspace = input.clone();
    for(int i = 0; i < input.cols; i++){
            for(int j = 0; j < input.rows; j++){
                houghspace.at<uchar>(j,i) = 0;
            }
        }
    cout << "Checkpoint 02: inited hough space" << endl;

    for(int r = minRadius; r < maxRadius-r_step_size; r=r+r_step_size){
        cout << "[DEBUG]: --- Start new Houghspace calculation for Radius '" << r << "' ---" << endl;
        cout << "[DEBUG]: Start resetting houghspace...";
        // reset hough space
        for(int i = 0; i < input.cols; i++){
            for(int j = 0; j < input.rows; j++){
                H[i][j] = 0;
            }
        }
        cout << "\tDone!" << endl;

        cout << "[DEBUG]: Start calculating the houghspace...";
        // calculate houghspace
        for(int y = 0; y < input.rows-y_step_size; y=y+y_step_size){
            for(int x = 0; x < input.cols-x_step_size; x=x+x_step_size){  
                uchar pixel = input.at<uchar>(y,x);
                if(pixel >= t1){
                    for(int theta = 0; theta < 360-theta_step_size; theta=theta+theta_step_size){
                        // calculate the polar coordinates for the center    
                        int a = x - r * cos(theta * CV_PI / 180);
                        if(a < 0 || a >= input.cols){
                            continue;
                        }
                        int b = y - r * sin(theta * CV_PI / 180);
                        if(b < 0 || b >= input.rows){
                            continue;
                        }
                        if (debug){
                            cout << "x: " << x << endl;
                            cout << "y: " << y << endl;
                            cout << "Increment!" << endl;
                        }
                        // increase voting
                        H[a][b] += 1;
                        
                    }
                }
            }
        }
        cout << "\tDone!" << endl;

        cout << "[DEBUG]: Start detecting the circles...";
        int max = 0;
        // look if pixels are abough a certain threshold then they are centers of a circle
        for(int i = 0; i < input.cols; i++){
            for(int j = 0; j < input.rows; j++){
                if (H[i][j] > max){
                    max = H[i][j];
                }
                if (H[i][j] > t){
                    // circle detected
                    if(debug){
                        cout << "Circle detected!" << endl;
                    }
                    output.push_back(Vec3f(i,j,r));
                }
            }
        }
        cout << "\tDone!" << endl;
        // merge the houghspace to a 2D image
        for(int i = 0; i < input.cols; i++){
            for(int j = 0; j < input.rows; j++){
                houghspace.at<uchar>(j,i) = houghspace.at<uchar>(j,i) + ( (H[i][j]*r_step_size*3)/(maxRadius-minRadius) );
            }
        }
        cout << "[DEBUG]: Max value found in the houghspace was '" << max << "'" << endl;
    }

    imwrite("houghspace.jpg", houghspace);

    cout << "[DEBUG]: Circle detecting for all radius finished!" << endl;
    return output;
}

