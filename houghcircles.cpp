#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "houghcircles.h"

using namespace cv;
using namespace std;

void convolute(Mat input, Mat output, Mat kernel){
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

void sobel(Mat input, Mat sobelX, Mat sobelY, Mat sobelMag, Mat sobelDir, bool debug_mode){
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
    
    //approximation for the sobel magnitude
    addWeighted(sobelX, 0.5, sobelY, 0.5, 0, sobelMag);

    // calculate the direction of the gradient
    // TOOD: check convertion here
    sobelX.convertTo(sobelX, CV_32F);
    sobelX.convertTo(sobelY, CV_32F);
    phase(sobelX, sobelY, sobelDir);
    sobelDir.convertTo(sobelDir, CV_8UC1);

    if(debug_mode){
        cout << "[DEBUG SOBEL]: Type of sobelX: " << type2str(sobelX.type()) << endl;
        cout << "[DEBUG SOBEL]: Type of sobelY: " << type2str(sobelY.type()) << endl;
        cout << "[DEBUG SOBEL]: Type of sobelMag: " << type2str(sobelMag.type()) << endl;
        cout << "[DEBUG SOBEL]: Type of sobelDir: " << type2str(sobelDir.type()) << endl;
    }

    // save all images
    imwrite("workdir/sobelGradientMagnitude.jpg", sobelMag);
    imwrite("workdir/sobelX.jpg", sobelX);
    imwrite("workdir/sobelY.jpg", sobelY);
    imwrite("workdir/sobelGradientDirection.jpg", sobelDir);
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
    imwrite("workdir/threshold.jpg", output);
}


vector<Vec3f> hough(Mat grad_mag, Mat grad_orient, int threshold, Mat org, bool debug_mode){
    vector<Vec3f> circles;
    // Apply the Hough Transform to find the circles
    circles = houghCircleCalculation(grad_mag, 30, 80, debug_mode);

    cout << "[INFO]: Found " << circles.size() << " circles in the image!" << endl;
    return circles;
}

vector<Vec3f> houghCircleCalculation(Mat input, int minRadius, int maxRadius, bool debug_mode){
    vector<Vec3f> output;
    // reimplement this opencv inbuilt function
    //HoughCircles(input, output, CV_HOUGH_GRADIENT, 1, input.rows/8, 200, 100, 0, 0 );
    //return output;

    // some parameters to increase performance and other adjustments
    // IMPORTANT: DO NOT CHANGE ANY OF THESE PARAMS IF YOU DO NOT KNOW EXACTLY WHAT YOU ARE DOING THERE!!!!
    int x_step_size = 1;
    int y_step_size = 1;
    int theta_step_size = 1;
    int r_step_size = 1;

    int pixelThreshold = 200; // this is the threshold for drawing a circle around the pixel if the pixel is bright enough
    int t = 120; // this is the threshold for detecting a center of a cricle as a center!
        t = t/ (y_step_size * x_step_size * theta_step_size);

    cout << "[INFO HOUGH]: minRadius: " << minRadius << endl;
    cout << "[INFO HOUGH]: maxRadius: " << maxRadius << endl;
    cout << "[INFO HOUGH]: debug_mode: " << debug_mode << endl;
    cout << "[INFO HOUGH]: x_step_size: " << x_step_size << endl;
    cout << "[INFO HOUGH]: y_step_size: " << y_step_size << endl;
    cout << "[INFO HOUGH]: r_step_size: " << r_step_size << endl;
    cout << "[INFO HOUGH]: theta_step_size: " << theta_step_size << endl;
    cout << "[INFO HOUGH]: threshold for detecting circles: " << t << endl;
    
    if (debug_mode){
        cout << "[DEBUG HOUGHCIRCLE]: Checkpoint 01: inited params" << endl;
    }

    // init houghspace H
    int H[input.cols][input.rows];
    Mat houghspace = input.clone();
    for(int i = 0; i < input.cols; i++){
            for(int j = 0; j < input.rows; j++){
                houghspace.at<uchar>(j,i) = 0;
            }
        }



    if (debug_mode){
        cout << "[DEBUG HOUGHCIRCLE]: Checkpoint 02: inited hough space" << endl;
    }

    for(int r = minRadius; r < maxRadius; r=r+r_step_size){
        if (debug_mode){
        cout << "[DEBUG HOUGHCIRCLE]: --- Start new Houghspace calculation for Radius '" << r << "' ---" << endl;
        cout << "[DEBUG HOUGHCIRCLE]: Start resetting houghspace...";
        }
        // kind of a progress indicator
        double progress = (r - minRadius) * 100 / (maxRadius - minRadius);
        cout << "[INFO HOUGH]: Progress: " <<  progress << " \%" << endl;

        // reset hough space
        for(int i = 0; i < input.cols; i++){
            for(int j = 0; j < input.rows; j++){
                H[i][j] = 0;
            }
        }

        if(debug_mode){
            cout << "\tDone!" << endl;
            cout << "[DEBUG HOUGHCIRCLE]: Start calculating the houghspace...";
        }

        // calculate houghspace
        for(int y = 0; y < input.rows-y_step_size; y=y+y_step_size){
            for(int x = 0; x < input.cols-x_step_size; x=x+x_step_size){  
                uchar pixel = input.at<uchar>(y,x);
                if(pixel >= pixelThreshold){
                    // draw circle around the pixel
                    for(int theta = 0; theta < 360-theta_step_size; theta=theta+theta_step_size){
                        // calculate the polar coordinates for the center    
                        int a = x - r * cos(theta * CV_PI / 180);
                        // skip coordinates outside the image
                        if(a < 0 || a >= input.cols){
                            continue;
                        }
                        int b = y - r * sin(theta * CV_PI / 180);
                        if(b < 0 || b >= input.rows){
                            continue;
                        }
                        // increase voting
                        H[a][b] += 1;                 
                    }
                }
            }
        }

        if (debug_mode){
            cout << "\tDone!" << endl;
            cout << "[DEBUG HOUGHCIRCLE]: Start detecting the circles...";
        }
        int max = 0;
        // look if pixels are abough a certain threshold then they are centers of a circle
        for(int i = 0; i < input.cols; i++){
            for(int j = 0; j < input.rows; j++){
                if (debug_mode){
                    if (H[i][j] > max){
                        // calculate the maximum in the houghspace to set the threshold to a good value
                        max = H[i][j];
                    }
                }

                if (H[i][j] > t){
                    // circle detected
                    // add the x and y coordinates and the radius of the detected circle to the output vector
                    output.push_back(Vec3f(i,j,r));
                }
            }
        }

        if(debug_mode){
            cout << "\tDone!" << endl;
        }

        // merge the houghspace to a 2D image
        for(int i = 0; i < input.cols; i++){
            for(int j = 0; j < input.rows; j++){
                houghspace.at<uchar>(j,i) = houghspace.at<uchar>(j,i) + ( (H[i][j]*r_step_size*3)/(maxRadius-minRadius) );
            }
        }

        if (debug_mode){
            cout << "[DEBUG HOUGHCIRCLE]: Max value found in the houghspace was '" << max << "'" << endl;
        }
    }

    cout << "[INFO HOUGH]: Progress: " <<  100 << " \%" << endl;
    imwrite("workdir/houghspace.jpg", houghspace);

    if (debug_mode){
        cout << "[DEBUG HOUGHCIRCLE]: Circle detecting for all radius finished!" << endl;
    }

    return output;
}

