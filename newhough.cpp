#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <opencv2/core.hpp>        //you may need to
#include <opencv2/highgui.hpp>   //adjust import locations
#include <opencv2/imgproc.hpp>    //depending on your machine setup
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

void convolute(Mat input_image, Mat output_image, Mat kernel);

void sobel(Mat input, Mat sobelX, Mat sobelY, Mat sobelMag, Mat sobelDir);

void thresholdX(Mat input, Mat output, int T);

void hough(Mat grad_mag, Mat grad_orient, int threshold, Mat org);

// just for debugging
string type2str(int type);

int main() {
    cout << "Hello New Hough!\n";

    Mat image = imread("dart1.jpg", 1);

    // Convert to gray scale
    Mat gray_image;
    cvtColor(image, gray_image, CV_BGR2GRAY);
    imwrite("gray_img.jpg", gray_image);

    // Sobel filter
    Mat sobelX = Mat(gray_image.size(), CV_8U);//gray_image.clone();
    Mat sobelY = Mat(gray_image.size(), CV_8U);//gray_image.clone();
    Mat sobelMag = Mat(gray_image.size(), CV_8U);//gray_image.clone();
    Mat sobelDir = Mat(gray_image.size(), CV_8U);//gray_image.clone();
    sobel(gray_image, sobelX, sobelY, sobelMag, sobelDir);

    printf("Hello!\n");
    Mat thresholdSobelMag = Mat(gray_image.size(), CV_8U);//gray_image.clone();
    thresholdX(sobelMag, thresholdSobelMag, 100);
    printf("Bye!\n");

    hough(gray_image,sobelDir, 100, image);

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

            sobelDir.at<float>(y,x) = orient;// * 180/CV_PI;
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


void hough(Mat grad_mag, Mat grad_orient, int threshold, Mat org){

Mat src = org;//Mat(341,441, CV_8U, Scalar(255));

vector<Vec3f> circles;

/// Apply the Hough Transform to find the circles
HoughCircles( grad_mag, circles, CV_HOUGH_GRADIENT, 1,
grad_mag.rows/8, 200, 100, 0, 0 );
/// Draw the circles detected
for( size_t i = 0; i < circles.size(); i++ )
{
    Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
    int radius = cvRound(circles[i][2]);
    // circle center
    circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
    // circle outline
    circle( src, center, radius, Scalar(0,0,255), 3, 8, 0 );
 }
/// Show your results
namedWindow( "Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE );
imshow( "Hough Circle Transform Demo", src );
imwrite("hough.jpg", src);
waitKey(0);
}