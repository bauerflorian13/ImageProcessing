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

vector<Vec2f> houghLinesCalculation(Mat input, int minDist, int angleStep, int maxRadius);

void findIntersections(vector<Vec2f> lines, Mat src);

vector<Vec3f> houghCircleCalculation(Mat input, int minDist, int minRadius, int maxRadius);

// just for debugging
string type2str(int type);

int main() {
    cout << "Hello Circle Detector" << endl;

    // input image
    String input_filename = "input_images/dart2.jpg";
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
    thresholdX(sobelMag, thresholdSobelMag, 100);
    cout << "Finished thresholding sobelMag image!" << endl;

    cout << "Begin hough transformation..." << endl;
    hough(thresholdSobelMag, sobelDir, 100, image);
    cout << "Finished hough transformation!" << endl;
    //
    // sobelX.deallocate();
    // sobelY.deallocate();
    // sobelMag.deallocate();
    // sobelDir.deallocate();
    // thresholdSobelMag.deallocate();

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

    Mat src = org;
    Mat cdist;

    vector<Vec2f> lines;
    // Apply the Hough Transform to find the circles
    lines = houghLinesCalculation( grad_mag, grad_mag.rows/8, 30, 80 );
    /*
    // Draw the circles detected
    for( size_t i = 0; i < .size(); i++ ) {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);

        cout << "Radius is: " << radius << endl;
        // circle center
        circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
        // circle outline
        circle( src, center, radius, Scalar(0,0,255), 3, 8, 0 );
    }
    // Show your results
    cout << "Found " << circles.size() << " circles in the image!" << endl;
    namedWindow( "Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE );
    imshow( "Hough Circle Transform Demo", src );
    */
    //
    // draw lines
    for( size_t i = 0; i < lines.size(); i++ )
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        line( org, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
    }

    // imshow("source", src);
    imshow("detected lines", org);

    // waitKey(0);

    imwrite("hough.jpg", src);
    cout << "Found " << lines.size() << " lines in the image!" << endl;
    findIntersections(lines, src);
    imshow("source", src);
    waitKey(0);

}

void findIntersections(vector<Vec2f> lines, Mat src){
    int rho1 = 0;
    int rho2 = 0;
    float theta1 = 0;
    float theta2 = 0;
    Rect rect(Point(), src.size());
    Point p(290,225);

    for( size_t i = 0; i < lines.size(); i++ )
    {
        float m1, dx1, dy1, c1;
        float intersection_X, intersection_Y;
        float rho1 = lines[i][0], theta1 = lines[i][1];
        Point pt11, pt12;
        double a1 = cos(theta1), b1 = sin(theta1);
        double x01 = a1*rho1, y01 = b1*rho1;
        pt11.x = cvRound(x01 + 1000*(-b1));
        pt11.y = -cvRound(y01 + 1000*(a1));
        pt12.x = cvRound(x01 - 1000*(-b1));
        pt12.y = -cvRound(y01 - 1000*(a1));
        // cout << "Points on first line:";
        // cout << "    " << pt11;
        // cout << "    " << pt12;

        dx1 = pt12.x - pt11.x;
        dy1 = pt12.y - pt11.y;

        m1 = dy1 / dx1;
        c1 = pt11.y - m1 * pt11.x;
        // cout << "m1 : " << m1 << endl;

        for( size_t j = 0; j < lines.size(); j++ )
        {
            float m2, dx2, dy2, c2;
            float rho2 = lines[j][0], theta2 = lines[j][1];
            Point pt21, pt22;
            double a2 = cos(theta2), b2 = sin(theta2);
            double x02 = a2*rho2, y02 = b2*rho2;
            pt21.x = cvRound(x02 + 1000*(-b2));
            pt21.y = -cvRound(y02 + 1000*(a2));
            pt22.x = cvRound(x02 - 1000*(-b2));
            pt22.y = -cvRound(y02 - 1000*(a2));
            // cout << "Points on second line:";
            // cout << "    " << pt21;
            // cout << "    " << pt22 << endl;

            dx2 = pt22.x - pt21.x;
            dy2 = pt22.y - pt21.y;

            m2 = dy2 / dx2;
            c2 = pt21.y - m2 * pt21.x;

            // cout << "m2 : " << m2 << endl;

            if( (m1 - m2) == 0)
              std::cout << "No Intersection between the lines\n";
            else
            {
                intersection_X = (c2 - c1) / (m1 - m2);
                intersection_Y = m1 * intersection_X + c1;
                std::cout << "Intersecting Point: = ";
                std::cout << intersection_X;
                std::cout << ",";
                std::cout << intersection_Y;
                std::cout << "\n";
                Point p(intersection_X, -intersection_Y);
                circle( src, p, 2, Scalar(255,0,0), -1, 8, 0);
            }

        }
    }










    // for( size_t i = 0; i < lines.size(); i++ )
    // {
    //     float rho1 = lines[i][0], theta1 = lines[i][1];
    //     Point pt11, pt12;
    //     double a1 = cos(theta2), b1 = sin(theta1);
    //     double x01 = a1*rho1, y01 = b1*rho1;
    //     pt11.x = cvRound(x01 + 1000*(-b1));
    //     pt11.y = cvRound(y01 + 1000*(a1));
    //     pt12.x = cvRound(x01 - 1000*(-b1));
    //     pt12.y = cvRound(y01 - 1000*(a1));
    //     cout << pt11 << endl;
    //     cout << pt12 << endl;
    //     // line( org, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
    //     for (size_t j=0; j < lines.size(); j++){
    //       float rho2 = lines[j][0], theta2 = lines[j][1];
    //       Point pt21, pt22;
    //       double a2 = cos(theta2), b2 = sin(theta2);
    //       double x02 = a2*rho2, y02 = b2*rho2;
    //       pt21.x = cvRound(x02 + 1000*(-b2));
    //       pt21.y = cvRound(y02 + 1000*(a2));
    //       pt22.x = cvRound(x02 - 1000*(-b2));
    //       pt22.y = cvRound(y02 - 1000*(a2));
    //       // cout << pt21 << endl;
    //       // cout << pt22 << endl;
    //
    //       cout << endl;
    //     }
    // }

/*
    for(int i = 0; i<lines.size(); i++){

    //     if (lines[i][1]*(180/CV_PI)>100) {
    //     cout << lines[i][0] << endl;
    //     cout << lines[i][1]*(180/CV_PI) << " degrees" << endl;
    // }
    //     if (lines[i][1]*(180/CV_PI)<60) {
    //     cout << lines[i][0] << endl;
    //     cout << lines[i][1]*(180/CV_PI) << " degrees" << endl;
    // }

        for(int j = 0; j < lines.size(); j++){
            // cout << "It " << i << " " << j << endl;

            // Mat line1 = lines[i];
            // Mat line2 = lines[j];
            double line1angle = double(lines[i][1]);
            double line2angle = double(lines[j][1]);
            double line1yintercept = double(-lines[i][0]);
            double line2yintercept = double(-lines[j][0]);

            double line1vector[2] = {sin(line1angle), cos(line1angle)};
            double line2vector[2] = {sin(line2angle), cos(line2angle)};

            Point2f line1p1 (0, line1yintercept);
            Point2f line1p2 (line1vector[0], line1yintercept+(line1vector[1]));
            Point2f line2p1 (0, line2yintercept);
            Point2f line2p2 (line2vector[0], line2yintercept+(line2vector[1]));
             cout << "  "<< line1p2 << endl;
            // cout << "  "<< line1p2 << endl;
            // cout << line1angle << endl;
            // cout << line1angle*180/CV_PI << endl;
            //
            // cout << line1vector[0] << " " << line1vector[1] << endl;
            //
            // cout << endl;


            Point2f x = line2p1 - line1p1;
            Point2f d1 = line1p2 - line1p1;
            Point2f d2 = line2p2 - line2p1;

            float cross = d1.x * d2.y - d1.y * d2.x;
            if (abs(cross) <(1e-8)) continue;

            double t1 = (x.x * d2.y - x.y * d2.x)/cross;
            Point2f r = line1p1 + d1 * t1;

            Point p(r.x, -r.y);

            // if (rect.contains(p) && src.at<uchar>(r.y, r.x) ==0){
                // r.x = cvRound(r.x);
                // r.y = cvRound(r.y);
                // if ((r.x<350) & (r.x>250)){
                // cout << r << endl;

                circle( src, p, 1, Scalar(255,0,0), -1, 8, 0);

                // cout << line1p1 << endl;
                // cout << line1angle << endl;
                // cout << line1p2 << endl;

            // }




            // rho1 = lines[i][0];
            // theta1 = lines[i][1];
            // rho2 = lines[j][0];
            // theta2 = lines[j][1];
            // double matrix = {{cos(theta1), sin(theta1)},{cos(theta2), sin(theta2)}};
            // double vector = {rho1, rho2};
            // cout << point << endl;

        }

    }
    */
}

vector<Vec2f> houghLinesCalculation(Mat input, int minDist, int minRadius, int maxRadius){
    vector<Vec2f> lines;
    // detect lines
    HoughLines(input, lines, 1, CV_PI/180, 150, 0, 0 );

    return lines;
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
    int r_step_size = 1;

    int t1 = 200;
    int r = 53;
    int t = 90; // this is the threshold for detecting a center of a cricle as a center!
        t = t/ (y_step_size * x_step_size * theta_step_size);
    int debug = 0;

    cout << "minDist: " << minDist << endl;
    cout << "minRadius: " << minRadius << endl;
    cout << "minRadius: " << maxRadius << endl;

    cout << "Checkpoint 01: inited params" << endl;

    // init houghspace H
    int H[input.cols][input.rows];
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
                    cout << "Circle detected!" << endl;
                    output.push_back(Vec3f(i,j,r));
                }
            }
        }
        cout << "\tDone!" << endl;
        cout << "[DEBUG]: Max value found in the houghspace was '" << max << "'" << endl;
    }

    cout << "[DEBUG]: Circle detecting for all radius finished!" << endl;
    return output;
}
