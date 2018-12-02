/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp" //done
#include "opencv2/opencv.hpp" //done
#include "opencv2/core.hpp" // done
#include "opencv2/highgui.hpp" // done
#include "opencv2/imgproc/imgproc.hpp" //done
#include <iostream>
#include <stdio.h>
#include <string>

using namespace std;
using namespace cv;

/** Function Headers */
int detectAndDisplay( Mat frame );

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;

// number of faces in 5 specific images
int number_of_face_dart4 = 1;
int number_of_face_dart5 = 11;
int number_of_face_dart13 = 1;
int number_of_face_dart14 = 2;
int number_of_face_dart15 = 3;

/** @function main */
int main( int argc, const char** argv )
{
  // 1. Read Input Image
	string filename = argv[1];
	Mat frame = imread("input_images/" + filename, CV_LOAD_IMAGE_COLOR);

	// Number of faces expected
	int num_of_expected_faces = 0;

	if (filename == "dart4.jpg"){
		num_of_expected_faces = number_of_face_dart4;
	}

	if (filename == "dart5.jpg"){
		num_of_expected_faces = number_of_face_dart5;
	}

	if (filename == "dart13.jpg"){
		num_of_expected_faces = number_of_face_dart13;
	}

	if (filename == "dart14.jpg"){
		num_of_expected_faces = number_of_face_dart14;
	}

	if (filename == "dart15.jpg"){
		num_of_expected_faces = number_of_face_dart15;
	}
	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	int num_of_detected_faces = detectAndDisplay( frame );

	// 4. Save Result Image
	string prefix = "output_images/detected_face_";
	imwrite( (prefix + filename) , frame );

	// formula for calculation F1 score: 2 * (precision * recall) / (precision + recall)
	double true_positives = num_of_expected_faces;
	double false_positives = num_of_detected_faces - num_of_expected_faces;
	double precision = true_positives / (true_positives + false_positives);
	double false_negatives = 0; // this is just an assumption
	double recall = true_positives / (true_positives + false_negatives);

	double f1_score = 100 * 2 * (precision * recall) / (precision + recall);

	cout << "Calculated F1 Score is " << f1_score << "\%!" << endl;
	return 0;
}

/** @function detectAndDisplay */
int detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

       // 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

       // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}

	return faces.size();

}
