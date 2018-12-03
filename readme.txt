# ImageProcessing
Coursework for Image Processing and Computer Vision at the University of Bristol in 2018.

## Prerequisite

* you need a working installation of OpenCV
* make should be installed

## Getting started
* cd to the main directory of this project and use make command to build and train model and compile all c++ sources: make 
* the input images should be in the folder input_images. 
* to run the dartboard detector for image input_images/dart1.jpg just type: ./dartboard dart1.jpg
* the images with the Houghspace and the thresholded gradient magnitude can be found in the work_dir directory

Note: Do not use input_images/dart1.jpg as argument as the program looks automatically per default only in the input_images directory. The output will be in the output_images directory and in this example it will be named as detected_dart1.jpg. 

## Run dartboard detector on all input images

For convenience there is a bash script run.sh in the main directory as well. This will run the dartboard detector for all input files in the input_images directory and output the resulting images in the output_images directory.


## Github repository

There is also a Github repository which contains all needed files. We used it to work on the detector. If you prefer you can download or clone it from https://github.com/darkcookie298/ImageProcessing.
Note: It was not published before the deadline for this project at Monday Dec 3rd at 17:59.
