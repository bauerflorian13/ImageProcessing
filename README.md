# ImageProcessing
Repository for the coursework in Image Processing and Computer Vision on the University of Bristol in 2018.

## Prerequisite

* you need a working installation of OpenCV
* __make__ should be installed

## Getting started
* cd in the main directory of this project and use make command to build and train model and compile all c++ sources:
```
make 
```

* the input images should be in the folder ___input_images___. 
* to run the dartboard detector for image __input_images/dart1.jpg__ just type:
```
./dartboard dart1.jpg
```
* the images with the houghspace and the thresholded gradient magnitude can be found in the __work_dir__ directory

__Note__: Do not use input_images/dart1.jpg as argument as the program looks automaticaly per default only in the __input_images__ directory. The ___output___ will be in the ___output_images___ directory and in this example it will be named as ___detected_dart1.jpg___. 

## Run dartboard detector on all input images

For convenience their is a bash script __run.sh__ in the main directory as well. This will run the dartboard detector for all input files in the __input_images__ directory and output the result images in the __output_images__ directory.


## Github repository

There is also a Github repository which contains all needed files. We used it to work on the detector. If you prefer it you can download or clone it from https://github.com/darkcookie298/ImageProcessing.
Note: It was not published before the deadline for this project at Monday Dec 3rd at 17:59.
