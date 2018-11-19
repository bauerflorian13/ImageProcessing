#
# Makefile to build and train the model and compile cpp source
#

CC=g++

EXES=dartboard createsamples trainmodel face houghtransform

NUM_OF_IMAGES=1000

all: $(EXES)

face: face.cpp
	$(CC) -o $@ $^ `pkg-config opencv --cflags --libs`

dartboard: dartboard.cpp
	$(CC) -o $@ $^ `pkg-config opencv --cflags --libs`

houghtransform: houghtransform.cpp
	$(CC) -o $@ $^ `pkg-config opencv --cflags --libs`

createsamples: dart.bmp
	opencv_createsamples -img dart.bmp -vec dart.vec -neg negatives.dat -w 20 -h 20 -num $(NUM_OF_IMAGES) -maxidev 80 -maxxangle 0.8 -maxyangle 0.8 -maxzangle 0.2

trainmodel: dart.vec
	opencv_traincascade -data dartcascade -vec dart.vec -bg negatives.dat -numPos $(NUM_OF_IMAGES) -numNeg $(NUM_OF_IMAGES) -numStages 3 -maxDepth 1 -w 20 -h 20 -minHitRate 0.999 -maxFalseAlarmRate 0.05 -mode ALL

.PHONY: clean all

clean:
	\rm face
	\rm dartboard
	\rm dart.vec
	\rm houghtransform
	\rm -r dartcascade/*
