#!/bin/bash

echo "**************************************"
echo "*                                    *"
echo "*         DARTBOARD DETECTOR         *"
echo "* by Florian Bauer & Ruben Powar     *"
echo "*                                    *"
echo "**************************************"
echo""
echo "Hello! This script runs the dartdetector on all images in the input_images directory.\n"

read -n 1 -s -r -p "Press any key to continue"


# simple bash script that runs the dartboard detector on every image in the input image directory
rm -r output_images/*
for filename in input_images/*.jpg; do
	fn=$(basename -- "$filename")	
	./dartboard $fn 0
done

duration=$(( SECONDS - start ))

echo "Time needed for complete run" $duration "s. Output Images are in the output_images directory!"
