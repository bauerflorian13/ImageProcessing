#
# Makefile to build the cpp sources
#

CC=g++

EXES=face dartboard

face: face.cpp
	$(CC) -o $@ $^ `pkg-config opencv --cflags --libs`
    
dartboard: dartboard.cpp
	$(CC) -o $@ $^ `pkg-config opencv --cflags --libs`

    
all: $(EXES)

.PHONY: clean all

clean:
	\rm face
	\rm dartboard

