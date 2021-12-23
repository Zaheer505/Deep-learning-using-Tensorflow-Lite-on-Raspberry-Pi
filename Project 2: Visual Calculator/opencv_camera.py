#!/usr/bin/env python3
#This is just video feed obtaining file
import cv2
import os


def main():
    video_in = cv2.VideoCapture(0)

    waitTime = 1
    while(1):
        _,frame = video_in.read()
        cv2.imshow("frame",frame)
        cv2.waitKey(1)

if __name__ == '__main__':
	main()



