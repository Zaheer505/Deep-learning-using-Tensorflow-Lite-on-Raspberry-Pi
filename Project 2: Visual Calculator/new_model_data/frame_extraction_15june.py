from tflite_runtime.interpreter import Interpreter
#import tensorflow as tf
import numpy as np
import time
import cv2
from PIL import Image

#capture = cv2.VideoCapture(0)
capture = cv2.VideoCapture('C:/Users/Zaheer43/Downloads/new_model_training/videos/div.avi')


def main():

    j = 0

    while(1):
        ret, frame = capture.read() # frame is in numpy array
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if ret == False:
                 break
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
        #cv2.imshow("frame", grayFrame)
        circles = cv2.HoughCircles(grayFrame, cv2.HOUGH_GRADIENT, 1.2, 20)
        if circles is not None:
                  circles = np.round(circles[0, :]).astype("int")
                  height = frame.shape[0]
                  width = frame.shape[1]
                  lemonROIs = []
                  for i in circles:
                           canvas = np.ones((height, width))
                           j+=1
                           color = (0, 0, 0)
                           thickness = -1
                           centerX = i[0]
                           centerY = i[1]
                           radius = i[2]
                           output = frame.copy()
                           output[canvas == 0] = (0, 0, 0)
                           x = centerX - radius
                           y = centerY - radius
                           h = 2 * radius
                           w = 2 * radius

                           print (x,y,h,w)


                           if x > 0 and y > 0:

                               print ('cirlce')

                               circle_image = output[y:y + h, x:x + w]
                               print (circle_image.shape)

                               circle_resize = cv2.resize(circle_image, (500, 500))
                               print (circle_resize.shape)

                               croppedImg = circle_resize[90:410, 90:410]
                               print (croppedImg.shape)
                               
                               # use this to save every frame which will be given to model
                               cv2.imwrite('C:/Users/Zaheer43/Downloads/new_model_training/div/'+str(j)+'.jpg', croppedImg)

                               print ("\nframe extracted"+str(j))

if __name__ == '__main__':
	main()
    
