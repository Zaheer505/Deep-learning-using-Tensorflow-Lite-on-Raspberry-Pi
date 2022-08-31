import cv2
from cv2 import HOUGH_GRADIENT
import numpy as np


def logger(var_name,var_value):
    print(" --- ")
    print(">> ",var_name," : ", var_value)


def main():
    video_feed = cv2.VideoCapture('data/videos/plus.m4v')

    iterator = 0



    while(1):
        _,frame = video_feed.read()
        gray_scaled = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        circle_location = cv2.HoughCircles(gray_scaled , cv2.HOUGH_GRADIENT,1.2,20)

        if circle_location is not None:
            circle_location = np.round(circle_location[0, :]).astype("int")

            for i in circle_location:
                center_x = i[0];center_y = i[1];radius = i[2]
                x=center_x - radius ;y=center_y - radius
                h=2*radius;w=2*radius
                # logger("X", x)
                # logger("Y", y)
                # logger("H", h)
                # logger("W", w)
            if y>0 and x>0:

                cropped_circle = gray_scaled[y:y+h , x:x+h]
                cropped_circle_resize = cv2.resize(cropped_circle , (500,500))
                roi = cropped_circle_resize[90:410 , 90:410]

                iterator = iterator +1
                cv2.imwrite("data/extracted_images/plus/"+str(iterator)+'.jpg',roi)
                # cv2.imshow("camera_feed", gray_scaled)
                # cv2.imshow("Cropper Circle Resize", cropped_circle_resize)
                cv2.imshow("Region of Interest", roi)
                cv2.waitKey(1)


if __name__ == '__main__':
    main()


