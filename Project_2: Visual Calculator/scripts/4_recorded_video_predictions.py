from calendar import c
import cv2
from matplotlib.pyplot import gray
import numpy as np
from sympy import Equality
import tensorflow as tf


def solve_equation(equation):
    for i in equation:
        print(i)



def main():
    interpreter = tf.lite.Interpreter('data/model/vc_model.tflite')
    labels = ["divide" , "eight","five","four","min","mul","nine","one","plus","seven","six","three","two"]
    input_details   = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    video_feed = cv2.VideoCapture(0)
    current_prediction  = 0
    previous_prediction = 0
    equation_array=[]
    while(1):
        _,frame = video_feed.read()
        gray_scaled = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        cv2.imshow("Full Frame", gray_scaled)
        # gray_scaled = gray_scaled [161:290,461:603 ]
        circle_location = cv2.HoughCircles(gray_scaled , cv2.HOUGH_GRADIENT,1.2,20)
        # x=461:603
        # y=161:290
        if circle_location is not None:
            circle_location = np.round(circle_location[0, :]).astype("int")

            for i in circle_location:
                center_x = i[0];center_y = i[1];radius = i[2]
                x=center_x - radius ;y=center_y - radius
                h=2*radius;w=2*radius
            if y>0 and x>0:

                cropped_circle = gray_scaled[y:y+h , x:x+h]
                cropped_circle_resize = cv2.resize(cropped_circle , (500,500))
                roi = cropped_circle_resize[90:410 , 90:410]

                input_image = cv2.resize(roi,(128,128))
                input_image = np.expand_dims(input_image,axis=2)
                input_image = np.expand_dims(input_image,axis=0)
                input_image = input_image.astype(np.float32)
#

                interpreter.allocate_tensors()
                interpreter.set_tensor(input_details[0]['index'] , input_image)
                interpreter.invoke()
                current_prediction = interpreter.get_tensor(output_details[0]['index'])
                current_prediction = np.argmax(current_prediction)

                if not current_prediction == previous_prediction:
                    previous_prediction = current_prediction
                    equation_array.append(labels[previous_prediction])
                    # print("\n\nPredictions Result :", labels[current_prediction],  "\n" )
                    solve_equation(equation_array)

                cv2.imshow("Region of Interest", roi)
                cv2.waitKey(1)


if __name__ == '__main__':
    main()


