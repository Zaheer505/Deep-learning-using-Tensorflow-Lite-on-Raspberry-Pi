import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import operator



def main():
    interpreter = tflite.Interpreter(model_path='/home/pi/Tiny-ML/Project_2: Visual Calculator/data/model/vc_model.tflite')
    labels = ["divide" , "eight","five","four","min","mul","nine","one","plus","seven","six","three","two"]
    integer_labels = [operator.floordiv ,8 , 5 , 4 , operator.sub, operator.mul , 9, 1 ,operator.add , 7 ,6 ,3,2]
    input_details   = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    video_feed = cv2.VideoCapture(0)

    current_prediction  = 0
    previous_prediction = 0
    prediction_array = [14]
    equation_array = []
    while(1):
        _,frame = video_feed.read()
        gray_scaled = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.rectangle(gray_scaled, (350,100),(550,300), (255,0,0), 2)
        sqaure_region = gray_scaled[100:300,350:550]
        cv2.imshow("Full Frame", gray_scaled)
        cv2.waitKey(1)
        circle_location = cv2.HoughCircles(sqaure_region , cv2.HOUGH_GRADIENT,1.2,20)

        if circle_location is not None:
            circle_location = np.round(circle_location[0, :]).astype("int")

            for i in circle_location:
                center_x = i[0];center_y = i[1];radius = i[2]
                x=center_x - radius ;y=center_y - radius
                h=2*radius;w=2*radius
            if y>0 and x>0:

                cropped_circle = sqaure_region[y:y+h , x:x+h]
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

                if current_prediction == previous_prediction:
                    if( prediction_array[len(prediction_array)-1] != current_prediction):
                        prediction_array.append(current_prediction)
                        equation_array.append(labels[current_prediction])

                    else:
                        print("Skipping Same Entry")

                previous_prediction = current_prediction

                print("Prediction Array ", prediction_array)
                print("Equation Array ", equation_array)

                if (len(prediction_array) >=4):
                    ele_a     = integer_labels[prediction_array[1]]
                    operation = integer_labels[prediction_array[2]]
                    ele_b     = integer_labels[prediction_array[3]]

                    print(" Equation Result -> " , operation(ele_a,ele_b))


                cv2.imshow("Region of Interest", roi)
                cv2.waitKey(1)


if __name__ == '__main__':
    main()



