#!/usr/bin/env python3
# This file is real time predicting file

from tflite_runtime.interpreter import Interpreter 
import numpy as np
import time
import cv2
from PIL import Image

def main():
    
    labels = ["one","two","three","four","five","six","seven","eight","nine","addition","division","multiplication","subtraction"]
    data_folder = "/home/pi/Desktop/Tiny-ML/model_files/visual_calculator/"
    model_path = data_folder + "tf_lite_model.tflite"
    interpreter = Interpreter(model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    
    video_in = cv2.VideoCapture(0)
    video_in.set(cv2.CAP_PROP_FRAME_WIDTH, 128)
    video_in.set(cv2.CAP_PROP_FRAME_HEIGHT, 128)
    while(1):
        _,frame = video_in.read() # frame is in numpy array
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # image = Image.open(frame).convert('L')
        image = np.expand_dims(frame, axis = 2)
        image = (np.expand_dims(image, axis = 0)).astype(np.float32)
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])
        print ('TF Lite Prediction : ',labels[np.argmax(tflite_model_predictions)-1])
        cv2.imshow("frame",frame)
        cv2.waitKey(1)

if __name__ == '__main__':
	main()