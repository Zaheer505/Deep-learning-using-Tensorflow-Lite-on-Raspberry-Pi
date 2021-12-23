#!/usr/bin/env python3
# This file is standalone single images prediction process
from tflite_runtime.interpreter import Interpreter 
import numpy as np
import time
from PIL import Image

labels = ["one","two","three","four","five","six","seven","eight","nine","addition","division","multiplication","subtraction"]

data_folder = "/home/pi/Desktop/Tiny-ML/model_files/visual_calculator/"

model_path = data_folder + "tf_lite_model.tflite"
label_path = data_folder + "visual_cal_labels.txt"
test_image_path=data_folder + "test set/div/div956.jpg"
## loading models
interpreter = Interpreter(model_path)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# print input & output values & shapes to check everything is same 
# as the orignal tf model.
# data types etc are changed during tf lite conversion & optimizations
print("Input Shape:", input_details[0]['shape'])
print("Input Type:", input_details[0]['dtype'])
print("Output Shape:", output_details[0]['shape'])
print("Output Type:", output_details[0]['dtype'])
print("---"*10)
interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]['shape']

test_image = Image.open(test_image_path).convert('L').resize((width, height))

test_image = np.expand_dims(test_image, axis = 2)
test_image = (np.expand_dims(test_image, axis = 0)).astype(np.float32)

# needed before execution
# tensorFlow lite pre-plans tensor allocations to optimize inference,
# so the user needs to call allocate_tensors() before any inference.
interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]['index'], test_image)


# predicting from tf lite model
time1 = time.time()
interpreter.invoke()
tflite_model_predictions = interpreter.get_tensor(output_details[0]['index'])
tf_lite_prediction = np.argmax(tflite_model_predictions)
time2 = time.time()
print ('TF Lite Prediction : ',tf_lite_prediction)
classification_time = np.round(time2-time1, 3)
print("Classificaiton Time =", classification_time, "seconds.")


classification_label = labels[tf_lite_prediction-1]
print(classification_label)
