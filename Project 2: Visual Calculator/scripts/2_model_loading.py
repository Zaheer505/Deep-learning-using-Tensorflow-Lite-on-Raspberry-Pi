
import os
import numpy as np
from tflite_runtime.interpreter import Interpreter

model_path = "/home/rpi/Tiny-ML/Project 2: Visual Calculator/data/tflite_model.tflite"

interpreter = Interpreter(model_path)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
    
interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]['shape']

print(output_details)