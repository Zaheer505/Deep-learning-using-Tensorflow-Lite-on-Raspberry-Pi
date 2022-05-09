#!/usr/bin/env python3
import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
classifier = load_model('/home/luqman/Tiny-ML/model_files/visual_calculator/cal_2e2.h5')
# classifier.summary()


test_image_path='/home/luqman/Tiny-ML/model_files/visual_calculator/dataset/test set/add/add929.jpg'
test_image = image.load_img(test_image_path, target_size = (128, 128), color_mode='grayscale')
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
tf_prediction = np.argmax(result)
print ('TF prediction:', tf_prediction)
# printing model size in mb
print ('TF Size:', round(os.path.getsize('/home/luqman/Tiny-ML/model_files/visual_calculator/cal_2e2.h5')/(1024*1024), 3) , 'MB')
# Image(test_image_path)