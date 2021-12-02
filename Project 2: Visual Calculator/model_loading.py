
import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


from keras.models import load_model

classifier = load_model('/content/cal_2e2.h5')
classifier.summary()