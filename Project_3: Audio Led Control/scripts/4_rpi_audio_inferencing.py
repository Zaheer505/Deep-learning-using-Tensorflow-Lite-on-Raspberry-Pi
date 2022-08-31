import numpy as np
import sounddevice as sd
import tflite_runtime.interpreter as tflite
from scipy import signal

import RPi.GPIO as GPIO
import time


led_red = 21
led_green = 20

GPIO.setmode(GPIO.BCM)
GPIO.setup(led_red,GPIO.OUT)
GPIO.setup(led_green,GPIO.OUT)
GPIO.setwarnings(False)

def logger(variable_name , variable_value):
    print(variable_name ," : " , variable_value)

def main():
    labels = ['off', 'on', 'green', 'red']
    ### Reading Audio from Mic
    duration = 1; fs = 22050
    print("Speak Now ")
    audio_rec = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    int_audio = (np.clip(audio_rec, -32768, 32767)) * 32767
    int_audio=int_audio.astype(np.int16)
    int_audio = np.squeeze(int_audio , axis =1)

    ### producing spectrogram
    f, t, spec = signal.stft(int_audio, fs=22050, nperseg=255, noverlap = 124, nfft=256)
    spec=np.abs(spec)
    input_data = np.reshape(spec , (1,1,spec.shape[0],spec.shape[1]) )
    # logger("Input Data Shape",input_data.shape)

    ### Model Loading
    interpreter = tflite.Interpreter('model/audio_led_model.tflite')

    input_details   = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    ## model Predicting
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'] , input_data)
    interpreter.invoke()

    tflite_prediction_result = interpreter.get_tensor(output_details[0]['index'])
    label_index = np.argmax(tflite_prediction_result)
    logger("\n\n\nLite Model Predictions --> ",labels[label_index] ) #  labels[np.argmax(tflite_prediction_result)]


    if(label_index == 0):
        GPIO.output(led_red,GPIO.LOW)
        GPIO.output(led_green,GPIO.LOW)

    elif(label_index == 1):
        GPIO.output(led_red,GPIO.HIGH)
        GPIO.output(led_green,GPIO.HIGH)

    elif(label_index == 2):
        GPIO.output(led_red,GPIO.LOW)
        GPIO.output(led_green,GPIO.HIGH)

    elif(label_index == 3):
        GPIO.output(led_red,GPIO.HIGH)
        GPIO.output(led_green,GPIO.LOW)



if __name__ == '__main__':
    main()
