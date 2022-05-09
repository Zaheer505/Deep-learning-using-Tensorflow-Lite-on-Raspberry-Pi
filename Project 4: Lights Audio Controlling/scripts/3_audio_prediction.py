
'''
- Loads model
- Recoding audio and saving wav file then predicting
'''

import sounddevice as sd
from scipy.io.wavfile import write
from tflite_runtime.interpreter import Interpreter
import numpy as np
from scipy import signal
from scipy.io import wavfile


audio_name='recording.wav'
'''
loading Trained Model
'''

model_path = '/home/luqman/Desktop/rpi_audio/tflite_audio_27mar.tflite'
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("recording")

fs = 22050;seconds = 1
audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()
audio = (np.clip(audio, -32768, 32767)) * 3276
write(audio_name, fs, audio.astype(np.int16))  # Save as WAV file

rate, audio = wavfile.read (audio_name)

f, t, spec = signal.stft(audio, fs=22050, nperseg=255, noverlap = 124, nfft=256)
spectrogram = np.abs(spec)
spec2 = np.reshape(spectrogram,(1, spectrogram.shape[0], spectrogram.shape[1], 1))
print (spec2.shape)

interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]['index'], spec2)
interpreter.invoke()

predictionx = interpreter.get_tensor(output_details[0]['index'])
output = np.argmax(predictionx[0])

if output==0:
    print ('on')
elif output==1:
    print ('off')
elif output==2:
    print ('green')
elif output==3:
    print ('red')









