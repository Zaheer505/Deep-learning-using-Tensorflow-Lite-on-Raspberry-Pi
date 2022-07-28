
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write


i = 0

for i in range(50):
    category = 'green'
    file_name = category+"_" + str(i)
    duration = 1  # seconds
    fs = 22050
    print(" Speak now -> " , category)
    audio_rec = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    int_audio = (np.clip(audio_rec, -32768, 32767)) * 32767
    write('data/' +category + "/" + file_name + ".wav", fs, int_audio.astype(np.int16))

    print("Recorded -> ", i)