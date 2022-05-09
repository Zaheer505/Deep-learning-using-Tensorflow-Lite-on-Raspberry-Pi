
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np


audio_name='rec_green.wav'
print("recording")

fs = 22050;seconds = 1
audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()
# write(audio_name, fs, audio)  # Save as WAV file# current encoding is float32 which is not what we have designed NN about
audio = (np.clip(audio, -32768, 32767)) * 32767 # why ? https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder/issues/46
write(audio_name, fs, audio.astype(np.int16))
# more detail https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html