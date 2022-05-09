'''
Get information about wav file properties through audioiotensor
'''
import tensorflow_io as tfio
import tensorflow as tf

audio_file="/home/luqman/Desktop/rpi_audio/data/on/Recording_10.wav"
audio = tfio.audio.AudioIOTensor(audio_file)

print("Audio Properties\n")
print(audio)
print("\nConverting into a tensor \n")
audio_slice = audio[100:]
audio_tensor = tf.squeeze(audio_slice, axis=[-1])

print(audio_tensor)

### Visualizing in python notebook
# def audio_details(data_file_path ):
#   audio = tfio.audio.AudioIOTensor(data_file_path)
#   print("Audio Properties\n")
#   print(audio)
#   print("\nConverting into a tensor \n")
#   audio_slice =audio[100:]
#   # remove last dimension
#   audio_tensor = tf.squeeze(audio_slice, axis=[-1])
#   tensor = tf.cast(audio_tensor, tf.float32) / 32768.0
#   plt.figure()
#   plt.plot(tensor.numpy())
#   print("audio listen")
#   Audio(audio_tensor.numpy(), rate=audio.rate.numpy())


# audio_details("/content/data_off.wav")
# audio_details("/content/rec_green.wav")