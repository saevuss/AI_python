import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

#reading stored audio file
frequency_sampling, audio_signal = wavfile.read("/Users/giorgiasavo/Documents/projects/personal/AI_python/SpeakingRec/audio.wav")
#display the parameters like sampling frequency of the audio signal
print("\nSignal shape: ", audio_signal.shape)
print('Signal datatype: ', audio_signal.dtype)
print('Signal duration: ', round(audio_signal.shape[0] / float(frequency_sampling), 2), 'seconds')

#normalizing signal
audio_signal = audio_signal / np.power(2, 15)
#extracting the first 100 values form this signal to visualize
audio_signal = audio_signal[:5000]
time_axis = 1000*np.arange(0, len(audio_signal), 1) / float(frequency_sampling)

#visualizing the signal
plt.plot(time_axis, audio_signal, color='blue')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.title('Input audio signal')
plt.show()