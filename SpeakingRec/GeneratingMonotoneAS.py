#this step is useful to generate the audio signal with some predefined
#parameters. This step will save the audio signal in an output file

#we are going to generate a monotone signal using python, which will be stored in a file
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
output_file = 'audio_signal_generated.wav'
duration = 4
frequency_sampling = 44100 #in Hz
frequency_tone = 784
min_val = -4 * np.pi
max_val = 4 * np.pi

#generating audio signal
t = np.linspace(min_val, max_val, duration*frequency_sampling)
audio_signal = np.sin(2*np.pi*frequency_tone*t)

#saving the audio file in the output file
write(output_file, frequency_sampling, audio_signal)

#extracting first 100 values of our graph
audio_signal = audio_signal[:100]
time_axis = 1000*np.arange(0, len(audio_signal), 1)/float(frequency_sampling)
plt.plot(time_axis, audio_signal, color='blue')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.title('Generated audio signal')
plt.show()