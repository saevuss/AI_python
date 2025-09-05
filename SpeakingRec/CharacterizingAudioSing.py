#Characterizing and audio signal involves converting the time domain signal
# into frequency domain

#using Fourier Transform mathematical tool to convert the audio signal into frequency domain
#Characterizing a Signal:
#   1) reading the .wav file
#   2) normalizing it
#   3) converting it in frequency domain obtaining power spectrum
# time domain: original audio wave form
# frequency domain: power spectrum with the dominant frequencies that characterize the signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

#reading the stored audio file
frequency_sampling, audio_signal = wavfile.read("/Users/giorgiasavo/Documents/projects/personal/AI_python/SpeakingRec/audio.wav")
#display parameters
print("\nSignal shape: ", audio_signal.shape)
print('Signal datatype: ', audio_signal.dtype)
print('Signal duration: ', round(audio_signal.shape[0] / float(frequency_sampling), 2), 'seconds')
#normalizing signal
audio_signal = audio_signal / np.power(2, 15)
#appling mathematics tools for transforming into frequency domain
signal_frequency = np.fft.fft(audio_signal)
#normalization of frequency domain signal
length_signal = len(audio_signal)
half_length = int(length_signal/2)
signal_frequency = abs(signal_frequency[0:half_length])/length_signal
signal_frequency **=2

len_fts = len(signal_frequency)
if length_signal %2 :
    signal_frequency[1:len_fts] *=2
else:
    signal_frequency[1:len_fts] *= 2

#extracting power in decibel (dB)
signal_power = 10*np.log10(signal_frequency)

#adjusting the frequency in kHz for x-axis
x_axis = np.arange(0, half_length, 1)*(frequency_sampling/length_signal)/1000.0

#visualizing the characterization
plt.figure()
plt.plot(x_axis, signal_power, color='b')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Power (dB)')
plt.show()

