#after converting the speech signal into the frequency domain, it is importanto to convert it into the usable form of feature
# vector. We can use different feature extraction techniques like MFCC, PLP, PLP-RASTA

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank

frequency_sampling, audio_signal = wavfile.read("/Users/giorgiasavo/Documents/projects/personal/AI_python/SpeakingRec/audio.wav")
audio_signal = audio_signal[:15000] #takes first 15000 values
#extracting mfcc features
features_mfcc = mfcc(audio_signal, frequency_sampling)
print('\nMFCC:\nNumber of windows: ', features_mfcc.shape[0])
print('Length of each feature=', features_mfcc.shape[1])

#plotting and visualizing the MFCC features
features_mfcc = features_mfcc.T
plt.matshow(features_mfcc)
plt.title('MFCC')
plt.show()

filterbanck_features = logfbank(audio_signal, frequency_sampling)
print('\nFilter bank: \nNumber of windows: ', filterbanck_features.shape[0])
print('Length of each feature=', filterbanck_features.shape[1])
features_filterbanck = filterbanck_features.T
plt.matshow(features_filterbanck)
plt.title('Filter bank')
plt.show()

#Immaginando l’audio come una foto con tutte le frequenze: MFCC prende questa “foto” e la riduce a una versione compatta,
# evidenziando solo le parti che il nostro orecchio percepisce come importanti.
#Questo è il motivo per cui MFCC funziona così bene in riconoscimento vocale e machine learning audio,
# perché rappresenta l’audio come lo sentirebbe un umano.
