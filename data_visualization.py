#Konrad Maciejczyk, 2023, Wrocław Uniwersity of Science and Technology
import pandas as pd 
import librosa
from random import randint, choice
import matplotlib.pyplot as plt
import numpy as np

from algorithms import fft

def loadCSVtoDataframe(path):
    datapd = pd.read_csv(path)
    genres = datapd['label'].unique()
    datapd['label'] = pd.factorize(datapd['label'])[0]
    return datapd, genres, datapd.shape

def loadAudioFile(genre = 'metal', track_num = '00054'):
    file_path = f'genres_original/{genre}/{genre}.{track_num}.wav'
    (samples, sampling_rate) = librosa.load(file_path, sr = None, mono = True, offset = 0.0, duration = None)

    duration = len(samples) / sampling_rate
    return samples, sampling_rate

def showAudioWavePlot(samples, sampling_rate, track="metal.00054", interval_start = 5, interval_length = 5):
    a = sampling_rate * interval_start
    b = (interval_length + interval_start) * sampling_rate 
    plt.figure(figsize = (14, 6), dpi = 120)
    librosa.display.waveshow(y = samples[a: b], sr = sampling_rate)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title(track + f' ({str(interval_start)}s - {str(interval_start + interval_length)}s)')
    plt.show()


def showFrequencies(samples, sampling_rate, track="metal.00054", interval_start = 5, interval_length = 5):
    a = sampling_rate * interval_start
    b = (interval_length + interval_start) * sampling_rate 

    T = 1/sampling_rate
    n = len(samples[a: b])
    y = fft(samples[a: b])
    x = np.linspace(0.0, 1.0/(2.0*T), n//2)
    plt.figure(figsize = (14, 6), dpi = 120)
    plt.plot(x, 2.0/n * np.abs(y[:n//2]))
    plt.grid()
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title(track + f' ({str(interval_start)}s - {str(interval_start + interval_length)}s)')
    plt.show()

if __name__ == "__main__":
    #Load CSV
    datapd, genres, dataset_shape = loadCSVtoDataframe('gtzan_extracted_features.csv')
    print(f"Genres ({len(genres)}): ", genres)
    print(f"Objects: {dataset_shape[0]}, features: {dataset_shape[1] - 2}")

    #Load and visualize audio data
    genre = choice(genres)
    track_num = randint(1, 100)
    track = None
    if track_num < 10:
        track = '0000' + str(track_num)
    elif track_num >= 10 and track_num < 99:
        track = '000' + str(track_num)
    else:
        track = '00100'

    samples, sampling_rate = loadAudioFile(genre, track)
    duration = len(samples) // sampling_rate
    print('Duration of single sound file: {:.1f} seconds.'.format(duration))

    showAudioWavePlot(samples, sampling_rate, track=f"{genre}.{track}", interval_start = 5, interval_length = 5)

    #Visualize frequencies
    showFrequencies(samples, sampling_rate, track=f"{genre}.{track}", interval_start = 5, interval_length = 5)


    