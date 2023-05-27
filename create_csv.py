#Konrad Maciejczyk, 2023, Wrocław Uniwersity of Science and Technology

import os
import csv
import random
import librosa
import numpy as np

from algorithms import fft

def checkCSV(file_path='./', file_name='gtzan_dataset.csv'):
    files = os.listdir(file_path)

    while True:
        if file_name in files:
            answer = input(f'File {file_name} already exist in {file_path} directory. Do you want to replace file? [y/n]: ')

            if answer.lower() == 'y':
                return file_path + file_name
            elif answer.lower() == 'n':
                return False
            else:
                continue
        else:
            return file_path + file_name

def checkDataset(dataset_path='./', dataset_name='genres_original'):
    files = os.listdir(dataset_path)
    
    if dataset_name in files:
        return dataset_path + dataset_name
    else:
        print(f'Dataset named {dataset_name} not found at {dataset_path}. Aborting.')
        return False

def get_probes(dataset_name, classes, probes_per_class, shuffle):
    probes = []
    for class_name in classes:
        files = list(filter(lambda x: x.endswith('.wav'), os.listdir(dataset_name + '/' + class_name + '/')))

        if shuffle:
            random.shuffle(files)

        for f in files[:probes_per_class]:
            probes.append(f)
    
    random.shuffle(probes)
    return probes

def extract_features(dataset_name, probes, sample_section_start, sample_section_length, feature_amount, verbose = True):
    """Extracting features by calculating FFT and calculating integral of each interval"""    
    rows = []
    rows.append(['filename'])
    #creating head for dataset
    for i in range(0, feature_amount):
        rows[0].append(f'feat_{i+1}')
    rows[0].append('label')

    for idx, probe in enumerate(probes):
        row = []
        row.append(probe)
        aux = dataset_name + '/' + probe.split('.')[0] + '/' + probe
        samples, sampling_rate = librosa.load(aux, sr = None, mono = True, offset = 0.0, duration = None) #opening file        
        samples = samples[sample_section_start * sampling_rate: sample_section_start*sampling_rate + sample_section_length * sampling_rate] #picking interval
        n = len(samples)
        if verbose:
            print(f"Extracting features for {probe} ({idx}/{len(probes)})...")
        harmonics = np.fft.fft(samples)
        #harmonics = fft(samples) #calculating fft
        m = len(harmonics)
        harmonics = 2.0/n * np.abs(harmonics[:n//2]) #reducing complex domain into real domain
        for interval in range(0, m, m // feature_amount + 1):
            feature_value = np.trapz(harmonics[interval: interval + m // 60]) #calculating integral for every interval
            row.append(feature_value)
        row.append(probe.split('.')[0])
        rows.append(row)

    return rows

def createDataset(csv_name, dataset_name, classes=['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'], probes_per_class=100, shuffle=True, samples_section_start = 9, samples_section_length = 3, feature_amount = 60):
    """
    Parameters:
        csv_name = output csv file for extracted features
        dataset_name (string) dataset containing .wav files
        classes = (list) describe which classes should be included in probes picking
        probes_per_class = (int) number of probes for each class
        shuffle = (bool) permutate dataset randomly before picking up probes
        sample_section_start = (int) timestamp as seconds (1-17) corresponding to beginning of interval for each probe
        sample_section_length = (int) length as seconds corresponding to length of each interval
    """
    probes = get_probes(dataset_name, classes, probes_per_class, shuffle) #picking up probes
    rows = extract_features(dataset_name, probes, samples_section_start, samples_section_length, feature_amount) #getting calculated features 

    #saving to csv file
    f = open(csv_name, 'w', newline="")   
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)
    f.close() 
    

if __name__ == '__main__':
    csv_name = checkCSV()
    dataset_name = checkDataset()

    if csv_name and dataset_name:
        createDataset(csv_name, dataset_name, probes_per_class = 50, samples_section_start = 15, samples_section_length = 10, feature_amount = 50)