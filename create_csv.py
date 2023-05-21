#Konrad Maciejczyk, 2023, Wrocław Uniwersity of Science and Technology

import os
import csv
import random
import librosa
import numpy as np

def fft(samples):
    n = len(samples)
    if n <= 1:
        return samples

    even = fft(samples[0::2])
    odd = fft(samples[1::2])

    temp = np.zeros(n).astype(np.complex64)

    for u in range(n // 2):
        temp[u] = even[u] + np.exp(-2j * np.pi * u / n) * odd[u]
        temp[u + n // 2] = even[u] - np.exp(-2j * np.pi * u / n) * odd[u]

    return temp

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

def extract_features(dataset_name, probes, sample_section_start, sample_section_length, feature_amount):
    rows = []

    rows.append(['track_name'])
    for i in range(1, feature_amount + 1):
        rows[0].append(f'feat_{i}')
    rows[0].append('class')

    for probe in probes:
        row = []
        row.append(probe)
        aux = dataset_name + '/' + probe.split('.')[0] + '/' + probe
        samples, sampling_rate = librosa.load(aux, sr = None, mono = True, offset = 0.0, duration = None)        
        samples = samples[sample_section_start: sample_section_start + sample_section_length * sampling_rate]

        print(f"Extracting features for {probe}...")
        harmonics = fft(samples)
        harmonics = 2.0/len(samples) * np.abs(harmonics[:len(samples)//2])
        for interval in range(0, len(harmonics), len(harmonics) // 60):
            feature_value = np.trapz(harmonics[interval: interval + len(harmonics) // 60])
            row.append(feature_value)
        row.append(probe.split('.')[0])
        rows.append(row)

    return rows

def createDataset(csv_name, dataset_name, classes=['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'], probes_per_class=100, shuffle=True, sample_section_start = 9, samples_section_length = 3, feature_amount = 60):

    probes = get_probes(dataset_name, classes, probes_per_class, shuffle)
    rows = extract_features(dataset_name, probes, sample_section_start, samples_section_length, feature_amount)  
    print(len(rows)) 

    f = open(csv_name, 'w', newline="")   
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)
    f.close() 
    

if __name__ == '__main__':
    csv_name = checkCSV()
    dataset_name = checkDataset()

    if csv_name and dataset_name:
        createDataset(csv_name, dataset_name, probes_per_class = 50)