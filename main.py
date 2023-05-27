#Konrad Maciejczyk, 2023, Wrocław Uniwersity of Science and Technology

import pandas as pd

from create_csv import checkDataset, checkCSV, createDataset
from data_visualization import loadCSVtoDataframe, loadAudioFile, showAudioWavePlot, showFrequencies

def loadData(path):
    datapd = pd.read_csv(path)
    datapd['label'] = pd.factorize(datapd['label'])[0]
    return datapd

if __name__ == '__main__':
    csv_file = 'gtzan_extracted_features.csv'
    
    #feature engineering
    csv_name, dataset_name = checkCSV(file_name='gtzan_extracted_features.csv'), checkDataset()

    if csv_name and dataset_name:
        createDataset(csv_name, dataset_name, probes_per_class = 10, samples_section_start = 15, samples_section_length = 10, feature_amount = 50)

    #visualization
    datapd = loadCSVtoDataframe('gtzan_extracted_features.csv')

    samples, sampling_rate = loadAudioFile('metal', '00004')

    showAudioWavePlot(samples, sampling_rate, track='metal.00004', interval_start = 5, interval_length = 5)
    showFrequencies(samples, sampling_rate, track="metal.00004", interval_start = 5, interval_length = 5)

    #training and testing models

    

    

