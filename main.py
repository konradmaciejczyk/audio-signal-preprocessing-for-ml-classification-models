#Konrad Maciejczyk, 2023, Wrocław Uniwersity of Science and Technology
import pandas as pd
import numpy as np

from create_csv import checkDataset, checkCSV, createDataset
from data_visualization import loadCSVtoDataframe, loadAudioFile, showAudioWavePlot, showFrequencies, showFeatureCorrelationHeatmap, showScore
from models import normalizeData

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, clone

from sklearn import neighbors, naive_bayes, svm

from scipy.stats import ttest_rel

def loadData(path):
    datapd = pd.read_csv(path)
    datapd['label'] = pd.factorize(datapd['label'])[0]
    return datapd

if __name__ == '__main__':
    # csv_file = 'gtzan_extracted_features.csv'
    
    # #feature engineering
    # csv_name, dataset_name = checkCSV(file_name='gtzan_extracted_features.csv'), checkDataset()

    # if csv_name and dataset_name:
    #     createDataset(csv_name, dataset_name, probes_per_class=100, samples_section_start = 9, samples_section_length = 3, feature_amount = 60, repeats = 1)

    # #data visualization
    # datapd, genres, data_shape = loadCSVtoDataframe('gtzan_extracted_features.csv')

    # samples, sampling_rate = loadAudioFile('metal', '00004')
    # showFeatureCorrelationHeatmap(datapd)
    # showAudioWavePlot(samples, sampling_rate, track='metal.00004', interval_start = 5, interval_length = 5)
    # showFrequencies(samples, sampling_rate, track="metal.00004", interval_start = 5, interval_length = 5)

    # #training and testing models on created datasets
    rskf = RepeatedStratifiedKFold(n_repeats= 5, n_splits = 2)
    colors = ['red', 'green', 'blue', 'purple']
    classifiers = ['KNN(3)', 'Nearest Centroid', 'Gaussian NaiveBayes', 'SVM']

    features_3_sec_dataset = pd.read_csv('extracted_features_3sec.csv')
    features_5_sec_dataset = pd.read_csv('extracted_features_5sec.csv')
    features_10_sec_dataset = pd.read_csv('extracted_features_10sec.csv')
    features_15_sec_dataset = pd.read_csv('extracted_features_15sec.csv')
    features_30_sec_dataset = pd.read_csv('extracted_features_30sec.csv')

    DATASETS = [features_30_sec_dataset, features_15_sec_dataset, features_10_sec_dataset, features_5_sec_dataset, features_3_sec_dataset]
    CLASSIFIERS = [neighbors.KNeighborsClassifier(), neighbors.NearestCentroid(), naive_bayes.GaussianNB(), svm.SVC()]

    scores = np.zeros(shape = (len(DATASETS), len(CLASSIFIERS), rskf.get_n_splits()))

    for dataset_idx, dataset in enumerate(DATASETS):
        X = dataset.drop(['label', 'filename'], axis = 1).to_numpy()
        dataset['label'] = pd.factorize(dataset['label'])[0]
        y = dataset['label'].to_numpy()
        for classifier_idx, clf_prot in enumerate(CLASSIFIERS):
            for fold_idx, (train, test) in enumerate(rskf.split(X, y)):
                X[train], X[test] = normalizeData(X[train], X[test])
                clf = clone(clf_prot)
                clf.fit(X[train], y[train])
                score = clf.score(X[test], y[test])
                scores[dataset_idx, classifier_idx, fold_idx] = score

    showScore(scores[0,:], classifiers, 'extracted_features_30sec.csv', colors)
    showScore(scores[1,:], classifiers, 'extracted_features_15sec.csv', colors)
    showScore(scores[2,:], classifiers, 'extracted_features_10sec.csv', colors)
    showScore(scores[3,:], classifiers, 'extracted_features_5sec.csv', colors)
    showScore(scores[4,:], classifiers, 'extracted_features_3sec.csv', colors)


