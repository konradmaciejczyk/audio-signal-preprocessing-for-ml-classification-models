#Konrad Maciejczyk, 2023, Wrocław Uniwersity of Science and Technology
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors

from algorithms import KNN, NearestCentroid

def normalizeData(x_train, x_test):
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    return x_train, x_test

def dataSplit(data, random_state = 42, test_size=.2, verbose=False):
    X = data.drop(["label", "filename"], axis = 1).to_numpy()
    datapd['label'] = pd.factorize(datapd['label'])[0]
    y = data['label'].to_numpy()
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, shuffle=True, test_size=.2)

    if verbose:
        print("x_train={}, x_test={}, y_train={}, y_test={}".format(x_train.shape, x_test.shape, y_train.shape, y_test.shape))

    return x_train, x_test, y_train, y_test

def trainKNN(x_train, x_test, y_train, y_test, n = 3, perform_test = False):
    _x_train, _x_test = normalizeData(x_train, x_test)

    my_knn_clf = KNN(k=n)
    my_knn_clf.fit(_x_train, y_train)

    if perform_test:
        print("Testing set score accuracy: {:.2f}% for KNN (my_implementation) ({} neighbors)".format(my_knn_clf.score(_x_test, y_test)*100, n))

    sklearn_knn_clf = neighbors.KNeighborsClassifier(3)
    sklearn_knn_clf.fit(_x_train, y_train)

    if perform_test:
        print("Testing set score accuracy: {:.2f}% for KNN (sklearn) ({} neighbors) \n".format(sklearn_knn_clf.score(_x_test, y_test)*100, n))

    return sklearn_knn_clf

def trainNearestCentroid(x_train, x_test, y_train, y_test, perform_test = False):
    _x_train, _x_test = normalizeData(x_train, x_test)

    my_min_centroid_clf = NearestCentroid()
    my_min_centroid_clf.fit(_x_train, y_train)

    if perform_test:
        print("Testing set score accuracy: {:.2f}% for NearestCentroid (my_implementation)".format(my_min_centroid_clf.score(_x_test, y_test)*100))

    sklearn_nearest_centroid_clf = neighbors.NearestCentroid()
    sklearn_nearest_centroid_clf.fit(_x_train, y_train)

    if perform_test:
        print("Testing set score accuracy: {:.2f}% for NearestCentroid (sklearn) \n".format(sklearn_nearest_centroid_clf.score(_x_test, y_test)*100))

    return sklearn_nearest_centroid_clf

if __name__ == '__main__':
    datapd = pd.read_csv('gtzan_extracted_features.csv')

    x_train, x_test, y_train, y_test = dataSplit(datapd)
    trainNearestCentroid(x_train, x_test, y_train, y_test, perform_test = True)
    trainKNN(x_train, x_test, y_train, y_test, perform_test = True)