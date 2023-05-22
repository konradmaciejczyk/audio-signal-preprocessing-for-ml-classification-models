import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from classifiers import KNN, NearestCentroid

def normalize_data(x_train, x_test):
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    return x_train, x_test

def loadData(path, verbose = False):
    datapd = pd.read_csv(path)
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    datapd['class'] = pd.factorize(datapd['class'])[0]

    if verbose:
        print("Number of sound tracks in dataset: {}\nFeatues: ".format(datapd['track_name'].count()), end="")
        for feature in datapd.keys():
            print(feature, end="     ")
        print("\nGenres: ", end="")
        for genre in genres:
            print(genre, end="     ")
        print("\n")
        datapd.head()

    return datapd, genres

def cat_data_handle(datapd, verbose=False):
    data = pd.get_dummies(data=datapd)
    
    if verbose:
        data.head()

    return data

def dataSplit(data, random_state = 42, test_size=.2, verbose=False):
    X = data.drop("class", axis = 1).to_numpy()
    y = data['class'].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, shuffle=True, test_size=.2)

    if verbose:
        print("x_train={}, x_test={}, y_train={}, y_test={}".format(x_train.shape, x_test.shape, y_train.shape, y_test.shape))

    return x_train, x_test, y_train, y_test

def runKNN(x_train, x_test, y_train, y_test, neighbors = 3):
    my_knn_clf = KNN(k=neighbors)
    my_knn_clf.fit(x_train, y_train)
    print("Testing set score accuracy: {:.2f}% for KNN({})". format(my_knn_clf.score(x_test, y_test)*100, neighbors))

def runNearestCentroid(x_train, x_test, y_train, y_test):
    my_min_centroid_clf = NearestCentroid()
    my_min_centroid_clf.fit(x_train, y_train)
    print("Testing set score accuracy: {:.2f}% for NearestCentroid". format(my_min_centroid_clf.score(x_test, y_test)*100))

if __name__ == '__main__':
    datapd, genres = loadData("./gtzan_dataset.csv")
    datapd = cat_data_handle(datapd)
    x_train, x_test, y_train, y_test = dataSplit(datapd)
    x_train, x_test = normalize_data(x_train, x_test)

    #runKNN(x_train, x_test, y_train, y_test)

    runNearestCentroid(x_train, x_test, y_train, y_test)
