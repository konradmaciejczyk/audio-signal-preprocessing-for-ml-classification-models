import numpy as np 

class Nearest_centroid(object):
    """A nearest centroid classifier. Similar class to NearestCentroid from module sklearn.neighbors.
    Methods:
        fit(np.array(X_train), np.array(y_train)) - model training method
        predict(np.array(X_test)) - method returning a numpy array of predicted class labels for input test data.
        score(np.array(X_test, y_test)) - method checking accuracy of the model by. Returns float number.
    """
    def __int__(self):
        self.X = None
        self.Y = None
        self.repr = None

    def fit(self, X_train, y_train):
        if X_train.shape[0] != y_train.shape[0]:
            print(A.shape[0], B.shape[0])
            raise ValueError("Training and testing sets are not same size")


        self.X = X_train
        self.Y = y_train
        self.repr = self._find_means()


    def _find_means(self):
        """Auxillary method used for computing feature classes means."""
        return np.array([np.mean(self.X[self.Y == i], axis = 0) for i in np.unique(self.Y)])

    def _find_distance(self, x):
        """Auxillary method for computing distances between one of the testing vector and classes feature means."""
        return np.sqrt(np.sum(np.power(self.repr - x, 2), axis = 1))

    
    def predict(self, A):
        if type(A) is not np.ndarray:
            raise ValueError("Both sets must numpy.ndarray type.")
        
        result = np.array([]).astype('int8')
        i = 0
        n = A.shape[0]
        while i < n:
            result = np.append(result , np.argmin(self._find_distance(A[i])))
            i += 1
        return result

    def score(self, A, B):
        results = self.predict(A)
        return np.mean(results == B)

class KNN:
    """A k-nearest neighbors classifier. Similar class to KNeighborsClassifier from module sklearn.neighbors.
    Methods:
        fit(np.array(X_train), np.array(y_train)) - model training method
        predict(np.array(X_test)) - method returning a numpy array of predicted class labels for input test data.
        score(np.array(X_test, y_test)) - method checking accuracy of the model by. Returns float number.
    """
    def __init__(self, k = 1):
        if k < 1 or type(k) is not int:
            raise ValueError("Number of nearest neighbors mu be positive integer.")

        self.k = k #k_neighbors
        self.x_train = None #storing data
        self.y_train = None #storing target

    def fit(self, A, B):
        if A.shape[0] != B.shape[0]:
            print(A.shape[0], B.shape[0])
            raise ValueError("Training and testing sets are not same size")

        self.x_train = A #fiting data
        self.y_train = B

    def predict(self, A):
        if type(A) is not np.ndarray:
            raise ValueError("Both sets must numpy.ndarray type.")

        result = np.array([]).astype('int16')
        i = 0
        n = A.shape[0]
        while i < n:
            distance = self._distance(A[i])
            distance = self.y_train[np.argsort(distance)[:self.k]]
            distance = np.bincount(distance)
            distance = np.argmax(distance)

            result = np.append(result, distance)
            i += 1
        
        return result

    def score(self, X, Y):
        if X.shape[0] != Y.shape[0]:

            raise ValueError("Training and are not same size")
        
        result = self.predict(X)
        return np.mean(result == Y)

    def _distance(self, A):
        """Auxillary method for computing distances between one of the testing vector and other training vectors"""
        return np.sqrt(np.sum(np.power(A - self.x_train, 2), axis = 1))