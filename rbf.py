import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted    

class RBFNetwork(BaseEstimator, ClassifierMixin):
    def __init__(self,n_centers=100, gamma=0.01,random_state=42):
        self.n_centers = n_centers
        self.gamma = gamma
        self.random_state = random_state
        self.centers_ = None
        self.clf_ = None # to 2o layer tou diktyou p arxikopoieitai meta to fit
        #kato paula sto telos gia na deijoume oti pernei timi meta to fit

    def _rbf_kernel(self, X):
        n_samples = X.shape[0]
        activations  = np.zeros((n_samples, self.n_centers))

        for i, center in enumerate(self.centers_):
            distance_squared = np.sum((X - center) ** 2, axis=1)
            activations[:, i] = np.exp(-self.gamma * distance_squared)

        return activations
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        
        kmeans = KMeans(n_clusters=self.n_centers, random_state=self.random_state)
        kmeans.fit(X)
        self.centers_ = kmeans.cluster_centers_

        X_rbf = self._rbf_kernel(X)

        self.clf_ = LogisticRegression(random_state=self.random_state)
        self.clf_.fit(X_rbf, y)

        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        X_rbf = self._rbf_kernel(X)

        return self.clf_.predict(X_rbf)