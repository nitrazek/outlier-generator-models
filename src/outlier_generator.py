import numpy as np
from sklearn import utils


class OutlierGenerator:
    def __init__(self, estimator):
        self._estimator = estimator
        self._outliers_fn = "uniform"
        self._outlier_ratio = 0.2
        self._outlier_margin = 0.1
        self._random_state = 42

    def _generate_outliers(self, X):
        outliers_fn_map = {
            "uniform": self._generate_uniform_outliers
        }

        return outliers_fn_map[self._outliers_fn](X)

    def _generate_uniform_outliers(self, X):
        n_samples, n_features = X.shape
        n_outliers = int(n_samples * self._outlier_ratio)
        
        mins = np.min(X, axis=0)
        maxs = np.max(X, axis=0)
        ranges = maxs - mins
        
        lower_bounds = mins - (ranges * self._outlier_margin)
        upper_bounds = maxs + (ranges * self._outlier_margin)
        
        outliers = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(n_outliers, n_features))

        return outliers

    def fit(self, X, y = None):
        outliers = self._generate_outliers(X)
        y = np.hstack([np.ones(len(X)), np.zeros(len(outliers))])
        print(y.shape)
        X = np.vstack([X, outliers])
        print(X.shape)
        X, y = utils.shuffle(X, y, random_state=self._random_state)
        self._estimator.fit(X, y)
        return self

    def predict(self, X):
        return self._estimator.predict(X)

    def predict_proba(self, X):
        return self._estimator.predict_proba(X)
