import numpy as np
from sklearn import utils


class OutlierGenerator:
    def __init__(self, estimator, outliers_fn = "uniform", random_state = 42):
        self._estimator = estimator
        self._outliers_fn = outliers_fn
        self._outlier_ratio = 0.2
        self._outlier_margin = 0.1
        self._random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def generate_outliers(self, X):
        outliers_fn_map = {
            "uniform": self._generate_uniform_outliers,
            "normal": self._generate_normal_outliers,
            "extreme": self._generate_extreme_outliers
        }

        if self._outliers_fn not in outliers_fn_map:
            raise ValueError(f"Nieznana metoda outliers_fn: '{self._outliers_fn}'. Dostępne: {list(outliers_fn_map.keys())}")

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

    def _generate_normal_outliers(self, X):
        n_samples, n_features = X.shape
        n_outliers = int(n_samples * self._outlier_ratio)

        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        
        expanded_stds = stds * (1.0 + self._outlier_margin * 5.0)

        outliers = self._rng.normal(loc=means, scale=expanded_stds, size=(n_outliers, n_features))
        return outliers

    def _generate_extreme_outliers(self, X):
        X_array = np.asarray(X)
        
        n_samples, n_features = X_array.shape
        n_outliers = int(n_samples * self._outlier_ratio)

        mins = np.min(X_array, axis=0)
        maxs = np.max(X_array, axis=0)
        ranges = maxs - mins

        outliers = np.zeros((n_outliers, n_features))
        
        for i in range(n_features):
            sides = self._rng.choice([-1, 1], size=n_outliers)
            
            lower_bound_extreme = mins[i] - (ranges[i] * self._outlier_margin)
            upper_bound_extreme = maxs[i] + (ranges[i] * self._outlier_margin)
            
            lower_points = self._rng.uniform(lower_bound_extreme, mins[i], size=n_outliers)
            upper_points = self._rng.uniform(maxs[i], upper_bound_extreme, size=n_outliers)
            
            outliers[:, i] = np.where(sides == -1, lower_points, upper_points)

        return outliers

    def fit(self, X, y = None, outliers = None):
        outliers = outliers or self.generate_outliers(X)
        y = np.hstack([np.ones(len(X)), np.zeros(len(outliers))])
        X = np.vstack([X, outliers])
        X, y = utils.shuffle(X, y, random_state=self._random_state)
        self._estimator.fit(X, y)
        return self

    def predict(self, X):
        return self._estimator.predict(X)

    def predict_proba(self, X):
        return self._estimator.predict_proba(X)
