class OutlierGenerator:
    def __init__(self, estimator):
        self._estimator = estimator

    # TODO
    def _generate_outliers(self, X):
        pass

    def fit(self, X, y):
        return self._estimator.fit(X, y)

    def predict(self, X):
        return self._estimator.predict(X)

    def predict_proba(self, X):
        return self._estimator.predict_proba(X)
