# (C) Mathieu Blondel 2012

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.extmath import safe_sparse_dot


class LbfgsNNLS(BaseEstimator, RegressorMixin):

    def __init__(self, tol=1e-6, callback=None):
        self.tol = tol
        self.callback = callback

    def fit(self, X, y):
        n_features = X.shape[1]
        def f(w, *args):
            return np.sum(np.power((safe_sparse_dot(X, w) - y), 2))

        def fprime(w, *args):
            if self.callback is not None:
                self.coef_ = w
                self.callback(self)
            return 2 * np.ravel(safe_sparse_dot(X.T, (safe_sparse_dot(X, w) - y).T))

        coef0 = np.zeros(n_features, dtype=np.float64)
        w, f, d = fmin_l_bfgs_b(f, x0=coef0, fprime=fprime, pgtol=self.tol,
                                bounds=[(0, None)] * n_features)
        self.coef_ = w

        return self

    def n_nonzero(self, percentage=False):
        nz = np.sum(self.coef_ != 0)

        if percentage:
            nz /= float(self.coef_.shape[0])

        return nz

    def predict(self, X):
        return safe_sparse_dot(X, self.coef_)
