import math
import numpy as np
from collections import defaultdict

class NaiveBayes:
    def __init__(self):
        self.class_priors = {}
        self.feature_probs = defaultdict(dict)
        self.feature_params = defaultdict(dict)  # For continuous features (mean, variance)

    def fit(self, X, y):
        """ Fit Naive Bayes model to the training data. """
        n_samples, n_features = X.shape
        classes = np.unique(y)
        # Class priors and feature probabilities
        for cls in classes:
            X_cls = X[y == cls]
            self.class_priors[cls] = len(X_cls) / n_samples
            for feature_idx in range(n_features):
                feature_values = X_cls[:, feature_idx]
                if self._is_discrete(feature_values):
                    unique_values, counts = np.unique(feature_values, return_counts=True)
                    self.feature_probs[feature_idx][cls] = {val: count / len(X_cls) for val, count in zip(unique_values, counts)}
                else:
                    mean, var = np.mean(feature_values), np.var(feature_values)
                    self.feature_params[feature_idx][cls] = (mean, var)

    def predict(self, X):
        probabilities = self.predict_probability(X)
        return np.array([max(probs, key=probs.get) for probs in probabilities])

    def predict_probability(self, X):
        n_samples, n_features = X.shape
        probabilities = []
        for i in range(n_samples):
            sample_probs = {}
            for cls in self.class_priors:
                cls_prob = math.log(self.class_priors[cls])
                for feature_idx in range(n_features):
                    feature_value = X[i, feature_idx]
                    if feature_idx in self.feature_probs:
                        cls_prob += math.log(self.feature_probs[feature_idx][cls].get(feature_value, 1e-6))
                    else:
                        mean, var = self.feature_params[feature_idx][cls]
                        cls_prob += self._gaussian_prob(feature_value, mean, var)
                sample_probs[cls] = cls_prob
            probabilities.append(sample_probs)
        return probabilities

    def _is_discrete(self, values):
        return len(np.unique(values)) < 10

    def _gaussian_prob(self, x, mean, var):
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var)))
        return (1 / math.sqrt(2 * math.pi * var)) * exponent
