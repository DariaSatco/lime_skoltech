import numpy as np
from sklearn.linear_model import lars_path, Ridge
from sklearn.metrics import pairwise_distances
from numpy import random
import copy

class Lime():

    def __init__(self, distance=None, kernel=None, kernel_width=3):
        if kernel is not None:
            self._kernel = kernel
        else:
            self._kernel = lambda x: gaussian_kernel(x, kernel_width)
        
        if distance is None:
            distance = 'euclidean' 
        self._distance = distance

       
    def weight_data(self, data, labels, weights):
        weighted_data = ((data - np.average(data, axis=0, weights=weights))
            * np.sqrt(weights.reshape(-1, 1)))
        weighted_labels = ((labels - np.average(labels, weights=weights))
            * np.sqrt(weights))
        return weighted_data, weighted_labels

    def feature_selection(self, weighted_data, weighted_labels, n_features):

        __, _, coeff = lars_path(weighted_data, weighted_labels, method='lasso')
        for i in range(len(coeff.T) - 1, 0, -1):
            cur_coeffs = coeff.T[i].nonzero()[0]
            if len(cur_coeffs) <= n_features:
                break
        return cur_coeffs

    def preturb_data(self, features, n_samples,  model, placeholder=None ):
        data = (np.random.randint(0, 2,
                                  n_samples * features.size).reshape(n_samples, -1))
        data[0, :] = 1
        result_data = np.zeros((n_samples, features.size))
        labels = np.zeros(n_samples)
        if placeholder is None:
            placeholder = np.mean(features)
        for i, sample in enumerate(data):
            temp = copy.deepcopy(features)
            temp[sample==0] = placeholder

            result_data[i, :] = temp
            labels[i] = model(temp)

        return result_data, labels


    def explain(self, features, model, n, n_features):
        preturbed_data, preturbed_labels = self.preturb_data(
            features, n, model, placeholder=0
        )
        distances = pairwise_distances(
            preturbed_data,
            features.reshape(1, -1),     
            metric=self._distance
        ).ravel()
        weights = self._kernel(distances)

        weighted_data, weighted_labels = self.weight_data(
            preturbed_data, preturbed_labels, weights
        )
        used_features = self.feature_selection(weighted_data,
        weighted_labels, n_features)
        
        explainer_model = Ridge()

        explainer_model.fit(preturbed_data[:, used_features], preturbed_labels,
                            sample_weight=weights)

        return (sorted(zip(used_features, explainer_model.coef_),
                      key=lambda a: np.abs(a[1]), reverse=True), 
                explainer_model.score(preturbed_data[:, used_features], preturbed_labels,
                                      sample_weight=weights))

def gaussian_kernel(x, sigma):
    return np.exp(-x ** 2 / sigma ** 2)






        
        

        