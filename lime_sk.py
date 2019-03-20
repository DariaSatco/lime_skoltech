import numpy as np
from sklearn.linear_model import lars_path, Ridge
from sklearn.metrics import pairwise_distances
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from numpy import random
import copy

class Lime():
    """
    implementation of basic LIME funtionality
    """
    def __init__(self, distance=None, kernel=None, kernel_width=3, random_state=None):
        if kernel is not None:
            self._kernel = kernel
        else:
            self._kernel = lambda x: gaussian_kernel(x, kernel_width)
        if distance is None:
            distance = 'euclidean'
        self._distance = distance
        self._random_state = random_state

    def _weight_data(self, data, labels, weights):
        """weight data for futher lasso regularization

        Arguments:
            data {np.array} -- samples
            labels {np.array} -- sample labels
            weights {np.array} -- sample weights

        Returns:
            Tuple(np.array, np.array) -- weighted data and weighted labels
        """

        weighted_data = ((data - np.average(data, axis=0, weights=weights))
            * np.sqrt(weights.reshape(-1, 1)))
        weighted_labels = ((labels - np.average(labels, weights=weights))
            * np.sqrt(weights))
        return weighted_data, weighted_labels

    def _feature_selection(self, data, labels, weights, n_features, method='lasso',
                           rf_n=200, rf_max_depth=5):
        """select most important features with lasso

        Arguments:
            weighted_data {np.array} -- weighted data
            weighted_labels {np.array} -- weighted labels
            n_features {[type]} -- number of features to preserve
            method {str} -- regularization method 'lasso' or 'random_forest'
            rf_n {int} -- number of estimators in random forest
            rf_max_depth -- max depth of estimators in random forest

        Returns:
            np.array -- indexes of the most important features
        """

        if method == 'lasso':
            weighted_data, weighted_labels = self._weight_data(data, labels, weights)
            __, _, coeff = lars_path(weighted_data, weighted_labels, method='lasso')
            for i in range(len(coeff.T) - 1, 0, -1):
                cur_coeffs = coeff.T[i].nonzero()[0]
                if len(cur_coeffs) <= n_features:
                    break
            return cur_coeffs
        elif method == 'random_forest':
            rf = RandomForestRegressor(n_estimators=rf_n, max_depth=rf_max_depth,
                                       random_state=self._random_state)
            rf.fit(data, labels, sample_weight=weights)
            feature_importances = rf.feature_importances_
        return [i for i, _ in sorted(enumerate(feature_importances), key=lambda x: x[1],
                                     reverse=True)[0:n_features]]

    def _perturb_data(self, features, n_samples,  model, placeholder=None ):
        """create smaples in the neighborhood of the explained instance

        Arguments:
            features {np.array} -- simplified features of theoriginal instance
                                    usually is an array of ones
            n_samples {np.array} -- number of random samples to create
            model {callable} -- takes array of the implified features and returns
                                probability of explained class

        Keyword Arguments:
            placeholder {object} -- value to replace features

        Returns:
            Tuple(np.array, np.array) -- perturbed data and labels in the neighborhood
                                         of the original data
        """

        data = (np.random.RandomState(self._random_state).randint(0, 2,
                                                                  n_samples * features.size,
                                                                 ).reshape(n_samples, -1))
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

    def explain(self, features, model, n, n_features=None, reg_method='lasso',
                rf_n=200, rf_max_depth=5, regressor='ridge', dt_max_depth=10):
        """explain instance

        Arguments:
            features {np.array} -- simplified features of theoriginal instance
                                    usually is an array of ones
            model {callable} -- takes array of the implified features and returns
                                probability of explained class
            n {int} -- number of samples to create
            n_features {int} -- number of features to preserve
            reg_method {str} -- regularization method 'lasso' or 'random_forest'
            rf_n {int} -- number of estimators in random forest
            rf_max_depth {int} -- max depth of estimators in random forest
            regressor {str} -- local estimator 'decision_tree' or 'ridge'
            dt_max_depth {int} -- decision tree max depth
        Returns:
            tuple -- most important features and coefficients, score of regressor model
        """

        perturbed_data, preturbed_labels = self._perturb_data(
            features, n, model, placeholder=0
        )
        distances = pairwise_distances(
            perturbed_data,
            features.reshape(1, -1),
            metric=self._distance
        ).ravel()
        weights = self._kernel(distances)

        if n_features == None:
            used_features = list(range(features.size))
        else:
            used_features = self._feature_selection(perturbed_data,
                                                    preturbed_labels,
                                                    weights, n_features,
                                                    reg_method, rf_n, 
                                                    rf_max_depth, )

        if regressor == 'ridge':
            explainer_model = Ridge(alpha=0.1, random_state=self._random_state)

            explainer_model.fit(perturbed_data[:, used_features], preturbed_labels,
                                sample_weight=weights)

            return (sorted(zip(used_features, explainer_model.coef_),
                            key=lambda a: np.abs(a[1]), reverse=True),
                    explainer_model.score(perturbed_data[:, used_features], preturbed_labels,
                                            sample_weight=weights))

        elif regressor =='decision_tree':
            explainer_model = DecisionTreeRegressor(max_depth=dt_max_depth)
            explainer_model.fit(perturbed_data[:, used_features], preturbed_labels,
                                sample_weight=weights)
            return (sorted(zip(used_features, explainer_model.feature_importances_),
                           key=lambda a: np.abs(a[1]), reverse=True),
                    explainer_model.score(perturbed_data[:, used_features], preturbed_labels,
                                          sample_weight=weights))

def gaussian_kernel(x, sigma):
    return np.exp(-(x ** 2) / (sigma ** 2))
