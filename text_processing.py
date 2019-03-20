import lime_sk

import numpy as np
import copy
import sklearn
from sklearn.utils import check_random_state

class TextExplanation(object):
    def __init__(self, instance):
        """
        Args:
            instance: numpy array (one sample) from a sparse matrix of TF-IDF features
        """
        self.instance = instance      # instance of vectorized textual data
        self.local_explanation = None # probability weights of considered class-label for each feature
        self.score = None             # scoring of goodness of g(x)

    def as_figure(self, class_name='0'):
        
        """Returns the explanation as a pyplot figure.
        Will throw an error if you don't have matplotlib installed
        Args:
            class_name: desired class label for which explanation was computed
        Returns:
            pyplot figure (horizontal barchart)
        """
        
        import matplotlib.pyplot as plt

        plt.rcParams.update({'font.size': 18})

        exp = self.local_explanation
        fig = plt.figure()
        vals = [x[1] for x in exp]
        names = [x[0] for x in exp]
        vals.reverse()
        names.reverse()
        colors = ['green' if x > 0 else 'red' for x in vals]
        pos = np.arange(len(exp)) + .5
        plt.barh(pos, vals, align='center', color=colors)
        plt.yticks(pos, names)
        title = 'Local explanation for class ' + class_name
        plt.title(title)
        plt.show()

class TextExplainer(object):
    """Explains predictions on textual data."""

    def __init__(self, distance='cosine', kernel_width=.25, random_state=None):
        """
        Args:
            label: label we want to explain
            distance: the distance metric to use for weights
            kernel_width: kernel width for the exponential kernel
            random_state: an integer or numpy.RandomState that is used to
                generate random numbers
        """
        
        self.random_state = check_random_state(random_state)
        self.base = lime_sk.Lime(kernel_width=kernel_width, distance=distance)

    def explain_prediction(self, instance, label, feature_names,
                           clf_model, reg_method='lasso', 
                           num_features=None,
                           num_samples=1000):
        """
        Generates explanations for a prediction.
        First, neighborhood data is generated by randomly deactivating features
        from the instance. Then locally weighted linear model is learned
        on this neighborhood data to explain the predicted class.
        ============================
        Args:
            instance: numpy array (one sample) from a sparse matrix of TF-IDF features
            label: label we want to explain
            feature_names: all TF-IDF features 
            clf_model: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities
            num_features: maximum number of features present in explanation (K in the paper)
            num_samples: size of the neighborhood to learn the linear model
        Returns:
            An Explanation object with the corresponding explanations and score for g(x)
        """
        
        features = np.ones(instance.nonzero()[1].shape)

        if num_features is None:
            num_features = features.size

        result = TextExplanation(instance)

        model = lambda x: self.sampling_pred(x, label, instance, clf_model)

        interim_result, result.score = self.base.explain(features=features, model=model, n=num_samples,
                                                         n_features=num_features, reg_method = reg_method)
        interim_result = np.array(interim_result)
        #print(interim_result)
        idx_mass = np.array([elem[0] for elem in interim_result]).astype(int)
        original_idx = instance.nonzero()[1][idx_mass]
        
        result.local_explanation = [(item, score) for item, score in zip(feature_names[original_idx], interim_result[:,1])]

        return result


    def sampling_pred(self, data, label, instance, clf_model):
        """
        gives prediction for the particular class = label
        for a sample generated by data vector
        ===========================
        Args:
                data: vector of shape = features.shape, which
                    contains 0 and 1, where 0 correspond to the
                    feature being excluded
                label: class label which we want to obtain prediction for
                instance: example to be explained
                clf_model: classifier
        Returns:
                probability of specified label for newly generated sample
        """

        # Obtain new sample
        new_instance = instance.copy()
        new_instance[instance.nonzero()] = np.multiply(new_instance[instance.nonzero()], data)
        prediction = clf_model.predict_proba(new_instance).ravel()[label]

        return prediction
