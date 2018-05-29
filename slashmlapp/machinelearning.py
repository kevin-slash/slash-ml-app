""" Machine learning public API
"""

from slashml.naive_bayes.naive_bayes_template import NaiveBayesTemplate
from slashml.algorithm.neural_network.main_ann import MainANN


class MachineLearning(object):
    """ Machine learning factory

    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def make_naivebayes(self):
        """ return naive_bayes_temmplate

        """

        return NaiveBayesTemplate(**self.kwargs)

    def make_nearalnetworks(self):
        """ return naive_bayes_temmplate

        """

        return MainANN(hidden_layer_sizes=(100,), learning_rate=0.5, max_iter=200, momentum=0.2,\
     random_state=1, activation='logistic', **self.kwargs)
