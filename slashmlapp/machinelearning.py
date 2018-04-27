""" Machine learning public API
"""

from slashml.naive_bayes.naive_bayes_template import NaiveBayesTemplate

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

        return NaiveBayesTemplate(**self.kwargs)
