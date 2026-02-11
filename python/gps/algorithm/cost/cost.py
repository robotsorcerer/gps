""" This file defines the base cost class. """
import abc


class Cost(abc.ABC):
    """ Cost superclass. """

    def __init__(self, hyperparams):
        self._hyperparams = hyperparams

    @abc.abstractmethod
    def eval(self, sample):
        """
        Evaluate cost function and derivatives.
        Args:
            sample:  A single sample.
        """
        raise NotImplementedError("Must be implemented in subclass.")
