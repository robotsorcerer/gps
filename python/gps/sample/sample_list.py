""" This file defines the sample list wrapper and sample writers. """
from typing import List, Optional, Union
import pickle
import logging

import numpy as np
import numpy.typing as npt

from gps.proto.gps_pb2 import NOISE
from gps.sample.sample import Sample

LOGGER = logging.getLogger(__name__)


class SampleList(object):
    """ Class that handles writes and reads to sample data. """
    def __init__(self, samples: List[Sample]) -> None:
        self._samples: List[Sample] = samples

    def get_X(self, idx: Optional[List[int]] = None) -> npt.NDArray[np.float64]:
        """ Returns N x T x dX numpy array of states. """
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get_X() for i in idx])

    def get_U(self, idx: Optional[List[int]] = None) -> npt.NDArray[np.float64]:
        """ Returns N x T x dU numpy array of actions. """
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get_U() for i in idx])

    def get_V(self, idx: Optional[List[int]] = None) -> npt.NDArray[np.float64]:
        """ Returns N x T x dV numpy array of disturbances. """
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get_V() for i in idx])

    def get_noise(self, idx: Optional[List[int]] = None) -> npt.NDArray[np.float64]:
        """ Returns N x T x dU numpy array of noise generated during rollouts. """
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get(NOISE) for i in idx])

    def get_obs(self, idx: Optional[List[int]] = None) -> npt.NDArray[np.float64]:
        """ Returns N x T x dO numpy array of features. """
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get_obs() for i in idx])

    def get_obs_adv(self, idx=None):
        """ Returns N x T x dO numpy array of features. """
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get_obs_adv() for i in idx])

    def get_samples(self, idx: Optional[List[int]] = None) -> List[Sample]:
        """ Returns N sample objects. """
        if idx is None:
            idx = range(len(self._samples))
        return [self._samples[i] for i in idx]

    def num_samples(self) -> int:
        """ Returns number of samples. """
        return len(self._samples)

    # Convenience methods.
    def __len__(self):
        return self.num_samples()

    def __getitem__(self, idx):
        return self.get_samples([idx])[0]


class PickleSampleWriter(object):
    """ Pickles samples into data_file. """
    def __init__(self, data_file):
        self._data_file = data_file

    def write(self, samples):
        """ Write samples to data file. """
        with open(self._data_file, 'wb') as data_file:
            cPickle.dump(data_file, samples)

class SysOutWriter(object):
    """ Writes notifications to sysout on sample writes. """
    def __init__(self):
        pass

    def write(self, samples):
        """ Write number of samples to sysout. """
        LOGGER.debug('Collected %d samples', len(samples))
