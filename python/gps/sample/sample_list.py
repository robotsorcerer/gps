""" This file defines the sample list wrapper and sample writers. """
from __future__ import annotations

import logging
import pickle
from typing import Any

import numpy as np

from gps.proto.gps_pb2 import NOISE

LOGGER = logging.getLogger(__name__)


class SampleList:
    """ Class that handles writes and reads to sample data. """

    def __init__(self, samples: list, samples_adv: list | None = None) -> None:
        self._samples = samples
        self._samples_adv = samples_adv if samples_adv is not None else []

    def get_X(self, idx: list[int] | None = None) -> np.ndarray:
        """ Returns N x T x dX numpy array of states. """
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get_X() for i in idx])

    def get_U(self, idx: list[int] | None = None) -> np.ndarray:
        """ Returns N x T x dU numpy array of actions. """
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get_U() for i in idx])

    def get_V(self, idx: list[int] | None = None) -> np.ndarray:
        """ Returns N x T x dV numpy array of disturbances. """
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get_V() for i in idx])

    def get_noise(self, idx: list[int] | None = None) -> np.ndarray:
        """ Returns N x T x dU numpy array of noise generated during rollouts. """
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get(NOISE) for i in idx])

    def get_obs(self, idx: list[int] | None = None) -> np.ndarray:
        """ Returns N x T x dO numpy array of features. """
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get_obs() for i in idx])

    def get_obs_adv(self, idx: list[int] | None = None) -> np.ndarray:
        """ Returns N x T x dO numpy array of features. """
        if idx is None:
            idx = range(len(self._samples))
        return np.asarray([self._samples[i].get_obs_adv() for i in idx])

    def get_samples(self, idx: list[int] | None = None) -> list:
        """ Returns N sample objects. """
        if idx is None:
            idx = range(len(self._samples))
        return [self._samples[i] for i in idx]

    def num_samples(self) -> int:
        """ Returns number of samples. """
        return len(self._samples)

    # Convenience methods.
    def __len__(self) -> int:
        return self.num_samples()

    def __getitem__(self, idx: int) -> Any:
        return self.get_samples([idx])[0]


class PickleSampleWriter:
    """ Pickles samples into data_file. """

    def __init__(self, data_file: str) -> None:
        self._data_file = data_file

    def write(self, samples: list) -> None:
        """ Write samples to data file. """
        with open(self._data_file, 'wb') as data_file:
            pickle.dump(samples, data_file)


class SysOutWriter:
    """ Writes notifications to sysout on sample writes. """

    def write(self, samples: list) -> None:
        """ Write number of samples to sysout. """
        LOGGER.debug('Collected %d samples', len(samples))
