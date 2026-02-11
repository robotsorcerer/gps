""" This file defines the base agent class. """
import abc
import copy
from typing import Dict, List, Optional, Any

import numpy.typing as npt

from gps.agent.config import AGENT
from gps.proto.gps_pb2 import ACTION
from gps.sample.sample_list import SampleList


class Agent(object):
    """
    Agent superclass. The agent interacts with the environment to
    collect samples.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams: Dict[str, Any]) -> None:
        config = copy.deepcopy(AGENT)
        config.update(hyperparams)
        self._hyperparams: Dict[str, Any] = config
        # Store samples, along with size/index information for samples.
        self._samples: List[List[Any]] = [[] for _ in range(self._hyperparams['conditions'])]
        self._samples_adv: List[List[Any]] = [[] for _ in range(self._hyperparams['conditions'])]
        self.T: int = self._hyperparams['T']
        self.dU: int = self._hyperparams['sensor_dims'][ACTION]
        self.dV: int = self._hyperparams['sensor_dims'][ACTION] #if self._hyperparams['mode'] == 'robust' else None

        self.x_data_types: List[Any] = self._hyperparams['state_include']
        self.obs_data_types: List[Any] = self._hyperparams['obs_include']
        if 'meta_include' in self._hyperparams:
            self.meta_data_types: List[Any] = self._hyperparams['meta_include']
        else:
            self.meta_data_types: List[Any] = []

        # List of indices for each data type in state X.
        self._state_idx: List[List[int]]
        self._state_idx, i = [], 0
        for sensor in self.x_data_types:
            dim = self._hyperparams['sensor_dims'][sensor]
            self._state_idx.append(list(range(i, i+dim)))
            i += dim
        self.dX: int = i

        # List of indices for each data type in observation.
        self._obs_idx: List[List[int]]
        self._obs_idx, i = [], 0
        for sensor in self.obs_data_types:
            dim = self._hyperparams['sensor_dims'][sensor]
            self._obs_idx.append(list(range(i, i+dim)))
            i += dim
        self.dO: int = i

        # List of indices for each data type in meta data.
        self._meta_idx: List[List[int]]
        self._meta_idx, i = [], 0
        for sensor in self.meta_data_types:
            dim = self._hyperparams['sensor_dims'][sensor]
            self._meta_idx.append(list(range(i, i+dim)))
            i += dim
        self.dM: int = i

        self._x_data_idx: Dict[Any, List[int]] = {d: i for d, i in zip(self.x_data_types,
                                                 self._state_idx)}
        self._obs_data_idx: Dict[Any, List[int]] = {d: i for d, i in zip(self.obs_data_types,
                                                   self._obs_idx)}
        self._meta_data_idx: Dict[Any, List[int]] = {d: i for d, i in zip(self.meta_data_types,
                                                   self._meta_idx)}

    @abc.abstractmethod
    def sample(self, policy: Any, condition: int, verbose: bool = True,
               save: bool = True, noisy: bool = True) -> Any:
        """
        Draw a sample from the environment, using the specified policy
        and under the specified condition, with or without noise.
        """
        raise NotImplementedError("Must be implemented in subclass.")

    def reset(self, condition: int) -> None:
        """ Reset environment to the specified condition. """
        pass  # May be overridden in subclass.

    def get_samples(self, condition: int, start: int = 0, end: Optional[int] = None) -> SampleList:
        """
        Return the requested samples based on the start and end indices.
        Args:
            start: Starting index of samples to return.
            end: End index of samples to return.
        """
        return (SampleList(self._samples[condition][start:]) if end is None
                else SampleList(self._samples[condition][start:end]))

    def get_samples_adv(self, condition: int, start: int = 0, end: Optional[int] = None) -> SampleList:
        """
        Return the requested samples based on the start and end indices.
        Args:
            start: Starting index of samples to return.
            end: End index of samples to return.
        """
        return (SampleList(self._samples[condition][start:], self._samples_adv[condition][start:]) if end is None
    else SampleList(self._samples[condition][start:end], self._samples_adv[condition][start:end]))

    def clear_samples(self, condition: Optional[int] = None) -> None:
        """
        Reset the samples for a given condition, defaulting to all conditions.
        Args:
            condition: Condition for which to reset samples.
        """
        if condition is None:
            self._samples = [[] for _ in range(self._hyperparams['conditions'])]
        else:
            self._samples[condition] = []

    def clear_samples_adv(self, condition: Optional[int] = None) -> None:
        """
        Reset the samples for a given condition, defaulting to all conditions.
        Args:
            condition: Condition for which to reset samples.
        """
        if condition is None:
            self._samples_adv = [[] for _ in range(self._hyperparams['conditions'])]
        else:
            self._samples_adv[condition] = []

    def delete_last_sample(self, condition: int) -> None:
        """ Delete the last sample from the specified condition. """
        self._samples[condition].pop()

    def get_idx_x(self, sensor_name: Any) -> List[int]:
        """
        Return the indices corresponding to a certain state sensor name.
        Args:
            sensor_name: The name of the sensor.
        """
        return self._x_data_idx[sensor_name]

    def get_idx_obs(self, sensor_name: Any) -> List[int]:
        """
        Return the indices corresponding to a certain observation sensor name.
        Args:
            sensor_name: The name of the sensor.
        """
        return self._obs_data_idx[sensor_name]

    def pack_data_obs(self, existing_mat: npt.NDArray, data_to_insert: npt.NDArray,
                      data_types: List[Any], axes: Optional[List[int]] = None) -> None:
        """
        Update the observation matrix with new data.
        Args:
            existing_mat: Current observation matrix.
            data_to_insert: New data to insert into the existing matrix.
            data_types: Name of the sensors to insert data for.
            axes: Which axes to insert data. Defaults to the last axes.
        """
        num_sensor = len(data_types)
        if axes is None:
            # If axes not specified, assume indexing on last dimensions.
            axes = list(range(-1, -num_sensor - 1, -1))
        else:
            # Make sure number of sensors and axes are consistent.
            if num_sensor != len(axes):
                raise ValueError(
                    'Length of sensors (%d) must equal length of axes (%d)',
                    num_sensor, len(axes)
                )

        # Shape checks.
        insert_shape = list(existing_mat.shape)
        for i in range(num_sensor):
            # Make sure to slice along X.
            if existing_mat.shape[axes[i]] != self.dO:
                raise ValueError('Axes must be along an dX=%d dimensional axis',
                                 self.dO)
            insert_shape[axes[i]] = len(self._obs_data_idx[data_types[i]])
        if tuple(insert_shape) != data_to_insert.shape:
            raise ValueError('Data has shape %s. Expected %s',
                             data_to_insert.shape, tuple(insert_shape))

        # Actually perform the slice.
        index = [slice(None) for _ in range(len(existing_mat.shape))]
        for i in range(num_sensor):
            index[axes[i]] = slice(self._obs_data_idx[data_types[i]][0],
                                   self._obs_data_idx[data_types[i]][-1] + 1)
        existing_mat[index] = data_to_insert

    def pack_data_meta(self, existing_mat: npt.NDArray, data_to_insert: npt.NDArray,
                       data_types: List[Any], axes: Optional[List[int]] = None) -> None:
        """
        Update the meta data matrix with new data.
        Args:
            existing_mat: Current meta data matrix.
            data_to_insert: New data to insert into the existing matrix.
            data_types: Name of the sensors to insert data for.
            axes: Which axes to insert data. Defaults to the last axes.
        """
        num_sensor = len(data_types)
        if axes is None:
            # If axes not specified, assume indexing on last dimensions.
            axes = list(range(-1, -num_sensor - 1, -1))
        else:
            # Make sure number of sensors and axes are consistent.
            if num_sensor != len(axes):
                raise ValueError(
                    'Length of sensors (%d) must equal length of axes (%d)',
                    num_sensor, len(axes)
                )

        # Shape checks.
        insert_shape = list(existing_mat.shape)
        for i in range(num_sensor):
            # Make sure to slice along X.
            if existing_mat.shape[axes[i]] != self.dM:
                raise ValueError('Axes must be along an dX=%d dimensional axis',
                                 self.dM)
            insert_shape[axes[i]] = len(self._meta_data_idx[data_types[i]])
        if tuple(insert_shape) != data_to_insert.shape:
            raise ValueError('Data has shape %s. Expected %s',
                             data_to_insert.shape, tuple(insert_shape))

        # Actually perform the slice.
        index = [slice(None) for _ in range(len(existing_mat.shape))]
        for i in range(num_sensor):
            index[axes[i]] = slice(self._meta_data_idx[data_types[i]][0],
                                   self._meta_data_idx[data_types[i]][-1] + 1)
        existing_mat[index] = data_to_insert

    def pack_data_x(self, existing_mat: npt.NDArray, data_to_insert: npt.NDArray,
                    data_types: List[Any], axes: Optional[List[int]] = None) -> None:
        """
        Update the state matrix with new data.
        Args:
            existing_mat: Current state matrix.
            data_to_insert: New data to insert into the existing matrix.
            data_types: Name of the sensors to insert data for.
            axes: Which axes to insert data. Defaults to the last axes.
        """
        num_sensor = len(data_types)
        if axes is None:
            # If axes not specified, assume indexing on last dimensions.
            axes = list(range(-1, -num_sensor - 1, -1))
        else:
            # Make sure number of sensors and axes are consistent.
            if num_sensor != len(axes):
                raise ValueError(
                    'Length of sensors (%d) must equal length of axes (%d)',
                    num_sensor, len(axes)
                )

        # Shape checks.
        insert_shape = list(existing_mat.shape)
        for i in range(num_sensor):
            # Make sure to slice along X.
            if existing_mat.shape[axes[i]] != self.dX:
                raise ValueError('Axes must be along an dX=%d dimensional axis',
                                 self.dX)
            insert_shape[axes[i]] = len(self._x_data_idx[data_types[i]])
        if tuple(insert_shape) != data_to_insert.shape:
            raise ValueError('Data has shape %s. Expected %s',
                             data_to_insert.shape, tuple(insert_shape))

        # Actually perform the slice.
        index = [slice(None) for _ in range(len(existing_mat.shape))]
        for i in range(num_sensor):
            index[axes[i]] = slice(self._x_data_idx[data_types[i]][0],
                                   self._x_data_idx[data_types[i]][-1] + 1)
        existing_mat[index] = data_to_insert

    def unpack_data_x(self, existing_mat: npt.NDArray, data_types: List[Any],
                      axes: Optional[List[int]] = None) -> npt.NDArray:
        """
        Returns the requested data from the state matrix.
        Args:
            existing_mat: State matrix to unpack from.
            data_types: Names of the sensor to unpack.
            axes: Which axes to unpack along. Defaults to the last axes.
        """
        num_sensor = len(data_types)
        if axes is None:
            # If axes not specified, assume indexing on last dimensions.
            axes = list(range(-1, -num_sensor - 1, -1))
        else:
            # Make sure number of sensors and axes are consistent.
            if num_sensor != len(axes):
                raise ValueError(
                    'Length of sensors (%d) must equal length of axes (%d)',
                    num_sensor, len(axes)
                )

        # Shape checks.
        for i in range(num_sensor):
            # Make sure to slice along X.
            if existing_mat.shape[axes[i]] != self.dX:
                raise ValueError('Axes must be along an dX=%d dimensional axis',
                                 self.dX)

        # Actually perform the slice.
        index = [slice(None) for _ in range(len(existing_mat.shape))]
        for i in range(num_sensor):
            index[axes[i]] = slice(self._x_data_idx[data_types[i]][0],
                                   self._x_data_idx[data_types[i]][-1] + 1)
        return existing_mat[index]
