"""
Translates gps_agent_pkg/src/sample.cpp + include/gps_agent_pkg/sample.h.

The C++ Sample class is the *robot-side* time-indexed sensor buffer.
It is distinct from the algorithm-side python/gps/sample/sample.py.
We name it ControllerSample to avoid the collision.

Data layout mirroring C++:
  - SampleDataFormatEigenVector → stored as np.ndarray of shape (size,)
  - SampleDataFormatEigenMatrix → stored as np.ndarray of shape (rows, cols)
  - get_data_vec flattens matrices row-major (matches C++ transpose+column-major flatten)
"""
from __future__ import annotations

import numpy as np
from typing import Optional

# Mirror C++ enum SampleDataFormat
SAMPLE_FORMAT_VECTOR = "vector"
SAMPLE_FORMAT_MATRIX = "matrix"
SAMPLE_FORMAT_DOUBLE = "double"


class ControllerSample:
    """
    Robot-side time-indexed sensor buffer, mirroring the C++ Sample class.

    Parameters
    ----------
    T : int
        Maximum trial length (number of timesteps).
    """

    MAX_T: int = 2000

    def __init__(self, T: int) -> None:
        self._T = T
        # dtype_int → list of Optional[np.ndarray], length T
        self._data: dict[int, list[Optional[np.ndarray]]] = {}
        # dtype_int → total element count (rows * cols)
        self._sizes: dict[int, int] = {}
        # dtype_int → (rows, cols) for matrices, (size,) for vectors
        self._shapes: dict[int, tuple] = {}
        # dtype_int → SAMPLE_FORMAT_* constant
        self._formats: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def set_meta_data(
        self,
        dtype: int,
        rows: int,
        cols: int = 1,
        fmt: str = SAMPLE_FORMAT_VECTOR,
        meta: Optional[dict] = None,
    ) -> None:
        """
        Configure storage for *dtype* and pre-allocate zero arrays.

        Mirrors C++ set_meta_data(type, rows, cols, format, opts).
        For vectors call with cols=1 (or use the 1-arg shortcut).
        """
        total = rows * cols
        self._sizes[dtype] = total
        self._formats[dtype] = fmt

        if fmt == SAMPLE_FORMAT_VECTOR:
            self._shapes[dtype] = (rows,)
            self._data[dtype] = [np.zeros(rows, dtype=np.float64) for _ in range(self._T)]
        elif fmt == SAMPLE_FORMAT_MATRIX:
            self._shapes[dtype] = (rows, cols)
            self._data[dtype] = [np.zeros((rows, cols), dtype=np.float64) for _ in range(self._T)]
        else:
            # SAMPLE_FORMAT_DOUBLE or unknown — store scalar zeros
            self._shapes[dtype] = (total,)
            self._data[dtype] = [np.zeros(total, dtype=np.float64) for _ in range(self._T)]

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------

    def set_data(self, t: int, dtype: int, array: np.ndarray) -> None:
        """
        Store *array* at timestep *t* for data type *dtype*.

        Mirrors C++ set_data(t, type, SampleVariant, size, format).
        """
        if t >= self._T:
            return
        if dtype not in self._data:
            return
        self._data[dtype][t] = np.asarray(array, dtype=np.float64)

    def set_data_vector(
        self,
        t: int,
        dtype: int,
        array: np.ndarray,
    ) -> None:
        """
        Validated store — checks that the array matches pre-allocated shape.

        Mirrors C++ set_data_vector(t, type, double*, rows, cols, format).
        """
        if t >= self._T:
            return
        if dtype not in self._data:
            return
        arr = np.asarray(array, dtype=np.float64)
        stored = self._data[dtype][t]
        if arr.size != stored.size:
            return  # shape mismatch — skip silently (mirrors C++ ROS_ERROR + return)
        self._data[dtype][t] = arr.reshape(stored.shape)

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_data_vec(self, t: int, dtypes: list[int]) -> np.ndarray:
        """
        Concatenate data from multiple *dtypes* at a single timestep *t*.

        Mirrors C++ get_data(t, X, datatypes_vector).
        Matrix data is flattened row-major (equivalent to C++ transpose + column-major).
        """
        if t >= self._T:
            total = sum(self._sizes.get(d, 0) for d in dtypes)
            return np.zeros(total, dtype=np.float64)

        parts: list[np.ndarray] = []
        for d in dtypes:
            if d not in self._data or self._data[d][t] is None:
                size = self._sizes.get(d, 0)
                parts.append(np.zeros(size, dtype=np.float64))
            else:
                arr = self._data[d][t]
                parts.append(arr.flatten())  # row-major (matches C++ transpose+col-major)
        return np.concatenate(parts) if parts else np.array([], dtype=np.float64)

    def get_data(self, T: int, dtype: int) -> np.ndarray:
        """
        Flatten first *T* timesteps for a single *dtype* into one vector.

        Mirrors C++ get_data(T, Eigen::VectorXd&, SampleType).
        """
        size = self._sizes.get(dtype, 0)
        out = np.zeros(T * size, dtype=np.float64)
        if dtype not in self._data:
            return out
        dtypes_list = [dtype]
        for t in range(T):
            chunk = self.get_data_vec(t, dtypes_list)
            out[t * size:(t + 1) * size] = chunk
        return out

    def get_available_dtypes(self) -> list[int]:
        """Return list of dtype integers that have metadata registered."""
        return [d for d, sz in self._sizes.items() if sz >= 0]

    def get_shape(self, dtype: int) -> list[int]:
        """
        Return the shape dimensions for *dtype*.

        Mirrors C++ get_shape(SampleType, vector<int>&):
          - vector → [size]
          - matrix → [rows, cols]
        """
        if dtype not in self._shapes:
            return []
        return list(self._shapes[dtype])

    def get_T(self) -> int:
        return self._T
