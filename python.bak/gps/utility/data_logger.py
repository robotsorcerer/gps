""" This file defines the data logger. """
from __future__ import annotations

import logging
import pickle
from typing import Any


LOGGER = logging.getLogger(__name__)


class DataLogger:
    """
    This class pickles data into files and unpickles data from files.
    TODO: Handle logging text to terminal, GUI text, and/or log file at
        DEBUG, INFO, WARN, ERROR, FATAL levels.
    TODO: Handle logging data to terminal, GUI text/plots, and/or data
          files.
    """

    def pickle(self, filename: str, data: Any) -> None:
        """ Pickle data into file specified by filename. """
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def unpickle(self, filename: str) -> Any | None:
        """ Unpickle data from file specified by filename. """
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except OSError:
            LOGGER.debug('Unpickle error. Cannot find file: %s', filename)
            return None
        except (pickle.UnpicklingError, EOFError, ValueError, TypeError,
                MemoryError) as exc:
            LOGGER.debug('Unpickle error. Corrupted file %s: %s', filename, exc)
            return None
