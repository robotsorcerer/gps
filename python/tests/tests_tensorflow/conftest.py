"""Skip TF tests when TensorFlow is not installed."""
import pytest

try:
    import tensorflow  # noqa: F401
except ImportError:
    collect_ignore_glob = ["*.py"]
