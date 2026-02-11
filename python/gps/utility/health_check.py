"""Health check and system monitoring utilities."""
import psutil
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class HealthChecker:
    """System health monitoring."""

    @staticmethod
    def check_memory() -> Dict[str, Any]:
        """Check memory usage."""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_percent': memory.percent,
            'status': 'healthy' if memory.percent < 90 else 'warning'
        }

    @staticmethod
    def check_cpu() -> Dict[str, Any]:
        """Check CPU usage."""
        cpu_percent = psutil.cpu_percent(interval=1)
        return {
            'usage_percent': cpu_percent,
            'cpu_count': psutil.cpu_count(),
            'status': 'healthy' if cpu_percent < 90 else 'warning'
        }

    @staticmethod
    def check_disk() -> Dict[str, Any]:
        """Check disk usage."""
        disk = psutil.disk_usage('/')
        return {
            'total_gb': disk.total / (1024**3),
            'used_gb': disk.used / (1024**3),
            'free_gb': disk.free / (1024**3),
            'used_percent': disk.percent,
            'status': 'healthy' if disk.percent < 90 else 'warning'
        }

    @staticmethod
    def check_all() -> Dict[str, Any]:
        """Run all health checks."""
        return {
            'memory': HealthChecker.check_memory(),
            'cpu': HealthChecker.check_cpu(),
            'disk': HealthChecker.check_disk(),
        }

    @staticmethod
    def log_health_status() -> None:
        """Log system health status."""
        health = HealthChecker.check_all()
        logger.info(f"Memory: {health['memory']['used_percent']:.1f}%")
        logger.info(f"CPU: {health['cpu']['usage_percent']:.1f}%")
        logger.info(f"Disk: {health['disk']['used_percent']:.1f}%")


def validate_training_data(X: np.ndarray, U: np.ndarray) -> bool:
    """Validate training data integrity."""
    checks = {
        'no_nans_X': not np.any(np.isnan(X)),
        'no_nans_U': not np.any(np.isnan(U)),
        'no_infs_X': not np.any(np.isinf(X)),
        'no_infs_U': not np.any(np.isinf(U)),
        'shape_match': X.shape[0] == U.shape[0] and X.shape[1] == U.shape[1],
    }

    if not all(checks.values()):
        logger.warning(f"Data validation failed: {checks}")
        return False

    return True
