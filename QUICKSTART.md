# GPS Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
make install
```

## Running Tests

```bash
# All tests
make test

# Specific test suites
make test-unit
make test-integration
make test-system

# With coverage
make coverage
```

## Development

```bash
# Lint code
make lint

# Check syntax
make check-syntax

# Clean build artifacts
make clean
```

## Usage Example

```python
from gps.utility.logging_config import setup_logging
from gps.utility.metrics import get_metrics

# Setup logging
setup_logging(level=logging.INFO, log_file='gps.log')

# Track metrics
metrics = get_metrics()
metrics.start_timer('training')
# ... your code ...
metrics.stop_timer('training')

# Get summary
summary = metrics.get_summary()
print(summary)
```

## Architecture

- **Algorithm**: Core GPS implementation
- **Cost**: Cost function evaluation
- **Dynamics**: Dynamics estimation
- **Policy**: Policy representation (Linear Gaussian, Neural Network)
- **Agent**: Environment interaction
- **TrajOpt**: Trajectory optimization (LQR, iLQG)

## Testing

- **Unit Tests** (117): Individual component testing
- **Integration Tests** (44): Component interaction testing
- **System Tests** (37): End-to-end workflow testing

Total: **198 tests**

## CI/CD

GitHub Actions automatically:
- Runs all tests on push
- Checks code quality
- Generates coverage reports
- Builds C++ components
- Deploys documentation

## Performance

Use profiling tools:
```python
from gps.utility.profiler import profile_function, PerformanceMonitor

@profile_function
def my_function():
    pass

with PerformanceMonitor("my_section"):
    # code here
    pass
```

## Monitoring

Check system health:
```python
from gps.utility.health_check import HealthChecker

health = HealthChecker.check_all()
print(health)
```
