# Iterative Migration Strategy with Docker-Based Validation
**Date:** 2026-02-10
**Approach:** Incremental, validated, Docker-containerized testing

---

## Overview

This document outlines the **iterative migration strategy** with comprehensive validation at each step. We'll use Docker to provide ROS 1.x environment since it's not available on the host.

---

## Phase Structure: Micro-Iterations with Validation

Each change follows this cycle:
1. **Modify** a small, testable unit
2. **Test** in Docker container
3. **Validate** against Python 2.7 baseline (if applicable)
4. **Commit** with detailed message
5. **Document** any behavioral changes

---

## Docker Strategy

### Three-Container Approach

**Container 1: Python 2.7 Baseline**
```dockerfile
FROM ubuntu:16.04
RUN apt-get update && apt-get install -y python2.7 python-pip
# Install original dependencies
```
Purpose: Run original code, generate baseline outputs

**Container 2: Python 3.11 + ROS Noetic**
```dockerfile
FROM ros:noetic-robot
RUN apt-get update && apt-get install -y python3.11 python3-pip
# Install modernized dependencies
```
Purpose: Run migrated code, compare with baseline

**Container 3: Build & Test Environment**
```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
RUN apt-get update && apt-get install -y \
    build-essential cmake \
    python3.11 python3-pip \
    ros-noetic-desktop-full \
    nvidia-docker2
```
Purpose: Full integration testing with GPU support

---

## Phase 1: Python Core Modernization (Weeks 1-3)

### Iteration 1.1: Fix Dict Methods (Day 1)
**Files:** 5 files with `.iteritems()`, `.iterkeys()`, `.itervalues()`

**Changes:**
```python
# BEFORE (Python 2)
for key, value in dict.iteritems():

# AFTER (Python 3)
for key, value in dict.items():
```

**Validation:**
- Unit test: Compare dict iteration outputs
- Integration test: Run simple GUI test
- Performance: Measure iteration speed (should be similar)

**Files to modify:**
1. `python/gps/gui/config.py` (3 occurrences)
2. `python/gps/gui/ps3_config.py` (2 occurrences)
3. `python/gps/gui/action_panel.py` (2 occurrences)
4. `python/gps/algorithm/policy_opt/tf_utils.py` (2 occurrences)

**Commit:** `"fix: Replace .iteritems() with .items() for Python 3 compatibility"`

---

### Iteration 1.2: Replace xrange with range (Day 1)
**Files:** 8 occurrences

**Changes:**
```python
# BEFORE
for i in xrange(1000):

# AFTER
for i in range(1000):
```

**Validation:**
- Memory profiling (range creates list in Py2, iterator in Py3)
- Performance benchmark on large loops
- Ensure no unexpected side effects

**Commit:** `"fix: Replace xrange with range for Python 3"`

---

### Iteration 1.3: Fix Integer Division (Day 2)
**Critical for numerical correctness!**

**Strategy:**
1. Add `from __future__ import division` to ALL Python files
2. Audit every `/` operator for intended behavior
3. Replace with `//` where integer division intended

**High-risk files:**
- `python/gps/algorithm/traj_opt/*` (trajectory computations)
- `python/gps/algorithm/dynamics/*` (dynamics fitting)
- `python/gps/algorithm/cost/*` (cost computations)

**Validation:**
- Run end-to-end trajectory optimization
- Compare numerical outputs with Python 2.7 baseline
- Tolerance: < 1e-10 for floating point ops

**Commit:** `"fix: Add future division, fix integer division for Python 3"`

---

### Iteration 1.4: Add Type Hints (Days 3-5)
**Incremental approach:** One module at a time

**Priority order:**
1. Core data structures (Sample, SampleList)
2. Algorithm interfaces (Algorithm, Cost, TrajOpt)
3. Policy optimization (PolicyOpt, TfPolicy)
4. Agent implementations
5. GUI and utilities (lower priority)

**Example:**
```python
# BEFORE
def fit_dynamics(self, X, U):

# AFTER
from typing import Tuple
import numpy.typing as npt

def fit_dynamics(
    self,
    X: npt.NDArray[np.float64],
    U: npt.NDArray[np.float64]
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
```

**Validation:**
- Run `mypy --strict` after each module
- Fix type errors incrementally
- Ensure no runtime changes

**Commit (per module):** `"feat: Add type hints to gps.algorithm.dynamics"`

---

### Iteration 1.5: Numpy API Updates (Days 6-7)

**Changes for numpy 1.20+ compatibility:**

```python
# BEFORE (deprecated)
np.int, np.float

# AFTER
np.int64, np.float64

# BEFORE
arr = np.array([1, 2, 3], dtype=np.int)

# AFTER
arr = np.array([1, 2, 3], dtype=np.int64)
```

**Validation:**
- Run all algorithm tests
- Check for FutureWarnings
- Numerical validation on matrix operations

**Commit:** `"fix: Update numpy API for 1.20+ compatibility"`

---

## Phase 2: TensorFlow 2.x Migration (Weeks 3-5)

### Strategy: Hybrid TF1/TF2 During Transition

**Step 2.1: Add TF2 Compatibility Layer (Day 8)**
```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```

**Step 2.2: Migrate Session to Function (Days 9-12)**
```python
# BEFORE (TF1)
sess = tf.Session()
sess.run(init_op)
result = sess.run(output, feed_dict={input: data})

# AFTER (TF2)
@tf.function
def inference(input_data):
    return model(input_data)

result = inference(data)
```

**Critical files:**
- `python/gps/algorithm/policy_opt/policy_opt_tf.py`
- `python/gps/algorithm/policy_opt/tf_utils.py`
- `python/gps/algorithm/policy/tf_policy.py`

**Validation:**
- Run policy optimization on simple task
- Compare loss curves with baseline
- Check GPU utilization (should improve with TF2)

---

## Phase 3: C++ Modernization to C++17/20 (Weeks 5-7)

### Iteration 3.1: Update CMake Configuration (Day 15)

**File:** `gps_agent_pkg/CMakeLists.txt`

```cmake
# BEFORE
OPTION(ENABLE_CXX11 "Enable C++11 support" ON)

# AFTER
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add compiler flags
add_compile_options(
    -Wall -Wextra -Wpedantic
    -Wthread-safety  # Clang thread safety analysis
)
```

**Validation:**
- Compile all C++ files
- Fix compilation errors
- Run catkin build

**Commit:** `"build: Update CMake to C++20 standard"`

---

### Iteration 3.2: Replace NULL with nullptr (Day 16)

**Files:** 17 C++ files, 9 occurrences

```cpp
// BEFORE
if (ptr == NULL) { ... }

// AFTER
if (ptr == nullptr) { ... }
```

**Validation:**
- Compile with `-Wzero-as-null-pointer-constant`
- Ensure no warnings

**Commit:** `"refactor: Replace NULL with nullptr (C++11)"`

---

### Iteration 3.3: Add Move Semantics (Days 17-19)

**Target:** Large data structures (Sample, sensor data)

```cpp
// BEFORE
class Sample {
    std::vector<double> data_;
    Sample(const Sample& other) : data_(other.data_) {}  // Copy
};

// AFTER
class Sample {
    std::vector<double> data_;

    // Copy constructor
    Sample(const Sample& other) : data_(other.data_) {}

    // Move constructor (C++11)
    Sample(Sample&& other) noexcept : data_(std::move(other.data_)) {}

    // Move assignment (C++11)
    Sample& operator=(Sample&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
        }
        return *this;
    }
};
```

**Validation:**
- Profiling: Measure data copy overhead before/after
- Memory analysis: Ensure no leaks
- Functional: Run sample collection tests

**Commit (per class):** `"perf: Add move semantics to Sample class"`

---

### Iteration 3.4: Fix Race Conditions (Days 20-23)

**Critical:** Thread safety in ROS callbacks

**Problem:**
```cpp
// robotplugin.cpp - RACE CONDITION
void RobotPlugin::data_request_subscriber_callback(...) {
    trial_data_request_waiting_ = true;  // NOT THREAD-SAFE
}

void RobotPlugin::update(...) {
    if (trial_data_request_waiting_) {  // RACE
        // ... handle request
        trial_data_request_waiting_ = false;
    }
}
```

**Solution:**
```cpp
#include <mutex>
#include <atomic>

class RobotPlugin {
private:
    std::atomic<bool> trial_data_request_waiting_{false};  // C++11 atomic
    mutable std::mutex sensors_mutex_;  // C++11 mutex

public:
    void data_request_subscriber_callback(...) {
        trial_data_request_waiting_.store(true, std::memory_order_release);
    }

    void update(...) {
        if (trial_data_request_waiting_.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lock(sensors_mutex_);  // C++11 RAII lock
            // ... handle request (thread-safe)
            trial_data_request_waiting_.store(false, std::memory_order_release);
        }
    }

    void configure_sensors(OptionsMap& opts) {
        std::lock_guard<std::mutex> lock(sensors_mutex_);
        // ... sensor configuration (thread-safe)
    }
};
```

**C++20 Enhancement:**
```cpp
#include <atomic>

// C++20: atomic wait/notify (more efficient than spin-loop)
std::atomic<bool> ready{false};

// Producer
ready.store(true, std::memory_order_release);
ready.notify_one();  // C++20

// Consumer
ready.wait(false, std::memory_order_acquire);  // C++20
```

**Validation:**
- Thread sanitizer: `catkin build --cmake-args -DCMAKE_CXX_FLAGS="-fsanitize=thread"`
- Stress test: High-frequency message publishing
- Helgrind: Valgrind race detection

**Commit:** `"fix: Add thread safety to ROS callbacks (fixes race conditions)"`

---

### Iteration 3.5: Modern C++20 Features (Days 24-26)

**Ranges (C++20):**
```cpp
// BEFORE (C++98)
for (int i = 0; i < sensors_.size(); i++) {
    sensors_[i]->configure_sensor(opts);
}

// BETTER (C++11)
for (auto& sensor : sensors_) {
    sensor->configure_sensor(opts);
}

// BEST (C++20 ranges)
#include <ranges>
namespace rng = std::ranges;

rng::for_each(sensors_, [&opts](auto& sensor) {
    sensor->configure_sensor(opts);
});

// C++20 range views (lazy evaluation)
auto configured_sensors = sensors_
    | rng::views::filter([](auto& s) { return s->is_initialized(); })
    | rng::views::transform([&opts](auto& s) {
        s->configure_sensor(opts);
        return s;
    });
```

**Concepts (C++20):**
```cpp
// Type-safe template constraints
template<typename T>
concept Sensor = requires(T t, OptionsMap opts) {
    { t.configure_sensor(opts) } -> std::same_as<void>;
    { t.is_initialized() } -> std::convertible_to<bool>;
};

template<Sensor S>
void configure_all_sensors(std::vector<S>& sensors, OptionsMap& opts) {
    // Compile-time enforced: S must satisfy Sensor concept
}
```

**std::span (C++20):**
```cpp
// BEFORE: Unsafe array passing
void process_data(double* data, size_t size);

// AFTER: Safe, bounds-checked view
#include <span>
void process_data(std::span<double> data);
```

**Commit:** `"feat: Modernize with C++20 ranges, concepts, and span"`

---

## Phase 4: Dependency Updates (Weeks 7-8)

### Iteration 4.1: MuJoCo Migration (Days 27-32)

**Step-by-step:**

1. **Install MuJoCo 3.x in Docker**
```dockerfile
RUN wget https://github.com/deepmind/mujoco/releases/download/3.0.0/mujoco-3.0.0-linux-x86_64.tar.gz
RUN tar -xzf mujoco-3.0.0-linux-x86_64.tar.gz -C /opt
ENV MUJOCO_PATH=/opt/mujoco-3.0.0
```

2. **Update XML models (10 files)**
- Validate each model in MuJoCo 3.x viewer
- Fix deprecated XML syntax
- Test physics simulation

3. **Update Python bindings**
- Replace `mjcpy2` with official `mujoco` package
- Update API calls: `mj_step(model, data)` etc.

4. **Validate physics equivalence**
- Run 100 rollouts with same initial conditions
- Compare trajectories (tolerance < 1e-6)
- Check if policy transfers

**Commit (per model):** `"feat: Migrate reacher.xml to MuJoCo 3.x"`

---

### Iteration 4.2: Boost.NumPy → pybind11 (Days 33-37)

**Why pybind11:**
- Modern, header-only library
- Better numpy integration
- Faster compilation
- Active maintenance

**Migration:**
```cpp
// BEFORE (Boost.Python + Boost.NumPy)
#include <boost/python.hpp>
#include <boost/numpy.hpp>

BOOST_PYTHON_MODULE(my_module) {
    boost::python::def("my_func", my_func);
}

// AFTER (pybind11)
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

PYBIND11_MODULE(my_module, m) {
    m.def("my_func", &my_func, "Function docstring");
}
```

**Validation:**
- Import all C++ extensions in Python
- Run numerical tests on C++ functions
- Check memory leaks with valgrind

**Commit:** `"refactor: Migrate from Boost.NumPy to pybind11"`

---

## Phase 5: Comprehensive Testing (Weeks 9-11)

### Testing Pyramid

**Level 1: Unit Tests (Days 38-45)**
- Target: 80% code coverage
- Framework: pytest + pytest-cov
- Focus: Algorithm correctness

**Level 2: Integration Tests (Days 46-50)**
- Test multi-component interactions
- ROS message passing
- GUI ↔ backend communication

**Level 3: System Tests (Days 51-60)**
- All 53 experiments
- Numerical validation vs baseline
- Performance benchmarks

**Level 4: Load/Soak Tests (Days 61-65)**
- 24-hour continuous run
- Memory leak detection
- GPU utilization monitoring

**Level 5: Fault Injection (Days 66-70)**
- Network failures
- GPU OOM scenarios
- Process crashes and recovery

---

## Validation Framework

### Numerical Validation Script

```python
# validate.py
import numpy as np
import subprocess
import json

def run_baseline(experiment_dir):
    """Run Python 2.7 baseline."""
    cmd = ["docker", "run", "gps-py27", "python2", "gps_main.py", experiment_dir]
    result = subprocess.run(cmd, capture_output=True)
    return parse_output(result.stdout)

def run_modernized(experiment_dir):
    """Run Python 3.11 modernized."""
    cmd = ["docker", "run", "gps-py311", "python3", "gps_main.py", experiment_dir]
    result = subprocess.run(cmd, capture_output=True)
    return parse_output(result.stdout)

def validate_equivalence(baseline, modernized, tolerance=1e-6):
    """Validate numerical equivalence."""
    for key in baseline.keys():
        baseline_val = np.array(baseline[key])
        modern_val = np.array(modernized[key])

        diff = np.abs(baseline_val - modern_val)
        max_diff = np.max(diff)

        if max_diff > tolerance:
            print(f"FAIL: {key} differs by {max_diff:.2e}")
            return False

    print("PASS: All outputs within tolerance")
    return True

# Run on all 53 experiments
for exp in get_all_experiments():
    print(f"Validating {exp}...")
    baseline = run_baseline(exp)
    modernized = run_modernized(exp)
    validate_equivalence(baseline, modernized)
```

---

## CI/CD Pipeline (Week 12)

### GitHub Actions Workflow

```yaml
name: GPS Modernization CI

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Python linting
        run: |
          pip install pylint mypy black
          black --check python/
          pylint python/gps
          mypy python/gps
      - name: C++ linting
        run: |
          sudo apt-get install -y cppcheck clang-tidy
          cppcheck --enable=all gps_agent_pkg/src/
          clang-tidy gps_agent_pkg/src/*.cpp

  test-python:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run unit tests
        run: |
          docker build -t gps-test -f Dockerfile.test .
          docker run gps-test pytest python/tests/ -v --cov=python/gps

  test-cpp:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build C++ with ROS
        run: |
          docker build -t gps-ros -f Dockerfile.ros .
          docker run gps-ros catkin build gps_agent_pkg
      - name: Run C++ tests
        run: |
          docker run gps-ros rostest gps_agent_pkg test_all.test

  integration:
    runs-on: ubuntu-latest
    needs: [test-python, test-cpp]
    steps:
      - name: Run integration tests
        run: |
          docker-compose up -d
          docker exec gps pytest python/tests/integration/ -v

  validate-experiments:
    runs-on: self-hosted  # Requires GPU
    needs: [integration]
    steps:
      - name: Run experiment validation
        run: |
          python scripts/validate_all_experiments.py --parallel 4
```

---

## Risk Mitigation

### Rollback Strategy

**Every phase has a rollback branch:**
```bash
git checkout -b phase1-dict-methods
# ... make changes ...
git commit -m "Phase 1.1 complete"

# If issues found:
git checkout main
git branch -D phase1-dict-methods  # Rollback
```

### Smoke Tests

**Run after every iteration:**
```bash
# Quick sanity check (< 5 min)
python -m pytest python/tests/smoke/
catkin build gps_agent_pkg && rostest gps_agent_pkg smoke.test
```

---

## Documentation

**Update after each phase:**
1. CHANGELOG.md - what changed
2. MIGRATION.md - known issues, breaking changes
3. API_CHANGES.md - API differences from Python 2.7 version
4. PERFORMANCE.md - benchmarks, optimizations

---

## Success Metrics

**Phase completion criteria:**

✅ **Phase 1 (Python Core):**
- All Python 3.8+ syntax correct
- `mypy --strict` passes
- No print statements, all logging
- 50% test coverage

✅ **Phase 2 (TensorFlow):**
- TF 2.x APIs only (no compat layer)
- GPU utilization > 80%
- Training time within 10% of baseline
- Loss curves match baseline

✅ **Phase 3 (C++):**
- C++20 standard compliant
- Zero thread sanitizer warnings
- Zero memory leaks (valgrind)
- 10% performance improvement from move semantics

✅ **Phase 4 (Dependencies):**
- MuJoCo 3.x physics equivalent (< 1e-6 error)
- pybind11 bindings functional
- All 10 XML models work

✅ **Phase 5 (Testing):**
- 80% code coverage
- All 53 experiments pass
- 24-hour soak test successful
- Load test: 10x normal throughput

---

## Timeline Summary

| Phase | Duration | Deliverable |
|-------|----------|------------|
| 1. Python Core | 2 weeks | Python 3.8+ compatible code |
| 2. TensorFlow | 2 weeks | TF 2.x migration complete |
| 3. C++ Modern | 2 weeks | C++20 with thread safety |
| 4. Dependencies | 1 week | MuJoCo 3.x, pybind11 |
| 5. Testing | 3 weeks | Full test suite, validation |
| 6. CI/CD | 1 week | Automated pipeline |
| **Total** | **11 weeks** | **Production-ready codebase** |

---

## Next Steps

1. Set up Docker environments (Baseline, Modernized, Test)
2. Create git branch: `claude_gps`
3. Begin Phase 1, Iteration 1.1: Fix dict methods
4. Commit incrementally with detailed messages
5. Run validation after each iteration

**Let's begin! 🚀**
