# GPS iDG — Staff-Level Production Readiness Report

**Branch:** `iDG`
**Date:** 2026-02-10
**Scope:** Python 2 → 3.11, Caffe → PyTorch, Boost → C++17 migration
**Test baseline:** 121/121 passing (2 nightly soak tests excluded from gate)

---

## 1. Executive Summary

The iDG migration is **feature-complete** but **not yet production-safe for hardware deployment**.
The Python and proto layers are clean; the C++ agent plugin has three unmitigated failure paths
that could corrupt a live trajectory or cause a hard crash on a physical robot. Six architectural
gaps identified below are blockers; another four are strong recommendations before any hardware
trial.

**Recommendation: CONDITIONAL GO** — clear the six blocking items before connecting to
hardware. The software-simulation path (Box2D agent) is safe to run now.

---

## 2. Migration Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Test infrastructure (pytest, CI, marks) | ✅ Complete |
| 1 | Python 2 → 3 (30 fixes: ABC, cPickle, xrange, imp, …) | ✅ Complete |
| 2 | PyTorch policy rewrite (remove TF, fix shapes, loss) | ✅ Complete |
| 3 | C++17 modernization (CMake 3.14, std::unique_ptr, spinlock) | ✅ Complete |
| 4 | Caffe retirement (C++ + Python, deprecated_caffe/ directory) | ✅ Complete |
| 5 | requirements.txt, Dockerfile (Ubuntu 22.04 / CUDA 12.4), proto3 | ✅ Complete |
| 6 | Expanded test suite (fault, integration, load, concurrency) | ✅ Complete |

**Test coverage:** 121 tests across 7 files.
No GPU, ROS, or soak tests run in the standard gate; all pass on CPU in CI.

---

## 3. Cross-Service Integration Risks

### 3.1 Python → C++ model transport (HIGH RISK)

`PolicyOptPyTorch.export_torchscript()` serialises the network to bytes via `torch.jit.save`,
which are then packed into `TorchParams.model_bytes` and sent over the ROS `/trial_controller`
topic. The C++ side loads it with `torch::jit::load(std::istringstream(bytes), torch::kCPU)`.

**Risk:** TorchScript serialisation format is tied to the PyTorch ABI version. A Python-side
`torch==2.1` export is not guaranteed to load on a C++ `libtorch 2.0` build, and vice versa.
There is no version handshake in the ROS message.

**Mitigation needed:** Embed the PyTorch version string in `TorchParams.msg` as a `string
torch_version` field, and validate it in `PyTorchController::configure_controller()` before
calling `torch::jit::load`.

### 3.2 ROS topic serialisation latency (MEDIUM RISK)

`TorchParams.model_bytes` carries the full serialised network on every trial. A minimal MLP
(dO=14, dU=7, 2 hidden layers of 42) serialises to ~30 KB; larger networks approach several MB.
At 100 Hz control frequency this creates multi-MB/s intra-process ROS traffic. For the PR2
arm this is acceptable; for embedded hardware with USB–CAN bridges it is not.

**Mitigation:** Send the model once on a latched topic (`latch=True`) and resend only when
weights change (compare a SHA-256 of `model_bytes` on both sides).

### 3.3 Unknown controller type leaves `trial_controller_` null (BLOCKER)

In `robotplugin.cpp`, `trial_subscriber_callback` dispatches on `controller_to_execute` with
no enclosing try/catch. When the enum value is unrecognised, `ROS_ERROR` is logged but
`trial_controller_` is **not reset** — it remains pointing to whatever object was set by the
previous trial. The subsequent call to `trial_controller_->get_action(...)` then operates on
stale state, producing undefined behaviour (and likely a segfault on a null unique_ptr).

**Fix required:** After the unknown-type log, call `trial_controller_.reset()` and return
early from the callback without starting the trial.

### 3.4 Proto3 default `T = 0` is a semantic change (MEDIUM RISK)

The proto2 schema had `[default = 100]` on `Sample.T`. After the proto3 migration `T` defaults
to `0`. Any code that creates a `Sample` proto without explicitly setting `T` will silently
receive a zero-length trajectory. The Python side always sets `T` explicitly; the risk is in
external tooling or legacy replay scripts that relied on the proto2 default.

**Mitigation:** Add a proto-validation helper that asserts `T > 0` before deserialisation.

---

## 4. Observability Gaps

### 4.1 No structured logging (MEDIUM RISK)

The Python codebase uses `logging.getLogger(__name__)` uniformly — a good pattern — but all
messages are free-form strings. There are no structured fields (JSON, key=value) that a log
aggregator (Elasticsearch, Loki) could index. Critical events like "model updated",
"trajectory failed", "cost exploded" are buried in unindexed text.

**Recommendation:** Add a thin `gps.utility.structured_log` wrapper that emits JSON-lines
to stderr alongside the human-readable logger. At minimum tag: `{"event": "policy_update",
"iteration": N, "loss": L, "dU": D, "dO": D}`.

### 4.2 No policy training metrics export (MEDIUM RISK)

`PolicyOptPyTorch.update()` computes a loss at every iteration but discards it after
`loss.backward()`. There is no way to observe whether training is converging, diverging, or
oscillating without attaching a debugger.

**Fix:** Return (or emit via callback) a `{"final_loss": float, "iterations": int,
"grad_norm": float}` dict from `update()`. Surface it in the GPS iteration loop logs.

### 4.3 No timing instrumentation (LOW RISK)

The C++ `get_action()` call is on the realtime control path but is never timed. A slow
TorchScript inference (e.g. due to a large model or thermal throttling) will cause missed
control deadlines silently. There is no latency histogram.

**Recommendation:** Wrap the `module_.forward()` call with a `std::chrono::high_resolution_clock`
timer and emit via `ROS_WARN_THROTTLE` if it exceeds half the control period.

### 4.4 No watchdog for the trial controller (LOW RISK)

If `get_action()` hangs (e.g. OOM inside PyTorch), the trial loop spins forever. The
`publish_sample_report()` polling trylock has 1000 retries but no absolute timeout.

**Recommendation:** Add a `std::atomic<bool> trial_timed_out_` flag and a one-shot
`ros::WallTimer` that fires at `T × control_period + 500ms` and aborts the trial.

---

## 5. Failure Isolation Analysis

### 5.1 NaN propagation in training (BLOCKER)

`policy_opt_pytorch.py` has no gradient clipping and no NaN check on the loss. Adversarial
precision matrices (`tgt_prc` with very large eigenvalues) or degenerate sample weights will
produce NaN loss on the first backward pass. NaN then propagates through the Adam moment
estimates and contaminates all future updates. The network silently outputs all-NaN actions.

The fault injection test `test_update_all_zero_weights_does_not_crash` confirms this path:
the code warns but does not raise, and the policy output becomes NaN.

**Fix required (blocking):**
```python
# In update(), before loss.backward():
if not torch.isfinite(loss):
    LOGGER.error("NaN/Inf loss at iteration %d — skipping backward", i)
    continue
torch.nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=10.0)
```

### 5.2 C++ forward pass exception handling is too narrow (BLOCKER)

`PyTorchController::get_action()` catches `c10::Error` from the TorchScript forward pass, but
`torch::jit::load` and the `module_.forward()` call can also throw `std::invalid_argument`,
`std::runtime_error`, and `at::Error`. These propagate uncaught through the ROS callback,
causing `std::terminate` — i.e. the entire agent process dies, dropping the robot into an
unsupported state.

**Fix required (blocking):**
```cpp
try {
    auto output = module_.forward({input_tensor});
    // ...
} catch (const std::exception& exc) {
    ROS_ERROR("PyTorchController::get_action failed: %s", exc.what());
    U.setZero();   // safe fallback: zero torques
}
```

### 5.3 `DataLogger.unpickle` swallowed MemoryError silently (FIXED)

A corrupted `.pkl` file with a large fake frame size caused `MemoryError` inside `pickle.load`,
which the original handler (catching only `OSError`) did not catch — crashing the algorithm
loop. Fixed in this sprint: `MemoryError`, `UnpicklingError`, and `EOFError` are now caught
and return `None`.

### 5.4 Caffe controller dispatch silently falls through (MEDIUM RISK)

`ControllerParams.msg` still carries a `CaffeParams caffe` field and the `CAFFE_CONTROLLER`
enum value remains at `1` for proto numbering stability. If a replay bag or external tool
sends `controller_to_execute = 1`, the C++ plugin will log an "unknown controller type" error
but not abort — leaving the previous policy active. The robot continues executing the last
trajectory, which may be unsafe.

**Mitigation:** Explicitly handle `CAFFE_CONTROLLER` in the dispatch block with a
`ROS_ERROR_ONCE + return` rather than falling through to the default.

---

## 6. Versioning and Backward-Compatibility Risks

| Risk | Severity | Notes |
|------|----------|-------|
| TorchScript ABI across torch versions | HIGH | No version embedded in proto; mismatch silently loads and infers garbage |
| proto3 `T` default changed from 100 → 0 | MEDIUM | Breaks any client that omits `T` in `Sample` messages |
| ROS message field additions to `TorchParams.msg` | LOW | `catkin_make` catches mismatches at build time |
| `deprecated_caffe/` Python modules unpicklable by old checkpoints | LOW | `PolicyOptCaffe` objects in old `.pkl` files can no longer be unpickled — intentional retirement |
| `torch.jit.freeze` conditional | LOW | `export_torchscript()` calls `torch.jit.freeze` only if available; freeze semantics changed in 2.0 |

---

## 7. Load and Soak Test Results

Tests run on CPU (Intel 12-core, no GPU). See `python/tests/test_load_concurrency.py`.

| Test | Configuration | Result |
|------|--------------|--------|
| `test_update_throughput_at_scale` | N=1…32, T=5…20 | Pass — all complete < 1s |
| `test_inference_throughput_at_obs_scale` | dO/dU = 7/3 … 100/30 | ≥ 100 infer/s for all sizes |
| `test_batch_prob_throughput` | N=32, T=20 | < 1s (well under 5s limit) |
| `test_concurrent_act_calls_thread_safe` | 8 threads × 50 calls | Pass — no races detected |
| `test_concurrent_prob_calls_thread_safe` | 4 threads × 20 calls | Pass |
| `test_repeated_inferences_no_memory_leak` | 1000 × act() | Pass — CPU path allocates no persistent tensors |
| `test_soak_training_loop_numerical_stability` *(nightly)* | 100 update iters | Pass locally — weights remain finite |
| `test_soak_inference_determinism_over_1000_calls` *(nightly)* | 1000 × act() same obs | Pass — output identical across all calls |

**Caveat:** GPU memory-growth test is a no-op on CPU. Must be validated on CUDA hardware
before a GPU-enabled deployment.

---

## 8. Go / No-Go Production Criteria

### Blocking (must fix before hardware trials)

1. **Gradient NaN guard** — add `torch.isfinite(loss)` check and `clip_grad_norm_` in
   `PolicyOptPyTorch.update()`. Without this a single bad sample corrupts the entire policy.

2. **C++ exception broadening** — `PyTorchController::get_action` must catch
   `std::exception` (not only `c10::Error`) and fall back to zero torques.

3. **Null `trial_controller_` guard** — unknown controller type in
   `trial_subscriber_callback` must `reset()` the controller and return without starting
   a trial.

4. **TorchScript version handshake** — embed `torch.__version__` in `TorchParams.msg` and
   validate on the C++ side before loading.

5. **`-Werror` in CI C++ build** — currently the C++ build does not fail on compiler
   warnings. Enable `add_compile_options(-Wall -Wextra -Werror)` in `CMakeLists.txt` and
   fix all resulting warnings before hardware deployment.

6. **GPU soak test** — the CUDA memory-growth test is a no-op on CPU. Run
   `test_repeated_inferences_no_memory_leak` and the soak suite on actual CUDA hardware
   before enabling `use_gpu: 1`.

### Strong recommendations (should fix)

7. **Training metrics export** — return `{"loss": float, "grad_norm": float}` from
   `update()` so the GPS main loop can log convergence.

8. **ROS model caching** — send `TorchParams.model_bytes` on a latched topic; resend only
   on weight change.

9. **Control-loop latency alarm** — add a `std::chrono` timer around `module_.forward()`
   and warn when inference exceeds half the control period.

10. **Proto validation helper** — assert `Sample.T > 0` after deserialisation to guard
    against the proto3 default-zero regression.

---

## 9. Architectural Recommendations

### 9.1 Separate the training and inference services

Currently `PolicyOptPyTorch` trains the network and also holds the `PyTorchPolicy` inference
object. This means training (CPU-heavy, variable latency) shares a process with the GPS
orchestrator. For real-hardware use, training should run in a separate process (or a
`multiprocessing.Process`) and publish new weights over a ZMQ push socket, avoiding GIL
contention on the inference side.

### 9.2 Replace the ROS–Python bridge with a typed service call

The current design packs a serialised binary blob (`model_bytes`) into a ROS string field.
This is opaque to introspection tools (`rostopic echo`, `rosbag`). A better design would use
a dedicated ROS service `UpdatePolicy.srv` with explicit `string torch_version` and
`bytes model_bytes` fields, enabling version-checking middleware.

### 9.3 Add adversarial-perturbation hardening for iDG

The iDG game introduces an antagonist policy (`act_v`). The current `CostAction.eval()` in
antagonist mode reads `sample_prot` from a keyword argument — there is no type check or
None-guard. If `sample_prot` is None (e.g. during first-iteration warm-up), `sample_prot.get_U()`
raises `AttributeError` silently swallowed by the GPS outer loop. Add an explicit guard:
```python
if kwargs.get('sample_prot') is None:
    raise ValueError("antagonist CostAction requires sample_prot kwarg")
```

### 9.4 Migrate from `catkin` to `colcon` (ROS 2 compatibility)

ROS Noetic is EOL in May 2025. The Dockerfile targets Noetic on an unofficial Ubuntu 22.04
PPA. The medium-term path is ROS 2 Humble (Ubuntu 22.04 native, LTS until 2027). The
GPS C++ package should be wrapped in an `ament_cmake` manifest alongside the existing
`catkin` manifest to allow dual builds during the transition.

---

## 10. Summary Table

| Category | Status | Blockers |
|----------|--------|---------|
| Python migration | ✅ Complete | None |
| PyTorch policy | ✅ Complete | NaN guard missing (#1) |
| C++ plugin | ⚠️ Functional | Exception handling (#2), null ptr (#3) |
| Proto / ROS messages | ✅ Complete | Version handshake (#4) |
| CI / test suite | ✅ Complete | -Werror (#5), GPU soak (#6) |
| Observability | ⚠️ Partial | Training metrics, latency alarm |
| Hardware readiness | 🔴 Not ready | All 6 blocking items must close |
| Simulation readiness | ✅ Ready | Box2D agent safe to run now |
