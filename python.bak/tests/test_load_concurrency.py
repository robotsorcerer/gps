"""
Load, concurrency, and soak tests.

load  — parametrized throughput / scalability (run in every CI gate after unit gate)
soak  — long-running stability (run nightly only, skipped in regular CI)

Marks are gated in pytest.ini:
  addopts = -m "not soak and not gpu and not ros"
"""
from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_opt(dO: int = 14, dU: int = 7, iters: int = 3) -> Any:
    from gps.algorithm.policy_opt.policy_opt_pytorch import PolicyOptPyTorch
    return PolicyOptPyTorch(
        {"random_seed": 0, "iterations": iters, "batch_size": 8,
         "lr": 1e-3, "weight_decay": 1e-4, "ent_reg": 0.0,
         "init_var": 0.1, "init_var_v": 0.1},
        dO, dU,
    )


def _train_once(opt: Any, N: int, T: int, seed: int = 0) -> None:
    dO = opt._dO
    dU = opt._dU
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((N, T, dO)).astype(np.float32)
    mu = rng.standard_normal((N, T, dU)).astype(np.float32)
    prc = np.tile(np.eye(dU, dtype=np.float32), (N, T, 1, 1))
    wt = np.ones((N, T), dtype=np.float32)
    opt.update(obs, mu, prc, wt)


# ---------------------------------------------------------------------------
# Shared trained policy (module-scoped to avoid re-training per test)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_opt():
    opt = _make_opt(dO=14, dU=7, iters=5)
    _train_once(opt, N=4, T=5)
    return opt


# ===========================================================================
# Load tests — parametrized N and T to measure scaling
# ===========================================================================

@pytest.mark.load
@pytest.mark.parametrize("N,T", [
    (1, 5),
    (4, 10),
    (16, 20),
    (32, 10),
])
def test_update_throughput_at_scale(N: int, T: int):
    """update() must complete in finite time for varying N and T."""
    opt = _make_opt(iters=2)
    t0 = time.perf_counter()
    _train_once(opt, N=N, T=T)
    elapsed = time.perf_counter() - t0
    # Not a hard latency assertion — just verifies it doesn't hang.
    assert elapsed < 60.0, f"update(N={N}, T={T}) took {elapsed:.1f}s — suspiciously slow"


@pytest.mark.load
@pytest.mark.parametrize("dO,dU", [
    (7, 3),
    (14, 7),
    (50, 20),
    (100, 30),
])
def test_inference_throughput_at_obs_scale(dO: int, dU: int):
    """act() must produce 100 inferences/second for any reasonable (dO, dU)."""
    opt = _make_opt(dO=dO, dU=dU, iters=2)
    _train_once(opt, N=2, T=3)

    rng = np.random.default_rng(0)
    obs = rng.standard_normal(dO).astype(np.float32)
    noise = rng.standard_normal(dU).astype(np.float32)

    N_INFER = 200
    t0 = time.perf_counter()
    for _ in range(N_INFER):
        opt.policy.act(None, obs, 0, noise)
    elapsed = time.perf_counter() - t0

    throughput = N_INFER / elapsed
    assert throughput >= 100.0, (
        f"act() throughput for (dO={dO}, dU={dU}): "
        f"{throughput:.0f} infer/s < 100 infer/s threshold"
    )


@pytest.mark.load
def test_batch_prob_throughput():
    """prob() on a batch of 32 trajectories must complete under 5 seconds."""
    opt = _make_opt(iters=2)
    _train_once(opt, N=2, T=3)

    rng = np.random.default_rng(1)
    obs = rng.standard_normal((32, 20, 14)).astype(np.float32)

    t0 = time.perf_counter()
    output, _, _, _ = opt.prob(obs)
    elapsed = time.perf_counter() - t0

    assert output.shape == (32, 20, 7)
    assert elapsed < 5.0, f"prob(N=32, T=20) took {elapsed:.2f}s > 5s"


# ===========================================================================
# Concurrency tests
# ===========================================================================

@pytest.mark.load
def test_concurrent_act_calls_thread_safe(trained_opt):
    """
    N_THREADS threads calling act() on the same policy simultaneously
    must all produce finite results without race conditions.
    """
    N_THREADS = 8
    N_CALLS = 50
    dO, dU = trained_opt._dO, trained_opt._dU
    results: list[np.ndarray | None] = [None] * N_THREADS
    errors: list[Exception | None] = [None] * N_THREADS

    def worker(tid: int) -> None:
        rng = np.random.default_rng(tid)
        try:
            for _ in range(N_CALLS):
                obs = rng.standard_normal(dO).astype(np.float32)
                u = trained_opt.policy.act(None, obs, 0, None)
                assert u.shape == (dU,), f"Thread {tid}: wrong shape {u.shape}"
            results[tid] = np.zeros(dU)  # sentinel: success
        except Exception as exc:
            errors[tid] = exc

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)
        assert not t.is_alive(), "act() thread did not complete in 30s"

    for tid, err in enumerate(errors):
        assert err is None, f"Thread {tid} raised: {err}"
    for tid, res in enumerate(results):
        assert res is not None, f"Thread {tid} produced no result"


@pytest.mark.load
def test_concurrent_prob_calls_thread_safe(trained_opt):
    """prob() must be thread-safe for concurrent read-only inference calls."""
    N_THREADS = 4
    dO, dU = trained_opt._dO, trained_opt._dU
    errors: list[Exception | None] = [None] * N_THREADS

    def worker(tid: int) -> None:
        rng = np.random.default_rng(tid)
        obs = rng.standard_normal((2, 5, dO)).astype(np.float32)
        try:
            for _ in range(20):
                output, _, _, _ = trained_opt.prob(obs)
                assert output.shape == (2, 5, dU)
        except Exception as exc:
            errors[tid] = exc

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)
        assert not t.is_alive()

    for tid, err in enumerate(errors):
        assert err is None, f"Thread {tid} raised: {err}"


# ===========================================================================
# Memory-stability load tests
# ===========================================================================

@pytest.mark.load
def test_repeated_inferences_no_memory_leak():
    """
    Running 1000 act() calls must not cause unbounded tensor allocation.
    Tracks RSS (resident set size) on CPU via psutil; GPU memory separately
    in test_gpu_inference_no_memory_growth.
    """
    import psutil, os
    opt = _make_opt(iters=2)
    _train_once(opt, N=2, T=3)

    rng = np.random.default_rng(42)
    dO, dU = opt._dO, opt._dU
    obs = rng.standard_normal(dO).astype(np.float32)
    noise = rng.standard_normal(dU).astype(np.float32)

    proc = psutil.Process(os.getpid())

    # Warm up — let allocators stabilise before measuring.
    for _ in range(20):
        opt.policy.act(None, obs, 0, noise)

    rss_before_mb = proc.memory_info().rss / 1e6

    for _ in range(1000):
        opt.policy.act(None, obs, 0, noise)

    rss_after_mb = proc.memory_info().rss / 1e6
    delta_mb = rss_after_mb - rss_before_mb

    assert delta_mb < 50.0, (
        f"Process RSS grew by {delta_mb:.1f} MB over 1000 inferences — "
        "possible tensor accumulation (threshold: 50 MB)"
    )


@pytest.mark.gpu
def test_gpu_inference_no_memory_growth():
    """
    CUDA-specific: 1000 act() calls on a GPU-placed model must not grow
    device memory beyond 10 MB.  Skipped when no CUDA device is available.
    """
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device available — skipping GPU memory test")

    opt = _make_opt(iters=2)
    # Force GPU placement by overriding use_cuda and _device directly.
    opt.use_cuda = True
    opt._device = torch.device("cuda:0")
    opt._net = opt._net.to(opt._device)
    opt.policy.with_gpu = True
    _train_once(opt, N=2, T=3)

    rng = np.random.default_rng(42)
    dO, dU = opt._dO, opt._dU
    obs = rng.standard_normal(dO).astype(np.float32)
    noise = rng.standard_normal(dU).astype(np.float32)

    # Warm up
    for _ in range(20):
        opt.policy.act(None, obs, 0, noise)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before_mb = torch.cuda.memory_allocated() / 1e6

    for _ in range(1000):
        opt.policy.act(None, obs, 0, noise)

    torch.cuda.synchronize()
    after_mb = torch.cuda.memory_allocated() / 1e6
    delta_mb = after_mb - before_mb

    assert delta_mb < 10.0, (
        f"GPU memory grew by {delta_mb:.1f} MB over 1000 inferences — "
        "possible tensor accumulation (threshold: 10 MB)"
    )


# ===========================================================================
# Soak tests — long-running stability (nightly only)
# ===========================================================================

@pytest.mark.soak
def test_soak_training_loop_numerical_stability():
    """
    Run 100 consecutive update() calls and verify that no NaN/Inf
    contaminates the network weights.
    """
    opt = _make_opt(iters=10)
    rng = np.random.default_rng(999)
    dO, dU = opt._dO, opt._dU

    for i in range(100):
        N, T = 4, 8
        obs = rng.standard_normal((N, T, dO)).astype(np.float32)
        mu = rng.standard_normal((N, T, dU)).astype(np.float32)
        prc = np.tile(np.eye(dU, dtype=np.float32), (N, T, 1, 1))
        wt = np.ones((N, T), dtype=np.float32)
        opt.update(obs, mu, prc, wt)

        # Verify weights remain finite
        for name, param in opt._net.named_parameters():
            if not torch.isfinite(param).all():
                pytest.fail(
                    f"NaN/Inf in parameter '{name}' after iteration {i+1}"
                )


@pytest.mark.soak
def test_soak_inference_determinism_over_1000_calls():
    """
    The same observation fed to act() 1000 times must always produce
    the same result (no hidden state drift).
    """
    opt = _make_opt(iters=5)
    _train_once(opt, N=4, T=5)

    rng = np.random.default_rng(1)
    dO = opt._dO
    obs = rng.standard_normal(dO).astype(np.float32)

    reference = opt.policy.act(None, obs, 0, None)
    for i in range(1000):
        u = opt.policy.act(None, obs, 0, None)
        np.testing.assert_array_equal(
            u, reference,
            err_msg=f"act() output changed at call {i+1}"
        )
