"""
Phase 1 tests — Python 3 compatibility gate.

Every test here corresponds to a specific Python-2-only pattern that was
fixed. The test names mirror the fix applied. All tests must pass before
Phase 2 (PyTorch rewrite) begins.
"""
import sys
import os
import pickle
import importlib
import importlib.util
import inspect

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helper: read source of a gps module by path (avoids triggering heavy imports)
# ---------------------------------------------------------------------------
_GPS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "gps"))


def _read_source(rel_path: str) -> str:
    """Read raw source of a gps file without importing it."""
    full = os.path.join(_GPS_ROOT, rel_path)
    with open(full) as f:
        return f.read()


# ---------------------------------------------------------------------------
# 1. ABC metaclass: classes must be abstract under Python 3
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_agent_is_abc():
    """Agent must be recognised as an abstract class in Python 3."""
    import abc
    from gps.agent.agent import Agent
    assert issubclass(Agent, abc.ABC) or inspect.isabstract(Agent), (
        "Agent is not abstract. __metaclass__ fix not applied."
    )


@pytest.mark.unit
def test_algorithm_is_abc():
    from gps.algorithm.algorithm import Algorithm
    assert inspect.isabstract(Algorithm)


@pytest.mark.unit
def test_cost_is_abc():
    from gps.algorithm.cost.cost import Cost
    assert inspect.isabstract(Cost)


@pytest.mark.unit
def test_dynamics_is_abc():
    from gps.algorithm.dynamics.dynamics import Dynamics
    assert inspect.isabstract(Dynamics)


@pytest.mark.unit
def test_policy_is_abc():
    from gps.algorithm.policy.policy import Policy
    assert inspect.isabstract(Policy)


@pytest.mark.unit
def test_policy_opt_is_abc():
    from gps.algorithm.policy_opt.policy_opt import PolicyOpt
    assert inspect.isabstract(PolicyOpt)


@pytest.mark.unit
def test_traj_opt_is_abc():
    from gps.algorithm.traj_opt.traj_opt import TrajOpt
    assert inspect.isabstract(TrajOpt)


# ---------------------------------------------------------------------------
# 2. sample_list: import pickle (not cPickle), correct dump arg order
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_sample_list_no_cpickle():
    """sample_list.py must not import cPickle."""
    source = _read_source("sample/sample_list.py")
    assert 'cPickle' not in source, "cPickle still present in sample_list.py"


@pytest.mark.unit
def test_sample_list_pickle_dump_order(tmp_path):
    """PickleSampleWriter.write must call pickle.dump(object, file) (correct order)."""
    from gps.sample.sample_list import PickleSampleWriter
    data_file = str(tmp_path / "test.pkl")
    writer = PickleSampleWriter(data_file)
    payload = [1, 2, 3]
    writer.write(payload)
    with open(data_file, 'rb') as f:
        result = pickle.load(f)
    assert result == payload, (
        f"PickleSampleWriter wrote wrong data: {result!r} != {payload!r}. "
        "Likely still using wrong cPickle.dump(file, object) arg order."
    )


@pytest.mark.unit
def test_sample_list_accepts_adv():
    """SampleList.__init__ must accept an optional samples_adv argument."""
    from gps.sample.sample_list import SampleList
    sig = inspect.signature(SampleList.__init__)
    params = list(sig.parameters.keys())
    assert 'samples_adv' in params, (
        "SampleList.__init__ missing 'samples_adv' parameter"
    )


# ---------------------------------------------------------------------------
# 3. data_logger: uses context managers (no fd leaks), Python 3 pickle
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_data_logger_no_cpickle():
    """data_logger.py must not import cPickle."""
    source = _read_source("utility/data_logger.py")
    assert 'cPickle' not in source


@pytest.mark.unit
def test_data_logger_round_trip(tmp_path):
    """DataLogger must successfully pickle and unpickle a dict."""
    from gps.utility.data_logger import DataLogger
    logger = DataLogger()
    data = {'key': [1.0, 2.0, 3.0], 'nested': {'a': 42}}
    path = str(tmp_path / "test.pkl")
    logger.pickle(path, data)
    recovered = logger.unpickle(path)
    assert recovered == data


@pytest.mark.unit
def test_data_logger_missing_file_returns_none(tmp_path):
    """DataLogger.unpickle must return None (not raise) for a missing file."""
    from gps.utility.data_logger import DataLogger
    logger = DataLogger()
    result = logger.unpickle(str(tmp_path / "nonexistent.pkl"))
    assert result is None


# ---------------------------------------------------------------------------
# 4. GUI ps3_config / config: no .iteritems()
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_ps3_config_no_iteritems():
    source = _read_source("gui/ps3_config.py")
    assert '.iteritems()' not in source
    assert '.itervalues()' not in source


@pytest.mark.unit
def test_gui_config_no_iteritems():
    source = _read_source("gui/config.py")
    assert '.iteritems()' not in source


# ---------------------------------------------------------------------------
# 5. lin_gauss_init: correct LinAlgError import
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_lin_gauss_init_linalgerror_import():
    """LinAlgError must be importable from lin_gauss_init module scope."""
    source = _read_source("algorithm/policy/lin_gauss_init.py")
    assert 'import numpy.linalg as LinAlgError' not in source, (
        "Old Python 2 'import numpy.linalg as LinAlgError' still present"
    )
    assert 'from numpy.linalg import LinAlgError' in source


# ---------------------------------------------------------------------------
# 6. cost_state: no hardcoded gamma/mode, return outside loop
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_cost_state_no_hardcoded_gamma():
    source = _read_source("algorithm/cost/cost_state.py")
    assert 'self.gamma = 5' not in source, (
        "cost_state.py still has hardcoded 'self.gamma = 5'"
    )


@pytest.mark.unit
def test_cost_state_no_hardcoded_mode():
    source = _read_source("algorithm/cost/cost_state.py")
    assert "self.mode = 'antagonist'" not in source, (
        "cost_state.py still has hardcoded \"self.mode = 'antagonist'\""
    )


@pytest.mark.unit
def test_cost_state_return_outside_loop():
    """The return statement must be outside the for-loop in cost_state.eval()."""
    source = _read_source("algorithm/cost/cost_state.py")
    # Extract just the eval method body by finding its def and the next def
    start = source.find('    def eval(self, sample):')
    next_def = source.find('\n    def ', start + 1)
    eval_src = source[start:next_def] if next_def != -1 else source[start:]
    lines = eval_src.split('\n')
    # Find the last 'return' line and the last 'for' line
    last_return_idx = max(
        (i for i, line in enumerate(lines) if line.strip().startswith('return')),
        default=-1
    )
    last_for_idx = max(
        (i for i, line in enumerate(lines) if line.strip().startswith('for ')),
        default=-1
    )
    assert last_return_idx != -1, "No return statement found in eval()"
    assert last_return_idx > last_for_idx, (
        "return statement appears to be inside the for-loop in cost_state.eval()"
    )


# ---------------------------------------------------------------------------
# 7. traj_opt_pi2: no xrange
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_traj_opt_pi2_no_xrange():
    source = _read_source("algorithm/traj_opt/traj_opt_pi2.py")
    assert 'xrange' not in source


# ---------------------------------------------------------------------------
# 8. cost_action: os._exit replaced with ValueError
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_cost_action_no_os_exit_string():
    source = _read_source("algorithm/cost/cost_action.py")
    assert 'os._exit(' not in source, (
        "cost_action.py still uses os._exit() — should raise ValueError"
    )


# ---------------------------------------------------------------------------
# 9. gps_main: imp replaced with importlib.util
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_gps_main_no_imp():
    source = _read_source("gps_main.py")
    # Use word-boundary check: 'import imp\n' or 'import imp ' to avoid
    # matching 'import importlib.util' which contains 'import imp' as substring.
    import re
    assert not re.search(r'\bimport imp\b(?!ort)', source), (
        "gps_main.py still uses deprecated 'import imp'"
    )
    assert 'importlib.util' in source


# ---------------------------------------------------------------------------
# 10. Confirm all fixes together: full import chain is Python 3 clean
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_full_import_chain():
    """All fixed modules must be importable without SyntaxError."""
    modules_to_test = [
        'gps.agent.agent',
        'gps.algorithm.algorithm',
        'gps.algorithm.cost.cost',
        'gps.algorithm.cost.cost_state',
        'gps.algorithm.cost.cost_binary_region',
        'gps.algorithm.dynamics.dynamics',
        'gps.algorithm.policy.policy',
        'gps.algorithm.policy.lin_gauss_init',
        'gps.algorithm.policy_opt.policy_opt',
        'gps.algorithm.traj_opt.traj_opt',
        'gps.algorithm.traj_opt.traj_opt_pi2',
        'gps.sample.sample_list',
        'gps.utility.data_logger',
    ]
    failed = []
    for mod_name in modules_to_test:
        try:
            importlib.import_module(mod_name)
        except Exception as exc:
            failed.append(f"{mod_name}: {exc}")
    assert not failed, "Import failures:\n" + "\n".join(failed)
