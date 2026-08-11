"""
Enforced truth/estimator separation.

Importing the `estimator` package installs a process-wide monkeypatch on
np.load that raises LeakageError the instant any code path (the forward
solver, the EnKF, a hyperparameter-tuning loop, anything) attempts to open
one of the withheld ground-truth files. This is intentionally a blunt,
global guard rather than a polite convention: the whole point of this
experiment is that a subtle leak (e.g. someone "just quickly" peeking at
Mtrue_v1 to sanity check an initial condition) would invalidate the result,
so accidental leakage must be loud and immediate, not a silent methodology
bug discovered after the fact.

`evaluation/` code -- the only code allowed to touch the truth files -- must
explicitly wrap those reads in `with allow_truth_access():`.
"""
import functools
import threading
import numpy as np

FORBIDDEN_SUBSTRINGS = [
    'reference_truth_modal.npz',
    'reference_truth_full.npz',
    'fixed_cylinder_atRe100',
    '_flow_cache.npz',
]

_state = threading.local()


class LeakageError(RuntimeError):
    pass


_original_load = np.load
_installed = False


@functools.wraps(_original_load)
def _guarded_load(file, *args, **kwargs):
    path = str(file if isinstance(file, str) else getattr(file, 'name', file))
    if not getattr(_state, 'allowed', False):
        for bad in FORBIDDEN_SUBSTRINGS:
            if bad in path:
                raise LeakageError(
                    "Blocked attempt to load a withheld ground-truth file "
                    "(%r). Estimator code (the NS forward model / EnKF) may "
                    "only load enkf_pressure_only/data/tap_observations.npz. "
                    "If this call is legitimately part of evaluation/, wrap "
                    "it in `with allow_truth_access():`." % path)
    return _original_load(file, *args, **kwargs)


def install():
    global _installed
    np.load = _guarded_load
    _installed = True


class allow_truth_access:
    """Context manager for evaluation/ code only -- lifts the guard."""

    def __enter__(self):
        self._prev = getattr(_state, 'allowed', False)
        _state.allowed = True
        return self

    def __exit__(self, *exc):
        _state.allowed = self._prev
        return False
