"""
Tests for ``m3_adjust_centroid`` -- the ``tune_centroid``-based variant
of the M3 mirror adjustment.

These tests focus on the M3-specific scaffolding (backlash unwind,
diagnostic insert/retract, CSV log, finalize cleanup, return contract),
**not** the peak-finding algorithm itself. ``bluesky.plans.tune_centroid``
is upstream-tested; we only validate that we wire it correctly and that
the surrounding concerns behave the same way they do for
``m3_adjust``.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
from bluesky import RunEngine
from ophyd.sim import SynAxis, SynGauss


PROFILE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROFILE_DIR / "startup"))
from m3_adjust_plan import m3_adjust_centroid  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes -- shared shape with test_m3_adjust.py but kept local to avoid an
# inter-test-module import dependency.
# ---------------------------------------------------------------------------


class RecordingAxis(SynAxis):
    """SynAxis that records every target it was asked to move to."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.move_log = []

    def set(self, value):
        self.move_log.append(value)
        return super().set(value)


STEP = 5e-5
DIAG_IN = -6
DIAG_OUT = 2


@pytest.fixture
def env():
    """Fake instrument suitable for ``m3_adjust_centroid``.

    Uses ``SynGauss`` so ``tune_centroid`` has a real peak to converge
    on. The detector field name is ``"signal"`` (matching ``signal.name``).
    """
    mirror = RecordingAxis(name="mirror", value=0.0)
    diag = RecordingAxis(name="diag", value=0.0)
    # SynGauss centred at zero; the centroid scan must find it.
    signal = SynGauss(
        name="signal",
        motor=mirror,
        motor_field="mirror",
        center=0.0,
        Imax=1.0,
        sigma=5 * STEP,
    )

    RE = RunEngine({})

    # Drop real sleeps so settle_time doesn't slow tests down.
    async def _noop_sleep(msg):
        return None

    RE.register_command("sleep", _noop_sleep)

    return type("Env", (), {
        "RE": RE,
        "mirror": mirror,
        "diag": diag,
        "signal": signal,
    })


def _run(env, **overrides):
    kwargs = dict(
        mirror=env.mirror,
        signal=env.signal,
        diag=env.diag,
        diag_in=DIAG_IN,
        diag_out=DIAG_OUT,
        tune_range=20 * STEP,
        min_step=STEP,
        num_points=10,
        step_factor=3.0,
        n_samples=5,
        sample_delay=0.0,
        settle_time=0.0,
    )
    kwargs.update(overrides)
    return env.RE(m3_adjust_centroid(**kwargs))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_returns_pos_and_au(env):
    """Plan runs to completion without error.

    NOTE: ``RunEngine.__call__`` returns the tuple of run-start UIDs,
    not the plan's ``return`` value, so we cannot directly assert on
    ``(pos, au)`` from ``RE(plan)``. Exposing the final pos/au to
    callers is tracked separately; for now this test just confirms the
    plan completes and leaves the mirror at a well-defined position.
    """
    env.mirror.set(-3 * STEP).wait()
    env.mirror.move_log.clear()

    _run(env)

    assert isinstance(env.mirror.position, float)


def test_backlash_unwind_is_first(env):
    """The first two mirror moves implement the backlash unwind."""
    start = -3 * STEP
    env.mirror.set(start).wait()
    env.mirror.move_log.clear()

    _run(env)

    assert env.mirror.move_log[0] == pytest.approx(start - 2 * STEP)
    assert env.mirror.move_log[1] == pytest.approx(start + 2 * STEP)


def test_diagnostic_inserted_and_retracted(env):
    """Diag is moved to diag_in first and diag_out last; exactly one of each."""
    env.mirror.set(-3 * STEP).wait()
    env.mirror.move_log.clear()

    _run(env)

    assert env.diag.move_log[0] == DIAG_IN
    assert env.diag.move_log[-1] == DIAG_OUT
    assert env.diag.move_log.count(DIAG_IN) == 1
    assert env.diag.move_log.count(DIAG_OUT) == 1


def test_converges_to_peak(env):
    """Smoke test: centroid search should leave the mirror near the peak."""
    env.mirror.set(-15 * STEP).wait()
    env.mirror.move_log.clear()

    _run(env)

    # Centroid of a Gaussian centred at zero should land within a few
    # min_steps of zero.
    assert abs(env.mirror.position) <= 3 * STEP, (
        f"final position {env.mirror.position} not near peak at 0"
    )


def test_csv_row_on_success(env, tmp_path):
    """One row appended on success with the expected 5 columns."""
    env.mirror.set(-3 * STEP).wait()
    env.mirror.move_log.clear()

    csv_path = tmp_path / "log.csv"
    eng, pgm_energy, pgm_focus = 850.0, 851.23, 12.34
    result = _run(
        env,
        csv_path=str(csv_path),
        eng=eng,
        pgm_energy=pgm_energy,
        pgm_focus=pgm_focus,
    )
    # RE returns run-start UIDs, not plan return value; read final_pos
    # from the mirror itself.
    final_pos = env.mirror.position
    del result

    with open(csv_path) as f:
        rows = list(csv.reader(f))

    assert len(rows) == 1, f"expected exactly one row, got {rows!r}"
    row = rows[0]
    assert len(row) == 5
    assert float(row[0]) == pytest.approx(eng)
    assert float(row[1]) == pytest.approx(round(pgm_energy, 2))
    assert float(row[2]) == pytest.approx(round(pgm_focus, 2))
    float(row[3])  # parses
    assert float(row[4]) == pytest.approx(round(final_pos, 5))


def test_diag_retracted_on_exception(env):
    """``finalize_wrapper`` retracts diag even when an exception propagates."""

    # Replace the signal's read with one that raises after a few calls so
    # ``tune_centroid``'s scan blows up mid-flight.
    calls = {"n": 0}
    orig_read = env.signal.read

    def boom():
        calls["n"] += 1
        if calls["n"] > 3:
            raise RuntimeError("simulated detector failure")
        return orig_read()

    env.signal.read = boom

    env.mirror.set(0.0).wait()
    env.mirror.move_log.clear()

    with pytest.raises(RuntimeError, match="simulated detector failure"):
        _run(env)

    assert env.diag.move_log[0] == DIAG_IN
    assert env.diag.move_log[-1] == DIAG_OUT
