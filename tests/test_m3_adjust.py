"""
Characterization tests for the M3 mirror pitch hill-climb.

These tests pin the externally observable behavior of two
implementations of the same algorithm:

* ``startup/m3_adjust_core.py:m3_hill_climb`` -- the mechanical
  extraction of ``m3_adjust.py:113-226`` (used as the behavioral
  baseline / "before" snapshot).
* ``startup/m3_adjust_plan.py:m3_adjust`` -- the proper Bluesky plan
  refactor.

Each test is parametrized over a ``runner`` fixture so the same scenario
runs against both implementations. Where behavior intentionally
diverges (the give-up restore bug, fixed in the plan), the test branches
on ``runner.kind``.

Fakes are built from ``ophyd.sim`` so the production code paths --
``mirror.position``, ``signal.get()``, ``signal.read()``,
``RE(mv(...))``, ``yield from bps.mv``, ``yield from bps.rd`` -- are
all exercised against realistic ophyd objects.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
from bluesky import RunEngine
from ophyd.sim import SynAxis, SynSignal


# Make startup/ importable so we can pull both implementations out of it.
PROFILE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROFILE_DIR / "startup"))
from m3_adjust_core import m3_hill_climb  # noqa: E402
from m3_adjust_plan import m3_adjust  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class RecordingAxis(SynAxis):
    """SynAxis that records every target it was asked to move to."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.move_log = []

    def set(self, value):
        self.move_log.append(value)
        return super().set(value)


class RecordingSignal(SynSignal):
    """SynSignal that re-evaluates its ``func`` on every read.

    ``ophyd.sim.SynSignal`` caches the value computed at ``trigger()``
    time. ``m3_adjust_core`` reads via bare ``.get()`` and the Bluesky
    plan reads via ``bps.rd`` (which calls ``.read()``); we override
    both so that the simulated signal always tracks the current mirror
    position, mirroring how a live ``EpicsSignal`` would behave.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read_count = 0

    def _refresh(self):
        self.read_count += 1
        self._readback = self._func()

    def get(self, **kwargs):
        self._refresh()
        return self._readback

    def read(self):
        self._refresh()
        return super().read()


class Env:
    """Container for the test environment, returned by the ``env`` fixture."""

    def __init__(self, RE, mirror, diag, signal, sleeps, state, msg_log):
        self.RE = RE
        self.mirror = mirror
        self.diag = diag
        self.signal = signal
        self.sleeps = sleeps
        self.state = state
        self.msg_log = msg_log

    def set_peak(self, peak_fn, noise=0.0, seed=0):
        self.state["peak_fn"] = peak_fn
        self.state["noise"] = noise
        self.state["rng"] = np.random.default_rng(seed)


@pytest.fixture
def env(monkeypatch):
    """A fake instrument: recording motors, a noisy peak signal, a RunEngine.

    Sleep durations are captured into ``env.sleeps`` regardless of which
    implementation is running:

    * ``m3_hill_climb`` (core) calls ``time.sleep`` -- monkeypatched here.
    * ``m3_adjust`` (plan) yields ``Msg('sleep', None, t)`` -- captured
      via ``RE.msg_hook``.
    """
    sleeps = []
    msg_log = []

    # Patch the time.sleep that m3_adjust_core imported as ``time.sleep``.
    monkeypatch.setattr("m3_adjust_core.time.sleep", lambda t: sleeps.append(t))

    mirror = RecordingAxis(name="mirror", value=0.0)
    diag = RecordingAxis(name="diag", value=0.0)

    state = {
        "peak_fn": lambda p: 0.0,
        "noise": 0.0,
        "rng": np.random.default_rng(0),
    }

    signal = RecordingSignal(
        name="signal",
        func=lambda: (
            state["peak_fn"](mirror.position) + state["rng"].normal(0, state["noise"])
        ),
    )

    RE = RunEngine({})

    # Replace the RunEngine's ``sleep`` handler with a no-op so the plan
    # path doesn't actually wait ``settle_time`` seconds per iteration.
    # The msg_hook below still observes the original Msg, so sleep
    # durations are captured into ``env.sleeps`` before being dropped.
    async def _noop_sleep(msg):
        return None

    RE.register_command("sleep", _noop_sleep)

    def hook(msg):
        msg_log.append(msg)
        if msg.command == "sleep":
            sleeps.append(msg.args[0])

    RE.msg_hook = hook

    return Env(RE, mirror, diag, signal, sleeps, state, msg_log)


# ---------------------------------------------------------------------------
# Runner: parametrize every test over core (m3_hill_climb) and plan (m3_adjust)
# ---------------------------------------------------------------------------


class _Runner:
    """Callable wrapper that adapts the two implementations to one signature."""

    def __init__(self, kind, fn):
        self.kind = kind
        self.fn = fn

    def __call__(self, env, **kwargs):
        return self.fn(env, **kwargs)


def _run_core(env, **overrides):
    kwargs = dict(
        RE=env.RE,
        mirror=env.mirror,
        signal=env.signal,
        diag=env.diag,
        diag_in=DIAG_IN,
        diag_out=DIAG_OUT,
        step=STEP,
        n_samples=10,
        sample_delay=0.1,
        settle_time=3.0,
        max_insignificant=5,
    )
    kwargs.update(overrides)
    return m3_hill_climb(**kwargs)


def _run_plan(env, **overrides):
    kwargs = dict(
        mirror=env.mirror,
        signal=env.signal,
        diag=env.diag,
        diag_in=DIAG_IN,
        diag_out=DIAG_OUT,
        step=STEP,
        n_samples=10,
        sample_delay=0.1,
        settle_time=3.0,
        max_insignificant=5,
    )
    kwargs.update(overrides)
    env.RE(m3_adjust(**kwargs))
    return env.mirror.position


@pytest.fixture(params=["core", "plan"])
def runner(request):
    if request.param == "core":
        return _Runner("core", _run_core)
    return _Runner("plan", _run_plan)


# ---------------------------------------------------------------------------
# Defaults used across tests
# ---------------------------------------------------------------------------

STEP = 5e-5
DIAG_IN = -6
DIAG_OUT = 2


def _gaussian(center, width):
    """Gaussian peak generator: f(p) = exp(-((p-center)/width)**2)."""
    return lambda p: float(np.exp(-(((p - center) / width) ** 2)))


def _sample_count(env, runner):
    """Number of ``signal`` samples taken, regardless of implementation.

    For the core path we trust ``RecordingSignal.read_count``.
    For the plan path we count ``read`` messages with our signal as the
    target (``RE.msg_hook`` populates ``env.msg_log``).
    """
    if runner.kind == "core":
        return env.signal.read_count
    return sum(
        1 for m in env.msg_log
        if m.command == "read" and m.obj is env.signal
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_climbs_to_peak_from_left(env, runner):
    """Starting on the negative side of the peak, ends near the peak."""
    env.set_peak(_gaussian(center=0.0, width=10 * STEP), noise=1e-6, seed=1)
    start = -5 * STEP
    env.mirror.set(start).wait()
    env.mirror.move_log.clear()

    final = runner(env)

    assert abs(final) <= STEP, f"final position {final} not within step of peak"


def test_climbs_to_peak_from_right(env, runner):
    """Starting on the positive side, the algorithm chooses direction -1."""
    env.set_peak(_gaussian(center=0.0, width=10 * STEP), noise=1e-6, seed=2)
    start = +5 * STEP
    env.mirror.set(start).wait()
    env.mirror.move_log.clear()

    final = runner(env)

    # Generous bound: the asymmetric "extra step" when direction == -1 may
    # cause the final point to land one step past the peak.
    assert abs(final) <= 3 * STEP, f"final position {final} too far from peak"
    # Confirm the algorithm actually moved toward negative values.
    assert any(t < start for t in env.mirror.move_log), (
        f"no negative-direction moves recorded: {env.mirror.move_log}"
    )


def test_restores_position_on_pure_noise(env, runner, tmp_path):
    """Give-up final mirror position differs between core (buggy) and plan (fixed).

    The original ``m3_adjust.py`` issues ``RE(mv(mirror, M3_Ry_0))`` on
    give-up but then falls through to code that moves the mirror back
    to the last attempted position. The Bluesky plan fixes this with an
    explicit early return after the restore.

    Core final position:
        start + 2*step          (backlash settle position = M3_Ry_0)
              + step            (first +step probe)
              + max_insignificant * step  (one increment per insig. iter.)

    Plan final position:
        start + 2*step          (M3_Ry_0; genuine restore)
    """
    env.set_peak(lambda p: 0.0, noise=0.0, seed=3)
    start = 0.001
    env.mirror.set(start).wait()
    env.mirror.move_log.clear()

    csv_path = tmp_path / "log.csv"
    max_insig = 5
    final = runner(
        env,
        csv_path=str(csv_path),
        eng=850.0,
        pgm_energy=850.0,
        pgm_focus=0.0,
        max_insignificant=max_insig,
    )

    if runner.kind == "core":
        expected = start + 2 * STEP + STEP + max_insig * STEP
    else:
        expected = start + 2 * STEP  # m3_0; plan fixes the fall-through

    assert final == pytest.approx(expected, abs=1e-12), (
        f"runner={runner.kind} expected {expected}, got {final}"
    )
    # The "restore" move target appears in the move log for both paths.
    assert pytest.approx(start + 2 * STEP, abs=1e-12) in [
        pytest.approx(v, abs=1e-12) for v in env.mirror.move_log
    ]
    # Diagnostic is retracted at the end on both paths.
    assert env.diag.move_log[-1] == DIAG_OUT


def test_backlash_unwind_is_first(env, runner):
    """The first two mirror moves implement the backlash unwind."""
    env.set_peak(_gaussian(center=0.0, width=10 * STEP), noise=1e-6, seed=4)
    start = -3 * STEP
    env.mirror.set(start).wait()
    env.mirror.move_log.clear()

    runner(env)

    assert env.mirror.move_log[0] == pytest.approx(start - 2 * STEP)
    assert env.mirror.move_log[1] == pytest.approx(start + 2 * STEP)


def test_diagnostic_inserted_and_retracted_on_success(env, runner):
    """Diagnostic is moved to diag_in first and diag_out last."""
    env.set_peak(_gaussian(center=0.0, width=10 * STEP), noise=1e-6, seed=5)
    env.mirror.set(-3 * STEP).wait()
    env.mirror.move_log.clear()

    runner(env)

    assert env.diag.move_log[0] == DIAG_IN
    assert env.diag.move_log[-1] == DIAG_OUT
    # Only one in / one out per invocation.
    assert env.diag.move_log.count(DIAG_IN) == 1
    assert env.diag.move_log.count(DIAG_OUT) == 1


def test_diagnostic_retracted_on_giveup(env, runner):
    """Diagnostic is also retracted when the algorithm gives up."""
    env.set_peak(lambda p: 0.0, noise=0.0, seed=6)
    env.mirror.set(0.0).wait()
    env.mirror.move_log.clear()

    runner(env)

    assert env.diag.move_log[0] == DIAG_IN
    assert env.diag.move_log[-1] == DIAG_OUT


def test_settle_after_every_mirror_move(env, runner):
    """Every mirror move after the backlash unwind is followed by a settle.

    Both implementations issue ``sleep(settle_time)`` after the backlash
    unwind, after the first ``+step``, and once per iteration of the
    direction-search and climb loops. We pin the relationship between
    the count of settle-time sleeps and the count of mirror moves.
    """
    env.set_peak(_gaussian(center=0.0, width=10 * STEP), noise=1e-6, seed=7)
    env.mirror.set(-3 * STEP).wait()
    env.mirror.move_log.clear()
    env.sleeps.clear()

    runner(env)

    n_settle = sum(1 for t in env.sleeps if t == 3.0)
    n_moves = len(env.mirror.move_log)
    # The first backlash move has no settle; every other move does, except
    # the optional final "step back" inside the climb loop and the diag-out
    # move (which is not a mirror move). So settles == mirror moves - 1 - k
    # for some 0 <= k <= 1.
    assert n_moves - 2 <= n_settle <= n_moves - 1, (
        f"settle count {n_settle} not consistent with {n_moves} mirror moves"
    )


def test_sample_count_is_multiple_of_n_samples(env, runner):
    """Signal reads are always issued in batches of ``n_samples``."""
    env.set_peak(_gaussian(center=0.0, width=10 * STEP), noise=1e-6, seed=8)
    env.mirror.set(-3 * STEP).wait()
    env.mirror.move_log.clear()

    n_samples = 10
    runner(env, n_samples=n_samples)

    count = _sample_count(env, runner)
    assert count > 0
    assert count % n_samples == 0, (
        f"read count {count} is not a multiple of {n_samples}"
    )


def test_csv_row_on_success(env, runner, tmp_path):
    """One row appended on success with the expected column order."""
    env.set_peak(_gaussian(center=0.0, width=10 * STEP), noise=1e-6, seed=9)
    env.mirror.set(-3 * STEP).wait()
    env.mirror.move_log.clear()

    csv_path = tmp_path / "log.csv"
    eng, pgm_energy, pgm_focus = 850.0, 851.23, 12.34
    final = runner(
        env,
        csv_path=str(csv_path),
        eng=eng,
        pgm_energy=pgm_energy,
        pgm_focus=pgm_focus,
    )

    with open(csv_path) as f:
        rows = list(csv.reader(f))

    assert len(rows) == 1, f"expected exactly one row, got {rows!r}"
    row = rows[0]
    assert len(row) == 5
    assert float(row[0]) == pytest.approx(eng)
    assert float(row[1]) == pytest.approx(round(pgm_energy, 2))
    assert float(row[2]) == pytest.approx(round(pgm_focus, 2))
    # row[3] is the most recent Au1 average; just verify it parses.
    float(row[3])
    assert float(row[4]) == pytest.approx(round(final, 5))


def test_asymmetric_extra_step_when_direction_negative(env, runner):
    """Lines 188-192: when direction == -1, the extra step is 2*step.

    Construct a Gaussian peak in the negative direction so the first
    comparison sees Au1 < Au0 (signal decreases as the mirror moves
    positive) and the algorithm picks direction = -1. The peak is real
    so the climb loop will terminate after overshooting.

    After the backlash unwind the mirror is at ``start + 2*step``.
    The direction-search probe moves to ``start + 3*step``.
    With direction == -1 the asymmetric "extra step" branch uses
    ``2 * direction * step`` instead of ``1 * direction * step``, so the
    next move target is ``(start + 3*step) + 2*(-1)*step == start + step``.
    """
    env.set_peak(_gaussian(center=-10 * STEP, width=4 * STEP), noise=1e-9, seed=10)
    start = 0.0
    env.mirror.set(start).wait()
    env.mirror.move_log.clear()

    runner(env)

    # Move sequence is:
    #   0: start - 2*step       (backlash)
    #   1: start + 2*step       (backlash; mirror now at +2*step)
    #   2: start + 3*step       (first +step probe -> direction-search seed)
    #   3: start + step         (extra step: 2*direction*step with direction=-1)
    assert env.mirror.move_log[3] == pytest.approx(start + STEP), (
        f"expected extra step of 2*step; move_log={env.mirror.move_log[:5]}"
    )


# ---------------------------------------------------------------------------
# Plan-only tests: behaviors guaranteed by the Bluesky refactor but not
# by the legacy core extraction.
# ---------------------------------------------------------------------------


def test_diag_retracted_on_exception_plan_only(env):
    """``finalize_wrapper`` guarantees diag retract even if the plan raises.

    The legacy ``m3_adjust.py`` retracts the diagnostic only at the
    bottom of its execution path. If something raised mid-algorithm the
    diagnostic would be left in the beam. The Bluesky refactor wraps the
    body in ``bpp.finalize_wrapper`` so the retract is unconditional.
    """
    # Cause _sample to raise after a few reads by toggling the peak_fn.
    calls = {"n": 0}

    def boom(p):
        calls["n"] += 1
        if calls["n"] > 5:
            raise RuntimeError("simulated detector failure")
        return 0.0

    env.set_peak(boom, noise=0.0, seed=42)
    env.mirror.set(0.0).wait()
    env.mirror.move_log.clear()

    with pytest.raises(RuntimeError, match="simulated detector failure"):
        env.RE(m3_adjust(
            mirror=env.mirror,
            signal=env.signal,
            diag=env.diag,
            diag_in=DIAG_IN,
            diag_out=DIAG_OUT,
            step=STEP,
            n_samples=10,
            sample_delay=0.1,
            settle_time=3.0,
        ))

    assert env.diag.move_log[0] == DIAG_IN
    assert env.diag.move_log[-1] == DIAG_OUT
