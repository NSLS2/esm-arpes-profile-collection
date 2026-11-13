import asyncio
import os
import sys
import threading
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch, MagicMock
import pytest
import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from bluesky import RunEngine
from bluesky.utils import RunEngineInterrupted
import matplotlib

# This is needed to prevent matplotlib from trying to use the X server
matplotlib.use('Agg')


@pytest.fixture
def mock_all_ophyd_devices():
    """
    Mock EpicsSignalBase methods to prevent any EPICS connections.
    All signals return 0 for reads and do nothing for writes.
    """
    import ophyd

    @classmethod
    def cls_noop(cls, *args, **kwargs):
        return
    
    def noop(self, *args, **kwargs):
        return
    
    def mock_get(self, *args, **kwargs):
        return 0
    
    def mock_subscribe(self, *args, **kwargs):
        return 0

    # Save originals
    originals = {
        'Device.wait_for_connection': ophyd.Device.wait_for_connection,
        'EpicsSignal.wait_for_connection': ophyd.signal.EpicsSignal.wait_for_connection,
        'EpicsSignalBase.wait_for_connection': ophyd.signal.EpicsSignalBase.wait_for_connection,
        'EpicsSignalBase.get': ophyd.signal.EpicsSignalBase.get,
        'EpicsSignalBase.put': ophyd.signal.EpicsSignalBase.put,
        'EpicsSignalBase.subscribe': ophyd.signal.EpicsSignalBase.subscribe,
        'EpicsSignalBase.set': ophyd.signal.EpicsSignalBase.set,
        'EpicsSignalBase.set_defaults': ophyd.signal.EpicsSignalBase.set_defaults,
    }

    # Apply mocks
    ophyd.Device.wait_for_connection = noop
    ophyd.signal.EpicsSignal.wait_for_connection = noop
    ophyd.signal.EpicsSignalBase.wait_for_connection = noop
    ophyd.signal.EpicsSignalBase.get = mock_get
    ophyd.signal.EpicsSignalBase.put = noop
    ophyd.signal.EpicsSignalBase.subscribe = mock_subscribe
    ophyd.signal.EpicsSignalBase.set = noop
    ophyd.signal.EpicsSignalBase.set_defaults = cls_noop
    
    yield
    
    # Restore originals
    ophyd.Device.wait_for_connection = originals['Device.wait_for_connection']
    ophyd.signal.EpicsSignal.wait_for_connection = originals['EpicsSignal.wait_for_connection']
    ophyd.signal.EpicsSignalBase.wait_for_connection = originals['EpicsSignalBase.wait_for_connection']
    ophyd.signal.EpicsSignalBase.get = originals['EpicsSignalBase.get']
    ophyd.signal.EpicsSignalBase.put = originals['EpicsSignalBase.put']
    ophyd.signal.EpicsSignalBase.subscribe = originals['EpicsSignalBase.subscribe']
    ophyd.signal.EpicsSignalBase.set = originals['EpicsSignalBase.set']
    ophyd.signal.EpicsSignalBase.set_defaults = originals['EpicsSignalBase.set_defaults']


@pytest.fixture
def mock_services():
    with patch("redis.Redis", return_value=MagicMock()), \
         patch("tiled.client.from_profile", return_value=MagicMock()), \
         patch("tiled.client.from_uri", return_value=MagicMock()), \
         patch("pyOlog.SimpleOlogClient", return_value=MagicMock()):
        os.environ["TILED_BLUESKY_WRITING_API_KEY_ARPES"] = "<mocked_api_key>"
        yield
    del os.environ["TILED_BLUESKY_WRITING_API_KEY_ARPES"]

@pytest.fixture
def mock_nslsii():
    def mock_configure_base(ipython_user_ns, beamline_name, **kwargs):
        ipython_user_ns['RE'] = MagicMock()
        ipython_user_ns['db'] = MagicMock()
        ipython_user_ns['sd'] = MagicMock()

    with patch('nslsii.configure_base', side_effect=mock_configure_base):
        yield


@pytest.fixture
def startup_dir():
    profile_dir = Path(__file__).parent.parent
    startup_dir = profile_dir / "startup"
    sys.path.insert(0, str(startup_dir))
    yield startup_dir
    sys.path.remove(str(startup_dir))


@pytest.fixture
def startup_shell(mock_all_ophyd_devices, mock_services, mock_nslsii, startup_dir):
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.core.profiledir import ProfileDir
    
    # Use the project directory as the profile directory (like --profile-dir=.)
    project_dir = startup_dir.parent
    profile_dir = ProfileDir(location=str(project_dir))
    
    shell = InteractiveShell.instance(profile_dir=profile_dir)
    
    try:
        for file in sorted(startup_dir.glob("*.py")):
            print(f"Running {file}")
            with open(file, "r") as f:
                code = f.read()
            result = shell.run_cell(code, store_history=True, silent=True)
            result.raise_error()
        globals().update(shell.user_ns)
        yield shell
    finally:
        InteractiveShell.clear_instance()


def test_startup_namespace(startup_shell):
    assert "RE" in globals(), "RunEngine not found"
    assert "db" in globals(), "Databroker not found"
    assert "sd" in globals(), "SupplementalData not found"


def test_move_to_retracts_other_epu_before_beamline_motion(startup_shell):
    namespace = startup_shell.user_ns

    for epu, other_epu_name in (("57", "EPU105"), ("105", "EPU57")):
        selected_epu = namespace[f"EPU{epu}"]
        with patch.object(selected_epu.gap, "safe_to_actuate", return_value=False):
            plan = namespace["Eph"].move_to(200, grating="800", EPU=epu)
            first_message = next(plan)

        assert first_message.command == "set"
        assert first_message.obj is namespace[other_epu_name].gap
        assert first_message.args == (220,)


def test_move_to_skips_retraction_when_other_epu_is_out(startup_shell):
    namespace = startup_shell.user_ns
    selected_epu = namespace["EPU57"]

    with patch.object(selected_epu.gap, "safe_to_actuate", return_value=True):
        plan = namespace["Eph"].move_to(200, grating="800", EPU="57")
        first_message = next(plan)

    assert first_message.obj is not selected_epu.gap.other


@pytest.mark.parametrize(
    "device_name, attr",
    [
        ("FEslit", "h_gap"),
        ("FEslit", "h_center"),
        ("EPU57", "gap"),
        ("EPU105", "phase"),
    ],
)
def test_gateway_positioners_move_and_stop_without_put_callbacks(
    startup_shell, device_name, attr
):
    positioner = getattr(startup_shell.user_ns[device_name], attr)
    signals = [
        sig
        for sig in (positioner.setpoint, positioner.actuate, positioner.stop_signal)
        if sig is not None
    ]

    with ExitStack() as stack:
        puts = [stack.enter_context(patch.object(sig, "put")) for sig in signals]
        positioner.set(1.0)
        positioner.stop()

    assert puts[0].call_args.args == (1.0,)
    calls = [call for put in puts for call in put.call_args_list]
    assert len(calls) == len(signals)
    for call in calls:
        assert not call.kwargs.get("wait", False)
        assert not call.kwargs.get("use_complete", False)
        assert call.kwargs.get("callback") is None


class _SweepMotor:
    """Manipulator axis stand-in.

    A move to the current position finishes at once. Any other move is a
    sweep that stays in flight until stop(), or arrives on its own after
    ``SWEEP_TIME`` so a flyer that never stops it fails a test instead of
    hanging it.
    """

    SWEEP_TIME = 5

    def __init__(self, name):
        from ophyd import Signal

        self.name = name
        self.velocity = Signal(name=f"{name}_velocity", value=1.0)
        self.user_readback = Signal(name=f"{name}_user_readback", value=0.0)
        self.commands = []
        self.arrivals = []  # targets of sweeps that ran to completion
        self.sweeps = threading.Semaphore(0)  # released when a sweep starts
        self.in_flight = None
        self._lock = threading.Lock()
        self._timer = None

    def set(self, target):
        from ophyd.status import StatusBase

        self.commands.append(target)
        status = StatusBase()
        if target == self.user_readback.get():
            status.set_finished()
            return status
        with self._lock:
            self.in_flight = status
            self._timer = threading.Timer(self.SWEEP_TIME, self._arrive, (target, status))
            self._timer.daemon = True
            self._timer.start()
        self.sweeps.release()
        return status

    def _arrive(self, target, status):
        with self._lock:
            if self.in_flight is not status:
                return
            self.in_flight = None
            self.arrivals.append(target)
        status.set_finished()

    def stop(self, *, success=False):
        with self._lock:
            status, self.in_flight = self.in_flight, None
            if self._timer is not None:
                self._timer.cancel()
        if status is not None:
            status.set_finished()


def _wait_for_sweep(motor):
    def started():
        return asyncio.to_thread(motor.sweeps.acquire, timeout=5)

    (task,) = yield from bps.wait_for([started])
    assert task.result(), f"{motor.name} never started a sweep"


def _fly_scan(namespace, x, y, inner):
    trajectory = ([x, y], [(0.0, 10.0), (0.0, 1.0, 0.5)])
    return namespace["fly_during_speed"](inner, trajectory, {x: 2.0})


@bpp.run_decorator()
def _pause_mid_sweep(x):
    yield from _wait_for_sweep(x)
    yield from bps.checkpoint()
    yield from bps.pause()
    yield from _wait_for_sweep(x)


def _pause_mid_sweep_fly_scan(namespace, x, y):
    return _fly_scan(namespace, x, y, _pause_mid_sweep(x))


def test_fly_scan_abort_stops_manipulator_mid_sweep(startup_shell):
    x, y = _SweepMotor("x"), _SweepMotor("y")
    RE = RunEngine({})
    stops = []
    RE.subscribe(lambda name, doc: stops.append(doc) if name == "stop" else None)

    with pytest.raises(RunEngineInterrupted):
        RE(_pause_mid_sweep_fly_scan(startup_shell.user_ns, x, y))
    assert x.in_flight is None
    commands_at_pause = list(x.commands)

    RE.abort()

    assert x.in_flight is None
    assert x.arrivals == []
    assert x.commands == commands_at_pause
    assert stops[0]["exit_status"] == "abort"


def test_fly_scan_resumes_interrupted_sweep_and_stops_at_run_end(startup_shell):
    x, y = _SweepMotor("x"), _SweepMotor("y")
    RE = RunEngine({})
    stops = []
    RE.subscribe(lambda name, doc: stops.append(doc) if name == "stop" else None)

    with pytest.raises(RunEngineInterrupted):
        RE(_pause_mid_sweep_fly_scan(startup_shell.user_ns, x, y))
    assert x.in_flight is None

    RE.resume()

    assert x.commands.count(10.0) == 2  # the interrupted sweep is re-commanded
    assert 1.0 not in y.commands
    assert x.in_flight is None
    assert x.arrivals == []  # stopped at run end, not left to finish the sweep
    assert stops[0]["exit_status"] == "success"


def test_fly_scan_halt_stops_manipulator_mid_sweep(startup_shell):
    x, y = _SweepMotor("x"), _SweepMotor("y")
    RE = RunEngine({})
    swept = []

    def halt_mid_sweep():
        swept.append(x.sweeps.acquire(timeout=5))
        RE.halt()

    @bpp.run_decorator()
    def inner():
        yield from bps.sleep(30)  # interrupted by the halt

    halter = threading.Thread(target=halt_mid_sweep)
    halter.start()
    with pytest.raises(RunEngineInterrupted):
        RE(_fly_scan(startup_shell.user_ns, x, y, inner()))
    halter.join()

    assert swept == [True]
    assert x.in_flight is None
    assert x.arrivals == []
