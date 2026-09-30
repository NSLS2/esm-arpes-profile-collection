import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
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
        from bluesky import RunEngine
        # A real RunEngine is required: init_devices() in 30-detectors.py connects
        # ophyd-async devices via call_in_bluesky_event_loop, and 04-tiled_writer.py
        # reads RE.md["data_session"].
        ipython_user_ns['RE'] = RunEngine(md={"data_session": "pass-000000", "cycle": "test"})
        ipython_user_ns['db'] = MagicMock()
        ipython_user_ns['sd'] = MagicMock()

    with patch('nslsii.configure_base', side_effect=mock_configure_base):
        yield


@pytest.fixture
def mock_ophyd_async_devices():
    """Force ophyd-async devices to connect in mock mode (no CA traffic)."""
    from ophyd_async.core import Device

    original_connect = Device.connect

    async def mock_connect(self, mock=False, timeout=10.0, force_reconnect=False):
        return await original_connect(self, mock=True)

    with patch.object(Device, "connect", mock_connect):
        yield


@pytest.fixture
def startup_dir():
    profile_dir = Path(__file__).parent.parent
    startup_dir = profile_dir / "startup"
    sys.path.insert(0, str(startup_dir))
    yield startup_dir
    sys.path.remove(str(startup_dir))


@pytest.fixture
def startup_shell(mock_all_ophyd_devices, mock_ophyd_async_devices, mock_services, mock_nslsii, startup_dir):
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


def test_qem_hint_fields(startup_shell):
    qem07 = globals()["qem07"]
    assert qem07.hints["fields"] == [f"qem07-current-{i}-mean_value" for i in range(1, 5)]


def test_qem_count_emits_current_means(startup_shell):
    import asyncio
    from bluesky.plans import count
    from ophyd_async.core import callback_on_mock_put, set_mock_value

    qem07, RE = globals()["qem07"], globals()["RE"]

    def clear_after_start(value, **kwargs):
        # Mock-mode Acquire never self-clears; complete it after the RE sets it.
        if value:
            asyncio.get_running_loop().call_soon(set_mock_value, qem07.driver.acquire, False)

    docs = []
    with callback_on_mock_put(qem07.driver.acquire, clear_after_start):
        RE(count([qem07]), lambda name, doc: docs.append((name, doc)))
    event = next(doc for name, doc in docs if name == "event")
    assert {f"qem07-current-{i}-mean_value" for i in range(1, 5)} <= set(event["data"])
