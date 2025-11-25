import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import matplotlib

# This is needed to prevent matplotlib from trying to use the X server
matplotlib.use('Agg')


@pytest.fixture(scope='function', autouse=True)
def mock_all_ophyd_devices():
    """
    Mock EpicsSignalBase methods to prevent any EPICS connections.
    All signals return 0 for reads and do nothing for writes.
    """
    import ophyd
    
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
    }

    # Apply mocks
    ophyd.Device.wait_for_connection = noop
    ophyd.signal.EpicsSignal.wait_for_connection = noop
    ophyd.signal.EpicsSignalBase.wait_for_connection = noop
    ophyd.signal.EpicsSignalBase.get = mock_get
    ophyd.signal.EpicsSignalBase.put = noop
    ophyd.signal.EpicsSignalBase.subscribe = mock_subscribe
    ophyd.signal.EpicsSignalBase.set = noop
    
    yield
    
    # Restore originals
    ophyd.Device.wait_for_connection = originals['Device.wait_for_connection']
    ophyd.signal.EpicsSignal.wait_for_connection = originals['EpicsSignal.wait_for_connection']
    ophyd.signal.EpicsSignalBase.wait_for_connection = originals['EpicsSignalBase.wait_for_connection']
    ophyd.signal.EpicsSignalBase.get = originals['EpicsSignalBase.get']
    ophyd.signal.EpicsSignalBase.put = originals['EpicsSignalBase.put']
    ophyd.signal.EpicsSignalBase.subscribe = originals['EpicsSignalBase.subscribe']
    ophyd.signal.EpicsSignalBase.set = originals['EpicsSignalBase.set']


@pytest.fixture(scope="function", autouse=True)
def mock_services():
    with patch("redis.Redis", return_value=MagicMock()), \
         patch("tiled.client.from_profile", return_value=MagicMock()), \
         patch("tiled.client.from_uri", return_value=MagicMock()), \
         patch("pyOlog.SimpleOlogClient", return_value=MagicMock()):
        os.environ["TILED_BLUESKY_WRITING_API_KEY_ARPES"] = "<mocked_api_key>"
        yield

@pytest.fixture(scope='function', autouse=True)
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


def test_startup(startup_dir):
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.core.profiledir import ProfileDir
    
    # Use the project directory as the profile directory (like --profile-dir=.)
    project_dir = startup_dir.parent
    profile_dir = ProfileDir(location=str(project_dir))
    
    shell = InteractiveShell.instance(profile_dir=profile_dir)
    
    try:
        for file in sorted(startup_dir.glob("*.py")):
            with open(file, "r") as f:
                code = f.read()
            result = shell.run_cell(code, store_history=True, silent=True)
            result.raise_error()
        
        ns = shell.user_ns
        
        # Bluesky core
        assert "RE" in ns, "RunEngine not found"
        assert "db" in ns, "Databroker not found"
        assert "sd" in ns, "SupplementalData not found"
    
    finally:
        InteractiveShell.clear_instance()