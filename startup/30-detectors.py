from pathlib import Path
from typing import Optional
import time as ttime
from itertools import count as ccount

import numpy as np
from bluesky.protocols import Readable, WritesExternalAssets
from bluesky.utils import SyncOrAsyncIterator, Asset
from event_model import DataKey, compose_resource
from ophyd import Kind
from ophyd.quadem import QuadEM  # , QuadEMPort  # TODO in the future once it's in ophyd
from ophyd import (
    Device,
    EpicsSignalRO,
    EpicsSignal,
    Component as Cpt,
    DynamicDeviceComponent as DDCpt,
    Signal,
    Staged,
)
from ophyd import AreaDetector, SingleTrigger, HDF5Plugin, TIFFPlugin
from ophyd.status import Status
from ophyd.areadetector.filestore_mixins import FileStoreHDF5IterativeWrite
from ophyd.areadetector.filestore_mixins import FileStoreTIFFIterativeWrite
from ophyd.areadetector import (
    ADComponent as ADCpt,
    EpicsSignalWithRBV,
    ImagePlugin,
    StatsPlugin,
    DetectorBase,
    ADBase,
    SingleTrigger,
    ROIPlugin,
    ProcessPlugin,
    TransformPlugin,
)
from ophyd.areadetector.plugins import (
    ImagePlugin_V33,
    StatsPlugin_V33
)

class HDF5PluginWithFileStore(HDF5Plugin, FileStoreHDF5IterativeWrite):

    def get_frames_per_point(self):
        # return self.num_capture.get()
        return 1

    # AD v2.2.0 (at least) does not have this. It is present in v1.9.1.
    # file_number_sync = None


class TIFFPluginWithFileStore(TIFFPlugin, FileStoreTIFFIterativeWrite):
    pass


# TODO: replace from one in the future ophyd
class QuadEMPort(ADBase):
    port_name = Cpt(Signal, value="")

    def __init__(self, port_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.port_name.put(port_name)


class ESMQuadEM(QuadEM):
    conf = Cpt(QuadEMPort, port_name="NSLS_EM")
    em_range = Cpt(EpicsSignalWithRBV, "Range", string=True)

    image = Cpt(ImagePlugin_V33, 'image1:')
    current1 = Cpt(StatsPlugin_V33, 'Current1:')
    current2 = Cpt(StatsPlugin_V33, 'Current2:')
    current3 = Cpt(StatsPlugin_V33, 'Current3:')
    current4 = Cpt(StatsPlugin_V33, 'Current4:')

    sum_all = Cpt(StatsPlugin_V33, 'SumAll:')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_sigs.update([(self.acquire_mode, "Single")])  # single mode
        self.configuration_attrs = [
            "integration_time",
            "averaging_time",
            "em_range",
            "num_averaged",
            "values_per_read",
        ]

    def set_primary(self, n, value=None):
        name_list = []
        if "All" in n:
            for k in self.read_attrs:
                getattr(self, k).kind = "normal"
            return

        for channel in n:
            cur = getattr(self, f"current{channel}")
            cur.kind |= Kind.normal
            cur.mean_value = Kind.hinted


class ESMbpm(ESMQuadEM):
    conf = Cpt(QuadEMPort, port_name="NSLS2_EM")


qem01 = ESMQuadEM("XF:21IDA-BI{EM:1}EM180:", name="qem01")
qem02 = ESMQuadEM("XF:21IDB-BI{EM:2}EM180:", name="qem02")
qem03 = ESMQuadEM("XF:21IDB-BI{EM:3}EM180:", name="qem03")
qem04 = ESMQuadEM("XF:21IDB-BI{EM:4}EM180:", name="qem04")
qem05 = ESMQuadEM("XF:21IDB-BI{EM:5}EM180:", name="qem05")

qem06 = ESMQuadEM("XF:21IDC-BI{EM:6}EM180:", name="qem06")
qem07 = ESMQuadEM("XF:21IDC-BI{EM:7}EM180:", name="qem07")
qem08 = ESMQuadEM("XF:21IDC-BI{EM:8}EM180:", name="qem08")



# qem09 not connected as of May 24, 2018
# qem09 = ESMQuadEM('XF:21IDC-BI{EM:9}EM180:', name='qem09')
##qem10 = ESMQuadEM("XF:21IDC-BI{EM:10}EM180:", name="qem10")
# qem11 not connected as of May 24, 2018
# qem11 = ESMQuadEM('XF:21IDC-BI{EM:11}EM180:', name='qem11')



qem12 = ESMQuadEM("XF:21IDC-BI{EM:12}EM180:", name="qem12")
qem13 = ESMQuadEM("XF:21IDC-BI{EM:13}EM180:", name="qem13")
#qem15 = ESMQuadEM("XF:21IDC-BI{EM:15}EM180:", name="qem15")
#qem16 = ESMQuadEM("XF:21IDC-BI{EM:16}EM180:", name="qem16")

xqem01 = ESMbpm("XF:21IDA-BI{EM:BPM01}", name="xqem01")


class MyDetector(SingleTrigger, AreaDetector):
    #    tiff = Cpt(TIFFPluginWithFileStore,
    #               suffix='TIFF1:',
    #               write_path_template='/nsls2/data/esm/legacy/image_files/',  # trailing slash!
    #               read_path_template='/nsls2/data/esm/legacy/image_files/',
    #               root='/direct'    )
    image = Cpt(ImagePlugin_V33, "image1:")
    stats1 = Cpt(StatsPlugin_V33, "Stats1:")
    stats2 = Cpt(StatsPlugin_V33, "Stats2:")
    stats3 = Cpt(StatsPlugin_V33, "Stats3:")
    stats4 = Cpt(StatsPlugin_V33, "Stats4:")
    stats5 = Cpt(StatsPlugin_V33, "Stats5:")
    trans1 = Cpt(TransformPlugin, "Trans1:")
    roi1 = Cpt(ROIPlugin, "ROI1:")
    roi2 = Cpt(ROIPlugin, "ROI2:")
    roi3 = Cpt(ROIPlugin, "ROI3:")
    roi4 = Cpt(ROIPlugin, "ROI4:")
    proc1 = Cpt(ProcessPlugin, "Proc1:")

    hdf5 = Cpt(
        HDF5PluginWithFileStore,
        suffix="HDF1:",
        write_path_template=f"/nsls2/data/esm/proposals/{RE.md["cycle"]}/{RE.md["data_session"]}/assets/default/",  # trailing slash!
        root=f"/nsls2/data/esm/proposals/{RE.md["cycle"]}/{RE.md["data_session"]}/assets/default/",
    )

    def set_primary(self, n, value=None):
        if "All" in n:
            for channel in range(1, 5):
                stats = getattr(self, f'stats{channel}')
                stats.kind |= Kind.normal
                stats.total.kind = 'hinted'
                stats.min_value.kind = 'hinted'
                stats.max_value.kind = 'hinted'
            return

        if value is None:
            value = [['total']] * len(n)

        if len(value) != len(n):
            raise ValueError(f'The length of "n" list ({len(n)}) must be equal to'
                             'the length of "value" list ({len{value}})')

        for value_list, channel in zip(value, n):
            stats = getattr(self, f'stats{channel}')
            stats.kind |= Kind.normal
            for val in value_list:
                if val in ["max", "min"]:
                    val = f"{val}_value"
                    getattr(stats, val).kind = 'hinted'


def _convert_path_to_posix(path: Path) -> Path:
    """Assumes that the path is on a Windows machine with Z: drive."""
    # Convert to string to manipulate
    path_str = str(path)
    
    # Replace Z: with the target directory
    if path_str.startswith("Z:"):
        path_str = path_str.replace("Z:", "/nsls2/data3/esm/proposals", 1)
    else:
        return path
    
    # Convert backslashes to forward slashes for POSIX compatibility
    path_str = path_str.replace("\\", "/")
    
    return Path(path_str)

class SpectrumAnalyzer(Device, Readable):
    # Acquisition control
    acquire = Cpt(EpicsSignal, "ACQUIRE")
    acquisition_status = Cpt(EpicsSignalRO, "ACQ:STATUS")

    # Detector control
    det_off = Cpt(EpicsSignal, "DET:OFF")

    # Live data monitoring
    live_monitoring = Cpt(EpicsSignal, "LIVE:MONITORING")
    live_max_count = Cpt(EpicsSignalRO, "LIVE:MAX_COUNT")
    live_last_update = Cpt(EpicsSignalRO, "LIVE:LAST_UPDATE")
    live_max_count_threshold = Cpt(EpicsSignal, "LIVE:MAX_COUNT_THRESH")
    live_max_count_exceeded = Cpt(EpicsSignal, "LIVE:MAX_COUNT_EXCEEDED")
    live_max_count_avg_n = Cpt(EpicsSignal, "LIVE:MAX_COUNT_AVG_N")

    # Status and info
    connection_status = Cpt(EpicsSignalRO, "SYS:CONNECTED")
    last_sync = Cpt(EpicsSignalRO, "SYS:LAST_SYNC")
    sync = Cpt(EpicsSignal, "SYS:SYNC")

    # File writing
    file_capture = Cpt(EpicsSignal, "FILE:CAPTURE")
    file_name = Cpt(EpicsSignal, "FILE:NAME", string=True)
    file_path = Cpt(EpicsSignal, "FILE:PATH", string=True)
    num_captured = Cpt(EpicsSignalRO, "FILE:NUM_CAPTURED")
    num_processed = Cpt(EpicsSignalRO, "FILE:NUM_PROCESSED")

    # Detector parameters
    state = Cpt(EpicsSignalRO, "STATE", string=True)
    endX = Cpt(EpicsSignal, "ENDX")
    startY = Cpt(EpicsSignal, "STARTY")
    num_slice = Cpt(EpicsSignal, "NUM_SLICE")
    endY = Cpt(EpicsSignal, "ENDY")
    startX = Cpt(EpicsSignal, "STARTX")
    frames = Cpt(EpicsSignal, "FRAMES")
    num_steps = Cpt(EpicsSignal, "NUM_STEPS")
    pass_energy = Cpt(EpicsSignal, "PASS_ENERGY")
    lens_mode = Cpt(EpicsSignal, "LENS_MODE")
    num_scans = Cpt(EpicsSignal, "NUM_SCANS")
    reg_num = Cpt(EpicsSignal, "REG_NUM")
    tot_steps = Cpt(EpicsSignal, "TOT_STEPS")
    add_fms = Cpt(EpicsSignal, "ADD_FMS")
    act_scans = Cpt(EpicsSignalRO, "ACT_SCANS")
    dith_steps = Cpt(EpicsSignal, "DITH_STEPS")
    start_ke = Cpt(EpicsSignal, "START_KE")
    step_size = Cpt(EpicsSignal, "STEP_SIZE")
    end_ke = Cpt(EpicsSignal, "END_KE")
    spin_offs = Cpt(EpicsSignal, "SPIN_OFFS")
    width = Cpt(EpicsSignal, "WIDTH")
    center_ke = Cpt(EpicsSignal, "CENTER_KE")
    first_energy = Cpt(EpicsSignal, "FIRST_ENERGY")
    deflX = Cpt(EpicsSignal, "DEFLX")
    deflY = Cpt(EpicsSignal, "DEFLY")
    dbl10 = Cpt(EpicsSignal, "DBL10")
    acq_mode = Cpt(EpicsSignal, "ACQ_MODE")
    date_number = Cpt(EpicsSignal, "DATE_NUMBER")
    loc_det = Cpt(EpicsSignal, "LOC_DET")
    xtab = Cpt(EpicsSignal, "XTAB")
    spin = Cpt(EpicsSignal, "SPIN")
    reg_name = Cpt(EpicsSignal, "REG_NAME")
    name_string = Cpt(EpicsSignal, "NAME_STRING")
    generated_name = Cpt(EpicsSignal, "GENERATED_NAME")
    comment1 = Cpt(EpicsSignal, "COMMENT1")
    start_time = Cpt(EpicsSignal, "START_TIME")
    discr = Cpt(EpicsSignal, "DISCR")
    adc_mask = Cpt(EpicsSignal, "ADC_MASK")
    adc_offset = Cpt(EpicsSignal, "ADC_OFFSET")
    p_cnt_type = Cpt(EpicsSignal, "P_CNT_TYPE")
    pc_mask = Cpt(EpicsSignal, "PC_MASK")
    soft_bin_x = Cpt(EpicsSignal, "SOFT_BIN_X")
    soft_bin_y = Cpt(EpicsSignal, "SOFT_BIN_Y")
    escale_mult = Cpt(EpicsSignal, "ESCALE_MULT")
    escale_max = Cpt(EpicsSignal, "ESCALE_MAX")
    escale_min = Cpt(EpicsSignal, "ESCALE_MIN")
    yscale_mult = Cpt(EpicsSignal, "YSCALE_MULT")
    yscale_max = Cpt(EpicsSignal, "YSCALE_MAX")
    yscale_min = Cpt(EpicsSignal, "YSCALE_MIN")
    yscale_name = Cpt(EpicsSignal, "YSCALE_NAME")
    xscale_mult = Cpt(EpicsSignal, "XSCALE_MULT")
    xscale_max = Cpt(EpicsSignal, "XSCALE_MAX")
    xscale_min = Cpt(EpicsSignal, "XSCALE_MIN")
    xscale_name = Cpt(EpicsSignal, "XSCALE_NAME")
    psu_mode = Cpt(EpicsSignal, "PSU_MODE")
    over_r_arr = Cpt(EpicsSignal, "OVER_R_ARR")
    over_range = Cpt(EpicsSignal, "OVER_RANGE")

    _min_frames = 100
    """TCP server can't keep up with frame rate faster than this value in non-swept mode"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._status = None
        self._index = 0
        self._last_emitted_index = 0
        self._composer = None
        self._full_path = None

    def stage(self):
        if self.file_capture.get(as_string=True) == "On":
            raise RuntimeError(
                "File capture must be off to stage the detector, otherwise the file will be corrupted"
            )

        # Must be in standby to start
        if self.state.get(as_string=True) == "RUNNING":
            self.acquire.set(0).wait(3.0)

        # Must be live monitoring to start
        if self.live_monitoring.get(as_string=True) == "Off":
            self.live_monitoring.set("On").wait(3.0)

        # File capture must be on and then turned off at unstage
        self.stage_sigs.update(
            [
                (self.file_capture, 1),
            ]
        )

        # Frame rate can't be faster than 200ms in any mode except swept
        if (
            self.frames.get() < self._min_frames
            and self.acq_mode.get(as_string=True) != "Swept"
        ):
            self.stage_sigs.update(
                [(self.frames, self._min_frames)],
            )

        # Rebase the path to the assets directory of the current cycle & data session
        path_extension = self.file_path.get(as_string=True)
        full_path = f"Z:\\{RE.md["cycle"]}\\{RE.md["data_session"]}\\assets\\{self.name}\\{path_extension}"
        if path_extension.startswith("\\"):
            raise ValueError("File path must be an extension to the assets directory, not a full path. "
                f"This is for data security reasons. Full path would be '{full_path}', which is not allowed.")
        self.file_path.set(full_path)
        path = _convert_path_to_posix(Path(self.file_path.get()))
        file_name = Path(self.file_name.get())
        self._full_path = str(path / file_name)
        self._index = 0
        self._last_emitted_index = 0

        # Subscribe to state and live max count exceeded to
        # handle the acquisition status
        self.state.subscribe(self._state_changed, run=False)
        self.live_max_count_exceeded.subscribe(
            self._live_max_count_exceeded_monitor, run=False
        )
        return super().stage()

    def _state_changed(self, value=None, old_value=None, **kwargs):
        if (
            self._status is not None
            and value == "STANDBY"
            and (old_value == "RUNNING" or old_value == "MOVING")
        ):
            self._status.set_finished()
            self._index += 1
            self._status = None

    def _live_max_count_exceeded_monitor(self, value=None, **kwargs):
        if self._status is not None and value:
            self._status.set_exception(
                RuntimeError(
                    f"Max count safety limit exceeded: {self.live_max_count.get()} > {self.live_max_count_threshold.get()}"
                )
            )
            self._status = None

    def trigger(self):
        if self._staged != Staged.yes:
            raise RuntimeError(
                "This detector is not ready to trigger."
                "Call the stage() method before triggering."
            )

        self._status = Status()
        self.acquire.put(1)
        return self._status

    def unstage(self):
        if self.state.get(as_string=True) == "RUNNING":
            self.acquire.set(0).wait(3.0)
        self.det_off.set(1).wait(3.0)
        super().unstage()
        self.state.unsubscribe(self._state_changed)
        self.live_max_count_exceeded.unsubscribe(self._live_max_count_exceeded_monitor)
        self._composer = None

    @property
    def index(self) -> int:
        return self._index


class SpectrumAnalyzerFileStore(SpectrumAnalyzer, WritesExternalAssets):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._datum_uids = []
        self._asset_docs_cache = []
        self._point_counter = None

    def _generate_resource(self):
        self._composer = compose_resource(
            spec="A1_HDF5",
            root=str(Path(self._full_path).parent),
            resource_path=self._full_path,
            resource_kwargs={"frame_per_point": 1},
            path_semantics="posix",
        )
        self._asset_docs_cache.append(("resource", self._composer.resource_doc))

    def generate_datum(self):
        timestamp = time.time()
        i = next(self._point_counter)
        datum = self._composer.compose_datum({"point_number": i})
        self._datum_uids.append({"value": datum["datum_id"], "timestamp": timestamp})
        self._asset_docs_cache.append(("datum", datum))

    def stage(self):
        self._datum_uids = []
        ret = super().stage()
        self._generate_resource()
        self._point_counter = ccount()
        return ret

    def trigger(self):
        s = super().trigger()
        self.generate_datum()
        return s

    def unstage(self):
        self._point_counter = None
        return super().unstage()

    def describe(self) -> dict[str, DataKey]:
        describe = super().describe()
        describe.update(
            {
                f"{self.name}_image": DataKey(
                    source=f"{self._full_path}",
                    shape=(1, self.num_slice.get(), self.num_steps.get()),
                    dtype="array",
                    dtype_numpy=np.dtype(np.uint32).str,
                    external="FILESTORE:",
                ),
            }
        )
        return describe

    def read(self):
        res = super().read()
        res[f"{self.name}_image"] = self._datum_uids[-1]
        return res

    def collect_asset_docs(self) -> SyncOrAsyncIterator[Asset]:
        items = list(self._asset_docs_cache)
        self._asset_docs_cache = []
        for item in items:
            yield item


mbs = SpectrumAnalyzerFileStore("XF:21ID1-ES{A1Soft}", name="mbs")

Diag1_CamH = MyDetector("XF:21IDA-BI{Diag:1-Cam:H}", name="Diag1_CamH")
Diag1_CamH.hdf5.write_path_template = f"/nsls2/data/esm/proposals/{RE.md["cycle"]}/{RE.md["data_session"]}/assets/{Diag1_CamH.name}/"

#Diag1_CamV = MyDetector("XF:21IDA-BI{Diag:1-Cam:V}", name="Diag1_CamV")
#Diag1_CamV.hdf5.write_path_template = f"/nsls2/data/esm/proposals/{RE.md["cycle"]}/{RE.md["data_session"]}/assets/{Diag1_CamV.name}/"

Lock23A_CamEA3_1 = MyDetector('XF:21IDD-BI{ES-Cam:3}', name='Lock23A_CamEA3_1')
Lock23A_CamEA3_1.hdf5.write_path_template = f"/nsls2/data/esm/proposals/{RE.md["cycle"]}/{RE.md["data_session"]}/assets/{Lock23A_CamEA3_1.name}/"

#Lock23A_CamEA3_1 = MyDetector(
#    "XF:21IDD-BI{Lock2:3A-Cam:EA3_1}", name="Lock23A_CamEA3_1"
#)
#Lock23A_CamEA3_1.hdf5.write_path_template = f"/nsls2/data/esm/proposals/{RE.md["cycle"]}/{RE.md["data_session"]}/assets/{Lock23A_CamEA3_1.name}/"


#Lock14A_CamEA4_1 = MyDetector(
#    "XF:21IDD-BI{Lock1:4A-Cam:EA4_1}", name="Lock14A_CamEA4_1"
#)
#Lock14A_CamEA4_1.hdf5.write_path_template = f"/nsls2/data/esm/proposals/{RE.md["cycle"]}/{RE.md["data_session"]}/assets/{Lock14A_CamEA4_1.name}/"

#Prep2A_CamEA2_1 = MyDetector("XF:21IDD-BI{Prep:2A-Cam:EA2_1}", name="Prep2A_CamEA2_1")
#Prep2A_CamEA2_1.hdf5.write_path_template = f"/nsls2/data/esm/proposals/{RE.md["cycle"]}/{RE.md["data_session"]}/assets/{Prep2A_CamEA2_1.name}/"

Mir3_Cam10_U_1 = MyDetector("XF:21IDB-BI{Mir:3-Cam:6}", name="Mir3_Cam10_U_1")
Mir3_Cam10_U_1.hdf5.write_path_template = f"/nsls2/data/esm/proposals/{RE.md["cycle"]}/{RE.md["data_session"]}/assets/{Mir3_Cam10_U_1.name}/"


# BC1_Diag1_U_1 = MyDetector('XF:21IDA-BI{BC:1-Diag:1_U_1}', name='BC1_Diag1_U_1')
# BC1_Diag1_U_1.hdf5.write_path_template = f"/nsls2/data/esm/proposals/{RE.md["cycle"]}/{RE.md["data_session"]}/assets/{BC1_Diag1_U_1.name}/"

#Anal1A_Camlens = MyDetector("XF:21IDD-BI{Anal:1A-Cam:lens}", name="Anal1A_Camlens")
#Anal1A_Camlens.hdf5.write_path_template = f"/nsls2/data/esm/proposals/{RE.md["cycle"]}/{RE.md["data_session"]}/assets/{Anal1A_Camlens.name}/"

#Anal1A_Cambeam = MyDetector("XF:21IDD-BI{Anal:1A-Cam:beam}", name="Anal1A_Cambeam")
#Anal1A_Cambeam.hdf5.write_path_template = f"/nsls2/data/esm/proposals/{RE.md["cycle"]}/{RE.md["data_session"]}/assets/{Anal1A_Cambeam.name}/"

Prep2A_CamLEED = MyDetector("XF:21IDD-BI{ES-Cam:9}", name="Prep2A_CamLEED")
Prep2A_CamLEED.hdf5.write_path_template = f"/nsls2/data/esm/proposals/{RE.md["cycle"]}/{RE.md["data_session"]}/assets/{Prep2A_CamLEED.name}/"

#Prep2A_Camevap1 = MyDetector("XF:21IDD-BI{Prep:2A-Cam:evap1}", name="Prep2A_Camevap1")
#Prep2A_Camevap1.hdf5.write_path_template = f"/nsls2/data/esm/proposals/{RE.md["cycle"]}/{RE.md["data_session"]}/assets/{Prep2A_Camevap1.name}/"

#Prep2A_Camevap2 = MyDetector("XF:21IDD-BI{Prep:2A-Cam:evap2}", name="Prep2A_Camevap2")
#Prep2A_Camevap2.hdf5.write_path_template = f"/nsls2/data/esm/proposals/{RE.md["cycle"]}/{RE.md["data_session"]}/assets/{Prep2A_Camevap2.name}/"

LOWT_5A_Cam1 = MyDetector("XF:21IDD-OP{ES-Cam:16}", name="LOWT_5A_Cam1")
LOWT_5A_Cam1.hdf5.write_path_template = f"/nsls2/data/esm/proposals/{RE.md["cycle"]}/{RE.md["data_session"]}/assets/{LOWT_5A_Cam1.name}/"

#LOWT_5A_Cam2 = MyDetector("XF:21IDD-OP{LOWT:5A-Cam:2}", name="LOWT_5A_Cam2")
#LOWT_5A_Cam2.hdf5.write_path_template = f"/nsls2/data/esm/proposals/{RE.md["cycle"]}/{RE.md["data_session"]}/assets/{LOWT_5A_Cam2.name}/"

#BTA2_Cam1 = MyDetector("XF:21IDD-OP{BT:A2-Cam:1}", name="BTA2_Cam1")
#BTA2_Cam1.hdf5.write_path_template = f"/nsls2/data/esm/proposals/{RE.md["cycle"]}/{RE.md["data_session"]}/assets/{BTA2_Cam1.name}/"

#BTB2_Cam1 = MyDetector("XF:21IDD-OP{BT:B2-Cam:1}", name="B2BT_Cam1")
#BTB2_Cam1.hdf5.write_path_template = f"/nsls2/data/esm/proposals/{RE.md["cycle"]}/{RE.md["data_session"]}/assets/{BTB2_Cam1.name}/"

#PEEM1B_Cam1 = MyDetector("XF:21IDD-OP{PEEM:1B-Cam:1}", name="PEEM1B_Cam1")
#PEEM1B_Cam1.hdf5.write_path_template = f"/nsls2/data/esm/proposals/{RE.md["cycle"]}/{RE.md["data_session"]}/assets/{PEEM1B_Cam1.name}/"

#BTB5_Cam1 = MyDetector("XF:21IDD-OP{BT:B5-Cam:1}", name="BTB5_Cam1")
#BTB5_Cam1.hdf5.write_path_template = f"/nsls2/data/esm/proposals/{RE.md["cycle"]}/{RE.md["data_session"]}/assets/{BTB5_Cam1.name}/"

all_standard_pros = [
    Diag1_CamH,
#    Diag1_CamV,
#    Lock23A_CamEA3_1,
#    Lock14A_CamEA4_1,
#    Prep2A_CamEA2_1,
    Mir3_Cam10_U_1,
#    Anal1A_Camlens,
#    Anal1A_Cambeam,
    Prep2A_CamLEED,
#    Prep2A_Camevap1,
#    Prep2A_Camevap2,
    LOWT_5A_Cam1,
#    LOWT_5A_Cam2,
#    BTA2_Cam1,
#    BTB2_Cam1,
#    PEEM1B_Cam1,
#    BTB5_Cam1,
]

for camera in all_standard_pros:
    camera.read_attrs = ["stats1", "stats2", "stats3", "stats4", "stats5", "hdf5"]
    camera.hdf5.read_attrs = []
    # camera.tiff.read_attrs = []  # leaving just the 'image'
    for stats_name in ["stats1", "stats2", "stats3", "stats4", "stats5"]:
        stats_plugin = getattr(camera, stats_name)
        stats_plugin.read_attrs = ["total", "min_value", "max_value"]
        camera.stage_sigs[stats_plugin.blocking_callbacks] = 1

    camera.stage_sigs[camera.roi1.blocking_callbacks] = 1
    camera.stage_sigs[camera.trans1.blocking_callbacks] = 1
    camera.stage_sigs[camera.cam.trigger_mode] = "Fixed Rate"
    camera.set_primary(["All"])




class flowmeter(Device):
	value = Cpt(EpicsSignal, 'XF:21ID1-ES{IOLogik:1}AI:1-I')

flowm = flowmeter(name='flowm')
