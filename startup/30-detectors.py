from pathlib import Path
from typing import Optional

import numpy as np
from bluesky.protocols import WritesStreamAssets, Readable
from bluesky.utils import SyncOrAsyncIterator, StreamAsset
from event_model import compose_stream_resource, DataKey
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
        write_path_template="/nsls2/data/esm/legacy/image_files/",  # trailing slash!
        root="/nsls2/data/esm/legacy/",
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
        path_str = path_str.replace("Z:", "/nsls2/data3/esm/legacy", 1)
    else:
        return path
    
    # Convert backslashes to forward slashes for POSIX compatibility
    path_str = path_str.replace("\\", "/")
    
    return Path(path_str)


class SpectrumAnalyzer(Device, WritesStreamAssets, Readable):
    # Acquisition control
    acquire = Cpt(EpicsSignal, "ACQUIRE")
    acquisition_status = Cpt(EpicsSignalRO, "ACQ:STATUS")

    # Status and info
    connection_status = Cpt(EpicsSignalRO, "SYS:CONNECTED")
    last_sync = Cpt(EpicsSignalRO, "SYS:LAST_SYNC")
    sync = Cpt(EpicsSignal, "SYS:SYNC")

    # File writing
    file_capture = Cpt(EpicsSignal, "FILE:CAPTURE")
    file_name = Cpt(EpicsSignal, "FILE:NAME")
    file_path = Cpt(EpicsSignal, "FILE:PATH")
    file_status = Cpt(EpicsSignalRO, "FILE:STATUS")
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_sigs.update(
            [
                (self.acquire, 0),
                (self.file_capture, 1),
            ]
        )
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

        path = _convert_path_to_posix(Path(self.file_path.get()))
        file_name = Path(self.file_name.get())
        self._full_path = str(path / file_name)
        self._index = 0
        self._last_emitted_index = 0

        self.state.subscribe(self._stage_changed, run=False)
        return super().stage()

    def _stage_changed(self, value=None, old_value=None, **kwargs):
        if self._status is None:
            return
        if value == "STANDBY" and old_value == "RUNNING":
            self._status.set_finished()
            self._index += 1
            self._status = None

    def trigger(self):
        if self._staged != Staged.yes:
            raise RuntimeError(
                "This detector is not ready to trigger."
                "Call the stage() method before triggering."
            )
        self.acquire.set(1)
        self._status = Status()
        return self._status

    def describe(self) -> dict[str, DataKey]:
        describe = super().describe()
        describe.update(
            {
                f"{self.name}_image": DataKey(
                    source=f"{self._full_path}",
                    shape=(1, 1080, self.num_steps.get()),
                    dtype="array",
                    dtype_numpy=np.dtype(np.uint32).str,
                    external="STREAM:",
                ),
            }
        )
        return describe

    def get_index(self) -> int:
        return self._index

    def collect_asset_docs(
        self, index: Optional[int] = None
    ) -> SyncOrAsyncIterator[StreamAsset]:
        if index is not None:
            msg = f"Indexing is not supported for this detector, got: {index}, current index: {self.get_index()}"
            raise NotImplementedError(msg)

        index = self.get_index()
        if index:
            if not self._composer:
                self._composer = compose_stream_resource(
                    data_key=f"{self.name}_image",
                    mimetype="application/x-hdf5",
                    uri=f"file://{self._full_path}",
                    parameters={"dataset": "entry1/analyzer/data"},
                )
                yield "stream_resource", self._composer.stream_resource_doc

            if index >= self._last_emitted_index:
                indices = {
                    "start": self._last_emitted_index,
                    "stop": index,
                }
                self._last_emitted_index = index
                yield "stream_datum", self._composer.compose_stream_datum(indices)

    def unstage(self):
        super().unstage()
        self.state.unsubscribe(self._stage_changed)
        self._composer = None

spectrum_analyzer = SpectrumAnalyzer("XF:21ID1-ES{A1Soft}", name="spectrum_analyzer")

Diag1_CamH = MyDetector("XF:21IDA-BI{Diag:1-Cam:H}", name="Diag1_CamH")
Diag1_CamH.hdf5.write_path_template = "/nsls2/data/esm/legacy/image_files/cam01/"

#Diag1_CamV = MyDetector("XF:21IDA-BI{Diag:1-Cam:V}", name="Diag1_CamV")
#Diag1_CamV.hdf5.write_path_template = "/nsls2/data/esm/legacy/image_files/cam02/"

Lock23A_CamEA3_1 = MyDetector('XF:21IDD-BI{ES-Cam:3}', name='Lock23A_CamEA3_1')
Lock23A_CamEA3_1.hdf5.write_path_template = '/nsls2/data/esm/legacy/image_files/cam03/'

#Lock23A_CamEA3_1 = MyDetector(
#    "XF:21IDD-BI{Lock2:3A-Cam:EA3_1}", name="Lock23A_CamEA3_1"
#)
#Lock23A_CamEA3_1.hdf5.write_path_template = "/nsls2/data/esm/legacy/image_files/cam03/"


#Lock14A_CamEA4_1 = MyDetector(
#    "XF:21IDD-BI{Lock1:4A-Cam:EA4_1}", name="Lock14A_CamEA4_1"
#)
#Lock14A_CamEA4_1.hdf5.write_path_template = "/nsls2/data/esm/legacy/image_files/cam04/"

#Prep2A_CamEA2_1 = MyDetector("XF:21IDD-BI{Prep:2A-Cam:EA2_1}", name="Prep2A_CamEA2_1")
#Prep2A_CamEA2_1.hdf5.write_path_template = "/nsls2/data/esm/legacy/image_files/cam05/"

Mir3_Cam10_U_1 = MyDetector("XF:21IDB-BI{Mir:3-Cam:6}", name="Mir3_Cam10_U_1")
Mir3_Cam10_U_1.hdf5.write_path_template = "/nsls2/data/esm/legacy/image_files/cam06/"


# BC1_Diag1_U_1 = MyDetector('XF:21IDA-BI{BC:1-Diag:1_U_1}', name='BC1_Diag1_U_1')
# BC1_Diag1_U_1.hdf5.write_path_template = '/nsls2/data/esm/legacy/image_files/cam07/'

#Anal1A_Camlens = MyDetector("XF:21IDD-BI{Anal:1A-Cam:lens}", name="Anal1A_Camlens")
#Anal1A_Camlens.hdf5.write_path_template = "/nsls2/data/esm/legacy/image_files/cam07/"

#Anal1A_Cambeam = MyDetector("XF:21IDD-BI{Anal:1A-Cam:beam}", name="Anal1A_Cambeam")
#Anal1A_Cambeam.hdf5.write_path_template = "/nsls2/data/esm/legacy/image_files/cam08/"

Prep2A_CamLEED = MyDetector("XF:21IDD-BI{ES-Cam:9}", name="Prep2A_CamLEED")
Prep2A_CamLEED.hdf5.write_path_template = "/nsls2/data/esm/legacy/image_files/cam09/"

#Prep2A_Camevap1 = MyDetector("XF:21IDD-BI{Prep:2A-Cam:evap1}", name="Prep2A_Camevap1")
#Prep2A_Camevap1.hdf5.write_path_template = "/nsls2/data/esm/legacy/image_files/cam10/"

#Prep2A_Camevap2 = MyDetector("XF:21IDD-BI{Prep:2A-Cam:evap2}", name="Prep2A_Camevap2")
#Prep2A_Camevap2.hdf5.write_path_template = "/nsls2/data/esm/legacy/image_files/cam11/"

LOWT_5A_Cam1 = MyDetector("XF:21IDD-OP{ES-Cam:16}", name="LOWT_5A_Cam1")
LOWT_5A_Cam1.hdf5.write_path_template = "/nsls2/data/esm/legacy/image_files/cam14/"

#LOWT_5A_Cam2 = MyDetector("XF:21IDD-OP{LOWT:5A-Cam:2}", name="LOWT_5A_Cam2")
#LOWT_5A_Cam2.hdf5.write_path_template = "/nsls2/data/esm/legacy/image_files/cam15/"

#BTA2_Cam1 = MyDetector("XF:21IDD-OP{BT:A2-Cam:1}", name="BTA2_Cam1")
#BTA2_Cam1.hdf5.write_path_template = "/nsls2/data/esm/legacy/image_files/cam16/"

#BTB2_Cam1 = MyDetector("XF:21IDD-OP{BT:B2-Cam:1}", name="B2BT_Cam1")
#BTB2_Cam1.hdf5.write_path_template = "/nsls2/data/esm/legacy/image_files/cam17/"

#PEEM1B_Cam1 = MyDetector("XF:21IDD-OP{PEEM:1B-Cam:1}", name="PEEM1B_Cam1")
#PEEM1B_Cam1.hdf5.write_path_template = "/nsls2/data/esm/legacy/image_files/cam18/"

#BTB5_Cam1 = MyDetector("XF:21IDD-OP{BT:B5-Cam:1}", name="BTB5_Cam1")
#BTB5_Cam1.hdf5.write_path_template = "/nsls2/xf21id/image_files/cam19/"

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
