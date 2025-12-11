from nslsii.devices import TwoButtonShutter
from ophyd import EpicsSignal,PVPositioner, EpicsSignalRO, EpicsMotor
from ophyd import Device, Component as Comp
from ophyd import Component as Cpt
from ophyd import DeviceStatus


MR_attrs = "'user_readback','user_setpoint','motor_is_moving','motor_done_move','motor_stop'"


# ---------------------------- Polarimeter -----------------------------
# Polarimeter
#
class Pol_mtr(Device):
    Rz = Comp(EpicsMotor,'Mtr', kind='hinted')

Pol = Pol_mtr('XF:21IDC-OP{Slt:1B-Pol:B1_1-Ax:B1_Rz}', name='Pol')


# ---------------------------- Load Lock -----------------------------
# Load Lock
#
class LL_mtr(Device):
    Claw_Trans = Comp(EpicsMotor,'2-Claw:EA3_1-Ax:T}Mtr', kind='hinted')
    Claw_Rotate = Comp(EpicsMotor,'2-Claw:EA3_1-Ax:R}Mtr', kind='hinted')
    Claw_Grab = Comp(EpicsMotor,'2-Claw:EA3_1-Ax:C}Mtr', kind='hinted')
    Dock_Trans = Comp(EpicsMotor,'1-Dock:EA4_1-Ax:F}Mtr', kind='hinted')

LL = LL_mtr('XF:21IDD-ES{LL:', name='LL')


#
# ---------------------------- LOW_TEMP Chamber -----------------------------
# Low_temp chamber:
#
class LT_mtr(Device):
   X = Comp(EpicsMotor,"-Ax:X}Mtr", kind='hinted')
   Y = Comp(EpicsMotor,"-Ax:Y}Mtr", kind='hinted')
   Z = Comp(EpicsMotor,"-Ax:Z}Mtr", kind='hinted')
   Ry =  Comp(EpicsMotor,"-Ax:R1}Mtr", kind='hinted')
   Rz =  Comp(EpicsMotor,"-Ax:R2}Mtr", kind='hinted')
   Rx =  Comp(EpicsMotor,"-Ax:R3}Mtr", kind='hinted')

#LT = LT_mtr('XF:21IDD-ES{LT:1-Manip:EA5_1', name='LT')
LT = LT_mtr('XF:21IDD-ES{PRV', name='LT')

#
# ---------------------------- SAMPLE_PREP Chamber -----------------------------
# Sample_prep chamber:
#
class SP_mtr(Device):
    X = Comp(EpicsMotor,"-Ax:X}Mtr", kind='hinted')
    Y = Comp(EpicsMotor,"-Ax:Y}Mtr", kind='hinted')
    Z = Comp(EpicsMotor,"-Ax:Z}Mtr", kind='hinted')
    Rx = Comp(EpicsMotor,"-Ax:Rx}Mtr", kind='hinted')

SP = SP_mtr('XF:21IDD-ES{SP:1-Manip:EA2_1', name='SP')

#
#
# ---------------------------- ANALYSIS Chamber -----------------------------
# Analysis chamber:
#

class An_mtr(Device):

    #Add micro-scanning stage motors here when done
    Claw_Trans = Comp(EpicsMotor,"{ST:1-Claw:EA1_1-Ax:T}Mtr", kind='hinted')
    Claw_Grab = Comp(EpicsMotor,"{ST:1-Claw:EA1_1-Ax:C}Mtr", kind='hinted')

An = An_mtr("XF:21IDD-ES", name='An')

#
#
# ---------------------------- Scienta Analyzer -----------------------------
# Scienta Analyzer:
#

class Analyzer_mtr(Device):
    Rz = Comp(EpicsMotor,"{SA:1-DPRF:EA1_1-Ax:1}Mtr", kind='hinted')

SA = Analyzer_mtr("XF:21IDD-ES", name='SA')
#
#
#---------------------------Mirrors------------------------------
class M1_mirror(Device):
    X = Comp(EpicsMotor,"Trans}Mtr", kind='hinted')
#    Mirror_Trans = Comp(EpicsMotor,"Trans}Mtr")
    Mirror_InOut = Comp(EpicsMotor,"Trans}Mtr")
    Ry = Comp(EpicsMotor,"Pitch}Mtr", kind='hinted')
    Mirror_Pitch = Comp(EpicsMotor,"Pitch}Mtr")
    Rz = Comp(EpicsMotor,"Roll}Mtr", kind='hinted')
    Mirror_Roll = Comp(EpicsMotor,"Roll}Mtr")

M1 = M1_mirror("XF:21IDA-OP{Mir:1-Ax:4_",name="M1")

class Hexapod_Mir(Device):
    X = Comp(EpicsMotor,"_X}Mtr", kind='hinted')
#    Mirror_Trans = Comp(EpicsMotor,"_X}Mtr")
    Mirror_InOut = Comp(EpicsMotor,"_X}Mtr")
    Y = Comp(EpicsMotor,"_Y}Mtr", kind='hinted')
    Mirror_Stripe = Comp(EpicsMotor,"_Y}Mtr")
    Mirror_Exchange = Comp(EpicsMotor,"_Y}Mtr")
    Z = Comp(EpicsMotor,"_Z}Mtr", kind='hinted')
    Rx = Comp(EpicsMotor,"_Rx}Mtr", kind='hinted')
    Mirror_Roll = Comp(EpicsMotor,"_Rz}Mtr")
    Ry = Comp(EpicsMotor,"_Ry}Mtr", kind='hinted')
    Mirror_Pitch = Comp(EpicsMotor,"_Ry}Mtr")
    Rz = Comp(EpicsMotor,"_Rz}Mtr", kind='hinted')
    Mirror_Yaw = Comp(EpicsMotor,"_Rx}Mtr")

M3 = Hexapod_Mir("XF:21IDB-OP{Mir:3-Ax:11",name="M3")
M4B = Hexapod_Mir("XF:21IDC-OP{Mir:4B-Ax:B4",name="M4B")

class Monochromator(Device):
    Focus_Const = Comp(EpicsMotor,"-Ax:8_Cff}Mtr", kind='hinted')
    Energy = Comp(EpicsMotor,"-Ax:8_Eng}Mtr", kind='hinted')
    Grating_Trans = Comp(EpicsMotor,"-Ax:8_GT}Mtr")
    Mirror_Pitch = Comp(EpicsMotor,"-Ax:8_MP}Mtr")
    Mirror_Pitch_off = Comp(EpicsSignal,"-Ax:8_MP}Mtr.OFF")
    Mirror_Pitch_set = Comp(EpicsSignal,"-Ax:8_MP}Mtr.SET")
    Mirror_Pitch_kill = Comp(EpicsSignal, "-Ax:8_MP}Cmd:Kill-Cmd")
    Grating_Pitch = Comp(EpicsMotor,"-Ax:8_GP}Mtr")
    Grating_Pitch_off = Comp(EpicsSignal,"-Ax:8_GP}Mtr.OFF")
    Grating_Pitch_set = Comp(EpicsSignal,"-Ax:8_GP}Mtr.SET")
    Grating_Pitch_kill = Comp(EpicsSignal, "-Ax:8_GP}Cmd:Kill-Cmd")
    Grating_lines = Comp(EpicsSignal,"}:LINES:RBV",write_pv="}:LINES:SET",put_complete=False, kind='hinted')

PGM = Monochromator("XF:21IDB-OP{Mono:1",name="PGM")


class EpicsSignalLastElement(EpicsSignal):
    def get(self):
        return float(super().get()[-1])


class BEST_Xaxis(Device):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.readback.name = self.name

    readback = Comp(EpicsSignalLastElement,"FM}:BPM0:PosX", kind='hinted')  # Make these EpicsSignal, 'PV:...'
    setpoint = Comp(EpicsSignal,"FM}:PID:SetpointX")
    tolerance = 0.1

        
    # Define the class properties here
    @property
    def read_val(self):
        return self.get()

    @property
    def hints(self):
        return{'fields':[self.readback.name]}

    
    def set(self, value):
        desired_value = value
        status = DeviceStatus(self)

        def are_we_there_yet(value, *args, **kwargs):
            if abs(value[-1] - desired_value) < self.tolerance:
                # This alerts the RunEngine that it can move on.
                status._finished()

        # Start us moving.
        self.setpoint.put(desired_value)
        # The function are_we_there_yet will receive
        # updates from pyepics as the readback changes.
        self.readback.subscribe(are_we_there_yet)
        # Hand this back the RunEngine immediately.
        return status

    def describe(self):
        ret = super().describe()
        for entry in ret.values():
            entry["dtype"] = "number"
            entry["shape"] = []
        return ret


class BEST_Yaxis(Device):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.readback.name = self.name

    readback = Comp(EpicsSignalLastElement,"FM}:BPM0:PosY", kind='hinted')  # Make these EpicsSignal, 'PV:...'
    setpoint = Comp(EpicsSignal,"FM}:PID:SetpointY")
    tolerance = 0.1

    def set(self, value):
        desired_value = value
        status = DeviceStatus(self)

        def are_we_there_yet(value, *args, **kwargs):
            if abs(value[-1] - desired_value) < self.tolerance:
                # This alerts the RunEngine that it can move on.
                status._finished()

        # Start us moving.
        self.setpoint.put(desired_value)
        # The function are_we_there_yet will receive
        # updates from pyepics as the readback changes.
        self.readback.subscribe(are_we_there_yet)
        # Hand this back the RunEngine immediately.
        return status
    
    def describe(self):
        ret = super().describe()
        for entry in ret.values():
            entry["dtype"] = "number"
            entry["shape"] = []
        return ret


class KB_pair(Device):
    VFM_Y = Comp(EpicsMotor,"Ax:A4_VFMTy}Mtr", kind='hinted')
    VFM_Mirror_InOut = Comp(EpicsMotor,"Ax:A4_VFMTy}Mtr")
    VFM_Mirror_Trans = Comp(EpicsMotor,"Ax:A4_VFMTy}Mtr")
    VFM_Z = Comp(EpicsMotor,"Ax:A4_VFMTz}Mtr", kind='hinted')
    VFM_Mirror_Astig = Comp(EpicsMotor,"Ax:A4_VFMTz}Mtr")
    VFM_Mirror_Horizontal = Comp(EpicsMotor,"Ax:A4_VFM_Astig}Mtr")
    VFM_Mirror_Incline = Comp(EpicsMotor,"Ax:A4_VFM_IO}Mtr")

    HFM_Z = Comp(EpicsMotor,"Ax:A4_HFM_Astig}Mtr", kind='hinted')
    HFM_X = Comp(EpicsMotor,"Ax:A4_HFM_IO}Mtr", kind='hinted')
    HFM_Mirror_Astig = Comp(EpicsMotor,"Ax:A4_HFM_Astig}Mtr")
    HFM_Mirror_InOut = Comp(EpicsMotor,"Ax:A4_HFM_IO}Mtr")
    HFM_Mirror_Trans = Comp(EpicsMotor,"Ax:A4_HFM_IO}Mtr")

    HFM_Au_Mesh = Comp(EpicsMotor,"Ax:A4_HGM}Mtr")
    VFM_Au_Mesh = Comp(EpicsMotor,"Ax:A4_VGM}Mtr")

    VFM_Rx = Comp(BEST_Xaxis,"", kind='hinted')
    VFM_Pitch = Comp(BEST_Xaxis,"")
    HFM_Ry = Comp(BEST_Yaxis,"", kind='hinted')
    HFM_Pitch = Comp(BEST_Yaxis,"")


M4A = KB_pair("XF:21IDC-OP{Mir:4A-",name="M4A")

#
# ---------------------------- SLITS AND DIAGNOSTICS -----------------------------
# Slits

#
class ESM_Diagon(Device):
    H_mirror = Comp(EpicsMotor,"-Ax:3_HLPM}Mtr", kind='hinted')
    H_Yag = Comp(EpicsMotor,"-Ax:3_HLPF}Mtr", kind='hinted')
    V_mirror = Comp(EpicsMotor,"-Ax:3_VLPM}Mtr", kind='hinted')
    V_Yag = Comp(EpicsMotor,"-Ax:3_VLPF}Mtr", kind='hinted')
#

Diagon = ESM_Diagon("XF:21IDA-OP{Diag:1",name='Diagon')

class ESMSlit_type1(Device):
    inboard = Comp(EpicsMotor,"-Ax:I}Mtr")
    outboard = Comp(EpicsMotor,"-Ax:O}Mtr")
    bottom = Comp(EpicsMotor,"-Ax:B}Mtr")
    top = Comp(EpicsMotor,"-Ax:T}Mtr")
    h_gap = Comp(EpicsMotor,"-Ax:HG}Mtr", kind='hinted')
    h_center = Comp(EpicsMotor,"-Ax:HC}Mtr", kind='hinted')
    v_gap = Comp(EpicsMotor,"-Ax:VG}Mtr", kind='hinted')
    v_center = Comp(EpicsMotor,"-Ax:VC}Mtr", kind='hinted')
#
M1Dslit = ESMSlit_type1("XF:21IDA-OP{Mir:1-Slt:4_D_1", name='M1Dslit')
PGMUslit = ESMSlit_type1("XF:21IDB-OP{Mono:1-Slt:7_U_1", name='PGMUslit')
PGMDslit = ESMSlit_type1("XF:21IDB-OP{Mono:1-Slt:8_D_1", name='PGMDslit')
M4BUslit = ESMSlit_type1("XF:21IDC-OP{Mir:4B-Slt:B4_U_1", name='M4BUslit')

class ESMSlit_type2(Device):
    h_scan = Comp(EpicsMotor,"-Ax:HS}Mtr")
    h_apperture = Comp(EpicsMotor,"-Ax:HA}Mtr")
    v_scan = Comp(EpicsMotor,"-Ax:VS}Mtr")
    v_apperture = Comp(EpicsMotor,"-Ax:VA}Mtr")
    h_gap = Comp(EpicsMotor,"-Ax:HG}Mtr", kind='hinted')
    h_center = Comp(EpicsMotor,"-Ax:HC}Mtr", kind='hinted')
    v_gap = Comp(EpicsMotor,"-Ax:VG}Mtr", kind='hinted')
    v_center = Comp(EpicsMotor,"-Ax:VC}Mtr", kind='hinted')

M3Uslit = ESMSlit_type2("XF:21IDB-OP{Mir:3-Slt:10_U_1", name='M3Uslit')
M4AUslit = ESMSlit_type2("XF:21IDC-OP{Mir:4A-Slt:A4_U_1", name='M4AUslit')
M4BDslit = ESMSlit_type2("XF:21IDC-OP{Mir:4B-Slt:B5_D_1", name='M4BDslit')

class ExitSlit(Device):
    v_gap = Comp(EpicsMotor,"_VG}Mtr")
    h_gap = Comp(EpicsMotor,"_HG}Mtr", kind='hinted')
    h_def = Comp(EpicsMotor,"_HDS}Mtr")
    v_def = Comp(EpicsMotor,"_VDS}Mtr", kind='hinted')

ExitSlitA = ExitSlit('XF:21IDC-OP{Slt:1A-Ax:A1',name='ExitSlitA')
ExitSlitB = ExitSlit('XF:21IDC-OP{Slt:1B-Ax:B1',name='ExitSlitB')

#Diagnostics
class DIAG(Device):
    trans = Comp(EpicsMotor,"}Mtr", kind='hinted')


BTA2diag = DIAG("XF:21IDC-OP{BT:A2-Diag:A2_1-Ax:1", name= 'BTA2diag')
BTB2diag = DIAG("XF:21IDC-OP{BT:B2-Diag:B2_1-Ax:1", name= 'BTB2diag')
M3Udiag = DIAG("XF:21IDB-OP{Mir:3-Diag:10_U_1-Ax:1", name= 'M3Udiag')
M4AUdiag = DIAG("XF:21IDC-OP{Mir:4A-Diag:A3_U_1-Ax:1",name = 'M4AUdiag')
M4BDdiag1 = DIAG("XF:21IDC-OP{Mir:4B-Diag:B5_D_1-Ax:1",name = 'M4BDdiag1')
M4BDdiag2 = DIAG("XF:21IDC-OP{Mir:4B-Diag:B5_D_2-Ax:2",name = 'M4BDdiag2')

# --------------------------FRONT END COMPONENTS -----------------------------
# Slits

class Virtual_Size(PVPositioner):
#This is used to connect to the front end gap size PV's.
    readback = Comp(EpicsSignalRO, 't2.C')
    setpoint = Comp(EpicsSignal,'size')
    done = Comp(EpicsSignalRO,'DMOV')
    done_value = 1

class Virtual_Center(PVPositioner):
#This is used to connect to the front end gap centre PV's.
    readback = Comp(EpicsSignalRO, 't2.D')
    setpoint = Comp(EpicsSignal,'center')
    done = Comp(EpicsSignalRO,'DMOV')
    done_value = 1

class Virtual_Motor_Center_And_Gap(Device):
#This is used to combine the gap centre and size classes from above into a single class object
    h_gap = Comp(Virtual_Size,'12-Ax:X}', kind='hinted')
    h_center = Comp(Virtual_Center,'12-Ax:X}', kind='hinted')
    v_gap = Comp(Virtual_Size,'12-Ax:Y}', kind='hinted')
    v_center = Comp(Virtual_Center,'12-Ax:Y}', kind='hinted')

class Blades(Device):
#This creates a class for the independent motion of each of the front end slit "blades"
    outboard = Comp(EpicsMotor, "1-Ax:O}Mtr")
    inboard = Comp(EpicsMotor, "2-Ax:I}Mtr")
    top = Comp(EpicsMotor, "1-Ax:T}Mtr")
    bottom = Comp(EpicsMotor, "2-Ax:B}Mtr")

class Virtual_Motor_Slits(Blades, Virtual_Motor_Center_And_Gap):
#This combines the Blades and VirtualMotorCenterAndGap classes from above to create a single class for the front end slits.
    pass

FEslit = Virtual_Motor_Slits("FE:C21A-OP{Slt:",name='FEslit')

#Define the shutters from the above class.
shutter_FE = TwoButtonShutter('XF:21ID-PPS{Sh:FE}', name='shutter_FE')
shutter_FOE = TwoButtonShutter('XF:21IDA-PPS{PSh}', name='shutter_FOE')
shutter_A = TwoButtonShutter('XF:21IDC-PPS{PSh:1A}', name='shutter_A')
shutter_B = TwoButtonShutter('XF:21IDC-PPS{PSh:1B}', name='shutter_B')

#Define the beamline gatevalves from the above class.
#Section A (Inside FOE)
BC1_GV2 = TwoButtonShutter('XF:21IDA-VA{BC:1-GV:2_D_1}', name='BC1_GV2')
Diag1_GV3 = TwoButtonShutter('XF:21IDA-VA{Diag:1-GV:3_D_1}', name='Diag1_GV3')
Mir1_GV4 = TwoButtonShutter('XF:21IDA-VA{Mir:1-GV:4_D_1}', name='Mir1_GV4')

#Section B (Between FOE and Mirror 3)
BT6_GV6 = TwoButtonShutter('XF:21IDB-VA{BT:6-GV:6_D_1}', name='BT6_GV6')
BT7_GV7 = TwoButtonShutter('XF:21IDB-VA{BT:7-GV:7_D_1}', name='BT7_GV7')
Mono1_GV8 = TwoButtonShutter('XF:21IDB-VA{Mono:1-GV:8_D_1}', name='Mono1_GV8')
BT9_GV9 = TwoButtonShutter('XF:21IDB-VA{BT:9-GV:9_D_1}', name='BT9_GV9')
BT10_GV10 = TwoButtonShutter('XF:21IDB-VA{BT:10-GV:10_D_1}', name='BT10_GV10')

#Section C, A Branch (A Branch after Mirror 3)
Mir3_GV11A = TwoButtonShutter('XF:21IDC-VA{Mir:3-GV:11_D_A}', name='Mir:3_GV11A')
Slt1A_GVA1 = TwoButtonShutter('XF:21IDC-VA{Slt:1A-GV:A1_D_1}', name='Slt1A_GVA1')
BTA2_GVA2 = TwoButtonShutter('XF:21IDC-VA{BT:A2-GV:A2_D_1}', name='BTA2_GVA2')
BTA3_GVA3 = TwoButtonShutter('XF:21IDC-VA{BT:A3-GV:A3_D_1}', name='BTA3_GVA3')

#Section C, B Branch (B Branch after Mirror 3)
Mir3_GV11B = TwoButtonShutter('XF:21IDC-VA{Mir:3-GV:11_D_B}', name='Mir:3_GV11B')
Slt1B_GVB1 = TwoButtonShutter('XF:21IDC-VA{Slt:1B-GV:B1_D_1}', name='Slt1B_GVB1')
BTB2_GVB2 = TwoButtonShutter('XF:21IDC-VA{BT:B2-GV:B2_D_1}', name='BTB2_GVB2')
BTB3_GVB3 = TwoButtonShutter('XF:21IDC-VA{BT:B3-GV:B3_D_1}', name='BTB3_GVB3')
Mir4B_GVB4 = TwoButtonShutter('XF:21IDC-VA{Mir:4B-GV:B4_D_1}', name='Mir4B_GVB4')
BTB5_GVB5 = TwoButtonShutter('XF:21IDC-VA{BT:B5-GV:B5_D_1}', name='BTB5_GVB5')

#Section D, (A Branch endstation)
Anal1A_GVEA1 = TwoButtonShutter('XF:21IDD-VA{ANAL:1A-GV:EA1_1}', name='Anal1A_GVEA1')

# Temperature monitoring
D1 = EpicsSignalRO("XF:21IDD-ES{PS:Heat3}D1_RB", name="D1")
D2 = EpicsSignalRO("XF:21IDD-ES{PS:Heat3}D2_RB", name="D2")
Stinger = EpicsSignalRO("XF:21ID1-ES{TCtrl:2-Chan:A}T-I", name="Stinger")

