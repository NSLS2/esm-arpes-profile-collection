from ophyd import EpicsSignal, EpicsSignalRO
from bluesky.suspenders import SuspendFloor,SuspendBoolHigh

from bluesky.magics import BlueskyMagics
get_ipython().register_magics(BlueskyMagics)


#gs.DETS = []
#all_objs = globals()
#counters = [counter for counter in all_objs.values() if isinstance(counter, EpicsSignal)]

#gs.DETS = counters

"""
This section is used to define a few settings values that are used for all scans.
"""

#This defines which motor devices to measure at the start and end of each scan. The axes that
#are measured are defined at the initialization of the motor device in the '.hints.' attribute.

#sd.baseline = [BeamSource, EPU105, FEslit, M1, PGM, M3, ExitSlitA, ExitSlitB, M4A, M4B, LT, SP]
#sd.baseline = [BeamSource,EPU57, EPU105, FEslit, M1, PGM, M3, ExitSlitA, ExitSlitB, M4A, M4B, LT, SP]
sd.baseline = [BeamSource,EPU57, EPU105, FEslit, M1, PGM, M3, ExitSlitA, ExitSlitB, M4A, M4B, LT, D1, D2, Stinger]

#This line command defines the list of motor axes that are to be displayed when using the magics
# command %wa.
BlueskyMagics.positioners = [EPU57.gap,EPU57.phase,EPU105.gap,EPU105.phase
                             ,FEslit.h_center,FEslit.h_gap,FEslit.v_center,FEslit.v_gap,
                             Diagon.H_mirror, Diagon.H_Yag, Diagon.V_mirror, Diagon.V_Yag,
                              M1.X,M1.Ry,M1.Rz,
                             M1Dslit.h_gap, M1Dslit.h_center,M1Dslit.v_gap,M1Dslit.v_center,
                             M1Dslit.inboard,M1Dslit.outboard,M1Dslit.bottom,M1Dslit.top,
                             PGMUslit.h_gap, PGMUslit.h_center,PGMUslit.v_gap,PGMUslit.v_center,
                             PGMUslit.inboard,PGMUslit.outboard,PGMUslit.bottom,PGMUslit.top,
                             PGM.Energy,PGM.Focus_Const,PGM.Grating_Trans,
                             PGMDslit.h_gap, PGMDslit.h_center,PGMDslit.v_gap,PGMDslit.v_center,
                             PGMDslit.inboard,PGMDslit.outboard,PGMDslit.bottom,PGMDslit.top,
                             M3Udiag.trans,
                             M3Uslit.h_gap, M3Uslit.h_center,M3Uslit.v_gap,M3Uslit.v_center,
                             M3Uslit.h_scan,M3Uslit.h_apperture,M3Uslit.v_scan,M3Uslit.v_apperture,
                             M3.X,M3.Y,M3.Z,M3.Rx,M3.Ry,M3.Rz,
                             ExitSlitA.v_gap, ExitSlitA.h_gap,ExitSlitA.v_def,ExitSlitA.h_def,
                             ExitSlitB.v_gap, ExitSlitB.h_gap,ExitSlitB.v_def,ExitSlitB.h_def,
                             BTA2diag.trans,BTB2diag.trans,M4AUdiag.trans,
                             M4AUslit.h_gap, M4AUslit.h_center,M4AUslit.v_gap,M4AUslit.v_center,
                             M4AUslit.h_scan,M4AUslit.h_apperture,M4AUslit.v_scan,M4AUslit.v_apperture,
                             M4A.VFM_Y,M4A.VFM_Z,M4A.VFM_Rx,M4A.HFM_X,M4A.HFM_Z,M4A.HFM_Ry,
                             M4A.HFM_Au_Mesh,M4A.VFM_Au_Mesh,
                             M4B.X,M4B.Y,M4B.Z,M4B.Rx,M4B.Ry,M4B.Rz,
                             M4BDslit.h_gap, M4BDslit.h_center,M4BDslit.v_gap,M4BDslit.v_center,
                             M4BDslit.h_scan,M4BDslit.h_apperture,M4BDslit.v_scan,M4BDslit.v_apperture,
                             M4BDdiag1.trans,M4BDdiag2.trans,
                             LT.X,LT.Y,LT.Z,LT.Ry, SP.X,SP.Y,SP.Z,SP.Rx,
                             LL.Claw_Trans,LL.Claw_Rotate,LL.Claw_Grab,LL.Dock_Trans]




#I need to add many more motors into this definition, also I should add a BlueskyMagics.dets line
#for detectors.


#The following set of commands is used to define the suspenders used for each scan.

#Suspender for beam current.
suspender_beam_current = SuspendFloor(EpicsSignalRO('SR:OPS-BI{DCCT:1}I:Real-I'), 5.0)
RE.install_suspender(suspender_beam_current)



#Suspender for photon shutters.
    #FE photon shutter
suspender_FE_photon_shutter = SuspendBoolHigh(EpicsSignalRO('XF:21ID-PPS{Sh:FE}Pos-Sts'))
RE.install_suspender(suspender_FE_photon_shutter)

    #FOE photon shutter
# DAMA (mrakitin): commenting out on 06/04/2018 due to some scans
#                  need to be run with the closed FOE shutter
# suspender_FOE_photon_shutter = SuspendBoolHigh(EpicsSignalRO('XF:21IDA-PPS{PSh}Pos-Sts'))
# RE.install_suspender(suspender_FOE_photon_shutter)



#Suspender for EPU57/EPU1.
    #Tilt_ST indicator suspenders
suspender_EPU1_Tilt_ST_LimPlus = SuspendBoolHigh(EpicsSignalRO('SR:C21-ID:G1A{EPU:1-Ax:TiltT}Lim:Plus-Sts'))
RE.install_suspender(suspender_EPU1_Tilt_ST_LimPlus)

suspender_EPU1_Tilt_ST_LimMinus = SuspendBoolHigh(EpicsSignalRO('SR:C21-ID:G1A{EPU:1-Ax:TiltT}Lim:Minus-Sts'))
RE.install_suspender(suspender_EPU1_Tilt_ST_LimMinus)

suspender_EPU1_Tilt_ST_URange = SuspendBoolHigh(EpicsSignalRO('SR:C21-ID:G1A{EPU:1-Ax:TiltT}Sts:UnderRange-Sts'))
RE.install_suspender(suspender_EPU1_Tilt_ST_URange)

suspender_EPU1_Tilt_ST_ORange = SuspendBoolHigh(EpicsSignalRO('SR:C21-ID:G1A{EPU:1-Ax:TiltT}Sts:OverRange-Sts'))
RE.install_suspender(suspender_EPU1_Tilt_ST_ORange)

    #Tilt_SB indicator suspenders
suspender_EPU1_Tilt_SB_LimPlus = SuspendBoolHigh(EpicsSignalRO('SR:C21-ID:G1A{EPU:1-Ax:TiltB}Lim:Plus-Sts'))
RE.install_suspender(suspender_EPU1_Tilt_SB_LimPlus)

suspender_EPU1_Tilt_SB_LimMinus = SuspendBoolHigh(EpicsSignalRO('SR:C21-ID:G1A{EPU:1-Ax:TiltB}Lim:Minus-Sts'))
RE.install_suspender(suspender_EPU1_Tilt_SB_LimMinus)

suspender_EPU1_Tilt_SB_URange = SuspendBoolHigh(EpicsSignalRO('SR:C21-ID:G1A{EPU:1-Ax:TiltB}Sts:UnderRange-Sts'))
RE.install_suspender(suspender_EPU1_Tilt_SB_URange)

suspender_EPU1_Tilt_SB_ORange = SuspendBoolHigh(EpicsSignalRO('SR:C21-ID:G1A{EPU:1-Ax:TiltB}Sts:OverRange-Sts'))
RE.install_suspender(suspender_EPU1_Tilt_SB_ORange)


#Suspender for EPU105/EPU2.
    #Tilt_ST indicator suspenders
suspender_EPU2_Tilt_ST_LimPlus = SuspendBoolHigh(EpicsSignalRO('SR:C21-ID:G1B{EPU:2-Ax:TiltT}Lim:Plus-Sts'))
RE.install_suspender(suspender_EPU2_Tilt_ST_LimPlus)

suspender_EPU2_Tilt_ST_LimMinus = SuspendBoolHigh(EpicsSignalRO('SR:C21-ID:G1B{EPU:2-Ax:TiltT}Lim:Minus-Sts'))
RE.install_suspender(suspender_EPU2_Tilt_ST_LimMinus)

suspender_EPU2_Tilt_ST_URange = SuspendBoolHigh(EpicsSignalRO('SR:C21-ID:G1B{EPU:2-Ax:TiltT}Sts:UnderRange-Sts'))
RE.install_suspender(suspender_EPU2_Tilt_ST_URange)

suspender_EPU2_Tilt_ST_ORange = SuspendBoolHigh(EpicsSignalRO('SR:C21-ID:G1B{EPU:2-Ax:TiltT}Sts:OverRange-Sts'))
RE.install_suspender(suspender_EPU2_Tilt_ST_ORange)

    #Tilt_SB indicator suspenders
suspender_EPU2_Tilt_SB_LimPlus = SuspendBoolHigh(EpicsSignalRO('SR:C21-ID:G1B{EPU:2-Ax:TiltB}Lim:Plus-Sts'))
RE.install_suspender(suspender_EPU2_Tilt_SB_LimPlus)

suspender_EPU2_Tilt_SB_LimMinus = SuspendBoolHigh(EpicsSignalRO('SR:C21-ID:G1B{EPU:2-Ax:TiltB}Lim:Minus-Sts'))
RE.install_suspender(suspender_EPU2_Tilt_SB_LimMinus)

suspender_EPU2_Tilt_SB_URange = SuspendBoolHigh(EpicsSignalRO('SR:C21-ID:G1B{EPU:2-Ax:TiltB}Sts:UnderRange-Sts'))
RE.install_suspender(suspender_EPU2_Tilt_SB_URange)

suspender_EPU2_Tilt_SB_ORange = SuspendBoolHigh(EpicsSignalRO('SR:C21-ID:G1B{EPU:2-Ax:TiltB}Sts:OverRange-Sts'))
RE.install_suspender(suspender_EPU2_Tilt_SB_ORange)
