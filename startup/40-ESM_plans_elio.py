import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.optimize as opt
import os
from bluesky.plans import scan, adaptive_scan, spiral_fermat, spiral,scan_nd
from bluesky.plan_stubs import abs_set, mv, rd
from bluesky.preprocessors import baseline_decorator, subs_decorator
# from bluesky.callbacks import LiveTable,LivePlot, CallbackBase
#from pyOlog.SimpleOlogClient import SimpleOlogClient
#from ophyd import EpicSignal
from esm import ss_csv
from cycler import cycler
from collections import ChainMap
import math
import re
from boltons.iterutils import chunked

def ESM_check(return_all=False):

    
                       ############# Define the Detector 
    
    detector = qem07      # The detector to use for the scan.
    det_range = '50 pC'     # The range to use for the scan
    det_vals_reading = 5  # The values per reading to use.
    det_avg_time = 0.1    # The averaging time to use.
    det_int_time = 0.0004 # The integration time to use.

    
                       ############# Define the Beamline Motors 
    Und = EPU1.gap

    FE_h_center = FE_slit.h_center             # The front end horizontal center motor
    FE_v_center = FE_slit.v_center             # The front end vertical gap motor

    FE_hgap_axis = FE_slit.h_gap     # The front end horizontal gap motor
    FE_vgap_axis = FE_slit.v_gap     # The front end vertical gap motor

    MM1_InOut_motor = M1.Mirror_InOut 
    MM1_Pitch_motor = M1.Mirror_Pitch 
    MM1_Roll_motor = M1.Mirror_Roll 

    MM3_X_motor = M3.X 
    MM3_Y_motor = M3.Y 
    MM3_Z_motor = M3.Z 
    MM3_Yaw_motor = M3.Mirror_Yaw 
    MM3_Pitch_motor = M3.Mirror_Pitch 
    MM3_Roll_motor = M3.Mirror_Roll 
    
    PGM_Energy_motor = PGM.Energy    #The motor required to move the PGM energy.

    Exit_Slit_hgap_motor = Exit_SlitA.h_gap    # THe motor required to move the horizontal exit slit
    Exit_Slit_vgap_motor = Exit_SlitA.v_gap    # THe motor required to move the vertical exit slit

    Diode_motor = BTA2_diag.trans    #The motor to move the diode into position.


            ############ Read in the initial settings (motors and detectors)######
                        #to be able to restore the initial configuration 

    initial_Und_gap =Und.position
    initial_FE_h_center_pos  =FE_h_center.position
    initial_FE_v_center_pos  =FE_v_center.position
    initial_FE_hgap_pos = FE_hgap_axis.position
    initial_FE_vgap_pos = FE_vgap_axis.position

    initial_MM1_InOut_pos = M1.Mirror_InOut.position
    initial_MM1_Pitch_pos = M1.Mirror_Pitch.position
    initial_MM1_Roll_pos =  M1.Mirror_Roll.position 

    initial_MM3_X_pos = M3.X.position 
    initial_MM3_Y_pos = M3.Y.position 
    initial_MM3_Z_pos = M3.Z.position 
    initial_MM3_Yaw_pos = M3.Mirror_Yaw.position 
    initial_MM3_Pitch_pos = M3.Mirror_Pitch.position 
    initial_MM3_Roll_pos = M3.Mirror_Roll.position 
    
    initial_Exit_Slit_hgap_pos = Exit_Slit_hgap_motor.position # The initial horizontal gap opening
    initial_Exit_Slit_vgap_pos = Exit_Slit_vgap_motor.position # The initial vertical gap opening

    initial_PGM_Energy_pos = PGM_Energy_motor.position         #The initial photon energy.
    initial_Diode_pos = Diode_motor.position                   # The initial location of the diode motor.

                           # Read the initial values for the 'Gas_cell' detector settings,
 
    initial_det_range = yield from rd(detector.driver.range)                    # The initial detector range
    initial_det_vals_reading = yield from rd(detector.driver.values_per_read)   # The initial values per reading.
    initial_det_avg_time = yield from rd(detector.driver.averaging_time)        # The initial averaging time.
    initial_det_int_time = yield from rd(detector.driver.integration_time)      # The initial integration time.


            ############  define the REFERENCE beamline configuration #########   

    Und_gap = 35.5
    
    FE_h_center_pos  = 0.750                   # The front end horizontal gap value
    FE_v_center_pos  = 0.325                   # The front end vertical gap value

    FE_hgap_pos = 0.75                    # The front end horizontal gap motor
    FE_vgap_pos = 0.75                    # The front end vertical gap motor
    
    MM1_InOut_pos = -3.906
    MM1_Pitch_pos = -3605.1
    MM1_Roll_pos = 5397.6

    MM3_X_pos = 0.25 
    MM3_Y_pos = -14.0 
    MM3_Z_pos = 0.0
    MM3_Yaw_pos = 0.0 
    MM3_Pitch_pos = -0.726190 
    MM3_Roll_pos = -0.25
    
    PGM_Energy_pos = 695             #THe energy at which to perform the scan.

    Exit_Slit_hgap_pos = 20                    # The horizontal gap opening to use
    Exit_Slit_vgap_pos = 20                    # The vertical gap opening to use

    Diode_pos = -63                  #The position of the diode motor to be used during the scan

              ############  before measuring the new flux, read-in the latest reference flux  #########       
    
    h = next(iter(db(scan_type='Reference_Flux_Check')))
    last_flux = h.table()['qem07-current-1-mean_value']


              ############  set the beamline to REFERENCE configuration #########       
    
    yield from mv( Und,Und_gap,
                   FE_h_center, FE_h_center_pos,
                   FE_v_center, FE_v_center_pos, 
                   FE_hgap_axis,FE_hgap_pos,
                   FE_vgap_axis,FE_vgap_pos)

    yield from mv( MM1_InOut_motor, MM1_InOut_pos,
                   MM1_Pitch_motor, MM1_Pitch_pos,
                   MM1_Roll_motor, MM1_Roll_pos)

    yield from mv( MM3_X_motor, MM3_X_pos,
                   MM3_Y_motor, MM3_Y_pos,
                   MM3_Z_motor, MM3_Z_pos,
                   MM3_Yaw_motor, MM3_Yaw_pos,
                   MM3_Pitch_motor, MM3_Pitch_pos,
                   MM3_Roll_motor, MM3_Roll_pos)
        
    yield from mv(PGM_Energy_motor,PGM_Energy_pos,
                  Exit_Slit_hgap_motor,Exit_Slit_hgap_pos,
                  Exit_Slit_vgap_motor,Exit_Slit_vgap_pos)
    
    yield from mv(Diode_motor,Diode_pos)

    yield from mv(detector.driver.range, det_range,                     # The range to use for the scan
                  detector.driver.values_per_read, det_vals_reading,    # The values per reading to use.
                  detector.driver.averaging_time, det_avg_time,         # The averaging time to use.
                  detector.driver.integration_time, det_int_time)       # The integration time to use.
                

              ############  measure the flux and compare with previous #########       
    
        # NB: the string 'Reference_Flux_Check' is crutial: to find the previous flux measuraments 

    uid=yield from (scan_time([detector],num=1,scan_type='Reference_Flux_Check'))  
    
        # Read in the flux just measured in the same conditions as last time to be able to compare
    h = next(iter(db(scan_type='Reference_Flux_Check')))
    new_flux = h.table()['qem07-current-1-mean_value']

    print('old_flux = %e, new_flux = %e, prc_diff = %f' %(last_flux, new_flux, (new_flux-last_flux)/new_flux))
    

    
             ############# Reset original position if 'return_all = True' #############

    if return_all is True:

        yield from mv( Und,initial_Und_gap,
                       FE_h_center,    initial_FE_h_center_pos,
                       FE_v_center,    initial_FE_v_center_pos, 
                       FE_hgap_axis,   initial_FE_hgap_pos,
                       FE_vgap_axis,   initial_FE_vgap_pos)

        yield from mv( MM1_InOut_motor, initial_MM1_InOut_pos,
                       MM1_Pitch_motor, initial_MM1_Pitch_pos,
                       MM1_Roll_motor,  initial_MM1_Roll_pos)
        
        yield from mv( PGM_Energy_motor,      initial_PGM_Energy_pos,
                       Exit_Slit_hgap_motor,  initial_Exit_Slit_hgap_pos,
                       Exit_Slit_vgap_motor,  initial_Exit_Slit_vgap_pos)

        yield from mv( MM3_X_motor,     initial_MM3_X_pos,
                       MM3_Y_motor,     initial_MM3_Y_pos,
                       MM3_Z_motor,     initial_MM3_Z_pos,
                       MM3_Yaw_motor,   initial_MM3_Yaw_pos,
                       MM3_Pitch_motor, initial_MM3_Pitch_pos,
                       MM3_Roll_motor,  initial_MM3_Roll_pos)
        
        yield from mv(Diode_motor,initial_Diode_pos)


        yield from mv(detector.driver.range, initial_det_range,                     # The range to use for the scan
                      detector.driver.values_per_read, initial_det_vals_reading,    # The values per reading to use.
                      detector.driver.averaging_time, initial_det_avg_time,         # The averaging time to use.
                      detector.driver.integration_time, initial_det_int_time)       # The integration time to use.
        

    return


def ESM_status(f_nm = None):
    ''' routine to read the current positions of the motors in the beamline
        and save a status-file ("/direct/XF21ID1/csv_files/status.txt")
        usage: ESM_status(f_nm = None)
        if no file name is provided, the current status is appended to the status-file above.
     '''

    import time

                       ############# Define the Beamline Motors 
    Und1 = EPU1.gap
    Und2 = EPU2.gap
    
    FE_h_center = F_slit.h_center             # The front end horizontal center motor
    FE_v_center = F_slit.v_center             # The front end vertical gap motor

    FE_hgap_axis = FEslit.h_gap     # The front end horizontal gap motor
    FE_vgap_axis = FEslit.v_gap     # The front end vertical gap motor

    MM1_InOut_motor = M1.Mirror_InOut
    MM1_Pitch_motor = M1.Mirror_Pitch 
    MM1_Roll_motor = M1.Mirror_Roll 

    
    MM3_X_motor = M3.X 
    MM3_Y_motor = M3.Y 
    MM3_Z_motor = M3.Z 
    MM3_Yaw_motor = M3.Mirror_Yaw 
    MM3_Pitch_motor = M3.Mirror_Pitch 
    MM3_Roll_motor = M3.Mirror_Roll 
    
    PGM_Energy_motor = PGM.Energy    #The motor required to move the PGM energy.
    PGM_cff_motor = PGM.Focus_Const
    PGM_Grating_motor = PGM.Grating_Trans

    
    Exit_SlitA_hgap_motor = ExitSlitA.h_gap    # THe motor required to move the horizontal exit slit
    Exit_SlitA_vgap_motor = ExitSlitA.v_gap    # THe motor required to move the vertical exit slit

    Exit_SlitB_hgap_motor = ExitSlitB.h_gap    # THe motor required to move the horizontal exit slit
    Exit_SlitB_vgap_motor = ExitSlitB.v_gap    # THe motor required to move the vertical exit slit
    
 #   Diode_motor = BTA2_diag.trans    #The motor to move the diode into position.


            ############ READ MOTOR POSITIONS ######


    Und1_gap =Und1.position
    Und2_gap =Und2.position
    FE_h_center_pos  =FE_h_center.position
    FE_v_center_pos  =FE_v_center.position
    FE_hgap_pos = FE_hgap_axis.position
    FE_vgap_pos = FE_vgap_axis.position

    MM1_InOut_pos = M1.Mirror_InOut.position
    MM1_Pitch_pos = M1.Mirror_Pitch.position
    MM1_Roll_pos =  M1.Mirror_Roll.position 

    PGM_Energy_pos = PGM_Energy_motor.position         #The initial photon energy.
    PGM_cff_pos = PGM_cff_motor.position
    PGM_Grating_pos = PGM_Grating_motor.position

    MM3_X_pos = M3.X.position 
    MM3_Y_pos = M3.Y.position 
    MM3_Z_pos = M3.Z.position 
    MM3_Yaw_pos = M3.Mirror_Yaw.position 
    MM3_Pitch_pos = M3.Mirror_Pitch.position 
    MM3_Roll_pos = M3.Mirror_Roll.position 
    
    Exit_SlitA_hgap_pos = Exit_SlitA_hgap_motor.position # The initial horizontal gap opening
    Exit_SlitA_vgap_pos = Exit_SlitA_vgap_motor.position # The initial vertical gap opening

    Exit_SlitB_hgap_pos = Exit_SlitB_hgap_motor.position # The initial horizontal gap opening
    Exit_SlitB_vgap_pos = Exit_SlitB_vgap_motor.position # The initial vertical gap opening

            ############ PRINT THE BEAMLINE STATUS FILE ######
    
    if f_nm == None: 
        fl="/direct/XF21ID1/csv_files/STATUS.txt"
    else:
        fl='/direct/XF21ID1/csv_files/'+ f_nm +'_STATUS.txt'
        
    f = open(fl, "a")

    f.write('************************************************************\n')
    f.write('BEAMLINE STATUS: '+ time.strftime("%c") + '\n')
    f.write('************************************************************\n\n')

    f.write('     EPU: \n')
    f.write('\t EPU1 gap -->  %f\n' %Und1_gap)
    f.write('\t EPU2 gap -->  %f\n\n' %Und2_gap)
    
    f.write('     FRONT END SLITS: \n')
    f.write('\t FE_h_ctr -->  %f\n' %FE_h_center_pos)
    f.write('\t FE_v_ctr -->  %f\n' %FE_v_center_pos)
    f.write('\t FE_h -->  %f\n' %FE_hgap_pos)
    f.write('\t FE_v -->  %f\n\n' %FE_vgap_pos)

    f.write('     M1 MIRROR \n')
    f.write('\t M1_inout -->  %f\n' %MM1_InOut_pos)
    f.write('\t M1_pitch -->  %f\n' %MM1_Pitch_pos)
    f.write('\t M1_roll -->  %f\n\n' %MM1_Roll_pos)

    f.write('     PGM \n')
    f.write('\t PGM_Energy -->  %f\n' %PGM_Energy_pos)
    f.write('\t PGM_Cff -->  %f\n' %PGM_cff_pos)
    f.write('\t PGM_Transl -->  %f\n\n' %PGM_Grating_pos)
    
    f.write('     M3 MIRROR \n')
    f.write('\t M3_X -->  %f\n' %MM3_X_pos)
    f.write('\t M3_Y -->  %f\n' %MM3_Y_pos)
    f.write('\t M3_Z -->  %f\n' %MM3_Z_pos)
    f.write('\t M3_Yaw -->  %f\n' %MM3_Yaw_pos)
    f.write('\t M3_Pitch -->  %f\n' %MM3_Pitch_pos)
    f.write('\t M3_Roll -->  %f\n\n' %MM3_Roll_pos)

    f.write('     EXIT SLITS \n')
    f.write('\t H_gap_A -->  %f\n' %Exit_SlitA_hgap_pos)
    f.write('\t V_gap_A -->  %f\n\n' %Exit_SlitA_vgap_pos)

    f.write('\t H_gap_B -->  %f\n' %Exit_SlitB_hgap_pos)
    f.write('\t V_gap_B -->  %f\n\n' %Exit_SlitB_vgap_pos)
    
    f.close()
    
    
    return
























def ESM_check_test(return_all=True):

    # Define some parameters that are to be used in the scan, these are designed to be 'preset' and hence are not
    #'inputted' into the scan.

    detector = qem07      # The detector to use for the scan.
    det_range = '50 pC'     # The range to use for the scan
    det_vals_reading = 5  # The values per reading to use.
    det_avg_time = 0.1    # The averaging time to use.
    det_int_time = 0.0004 # The integration time to use.

    Exit_Slit_hgap_motor = Exit_SlitA.h_gap # THe motor required to move the horizontal exit slit
    Exit_Slit_hgap_pos = 20                # The horizontal gap opening to use
    Exit_Slit_vgap_motor = Exit_SlitA.v_gap # THe motor required to move the vertical exit slit
    Exit_Slit_vgap_pos = 10                # The vertical gap opening to use

    PGM_Energy_motor = PGM.Energy    #The motor required to move the PGM energy.
    PGM_Energy_pos = 695             #THe energy at which to perform the scan.


    Diode_motor = BTA2_diag.trans    #The motor to move the diode into position.
    Diode_pos = -63                  #The position of the diode motor to be used during the scan

    scan_type_str='Reference_Flux_Check'

        # Read in initial values to be able to go back at the end
        # Read the intial values for each motor that is to be moved.
    
       # Read the initial values for the 'Gas_cell' detector settings, PGM slits, PGM energy and diode position

    initial_det_range = yield from rd(detector.driver.range)                    # The initial detector range
    initial_det_vals_reading = yield from rd(detector.driver.values_per_read)   # The initial values per reading.
    initial_det_avg_time = yield from rd(detector.driver.averaging_time)        # The initial averaging time.
    initial_det_int_time = yield from rd(detector.driver.integration_time)      # The initial integration time.

    initial_Exit_Slit_hgap_pos = Exit_Slit_hgap_motor.position # The initial horizontal gap opening
    initial_Exit_Slit_vgap_pos = Exit_Slit_vgap_motor.position # The initial vertical gap opening

    initial_PGM_Energy_pos = PGM_Energy_motor.position         #The initial photon energy.
    initial_Diode_pos = Diode_motor.position                   # The initial location of the diode motor.

    #ADD A CLOSE SHUTTER CALL HERE

       # Read in the flux measured last time in the same condition to be able to compare with the new value measured now
    h = next(iter(db(scan_type='Reference_Flux_Check')))
    last_flux = h.table()['qem07-current-1-mean_value']

    #Move the values to the starting reference positions for the scan.
    
 
    yield from mv(Exit_Slit_hgap_motor,Exit_Slit_hgap_pos,
                  Exit_Slit_vgap_motor,Exit_Slit_vgap_pos,
                  PGM_Energy_motor,PGM_Energy_pos,
                  Diode_motor,Diode_pos)

    yield from mv(detector.driver.range, det_range,                     # The range to use for the scan
                  detector.driver.values_per_read, det_vals_reading,    # The values per reading to use.
                  detector.driver.averaging_time, det_avg_time,         # The averaging time to use.
                  detector.driver.integration_time, det_int_time)       # The integration time to use.
                
    #ADD AN OPEN SHUTTER CALL HERE

    uid=yield from (scan_time([detector],num=1,DET_channel=1, scan_type=scan_type_str))  


           # Read in the flux just measured in the same conditions as last time to be able to compare
    h = next(iter(db(scan_type='Reference_Flux_Check')))
    new_flux = h.table()['qem07-current-1-mean_value']

    print('old_flux = %e, new_flux = %e, prc_diff = %f' %(last_flux, new_flux, (new_flux-last_flux)/new_flux))
    
    

#    hdr = db[uid]
#    df =get_table(hdr,['qem07_current1_mean_value'])

#    fl="/direct/XF21ID1/csv_files/flux_check_history.txt"
#    fl_latest="/direct/XF21ID1/csv_files/flux_check_latest.txt"

    # this create a file with the latest record of diode current
#    cols=df.columns.tolist()
#    df=df[cols]
#    df.to_csv(fl_latest,index=False)
    # this read that record and append it to the flux_hystory file
#    with open(fl_latest) as f:
#        last_line = f.readlines()
#    f.close()
#    f = open(fl, "a")
#    f.write(last_line[1])
#    f.close()
    
        # Reset the values to the original position if 'return_all = True'
    #ADD A CLOSE PHOTON SHUTTER LINE HERE.
    if return_all is True:

        yield from mv(Exit_Slit_hgap_motor,initial_Exit_Slit_hgap_pos,
                      Exit_Slit_vgap_motor,initial_Exit_Slit_vgap_pos,
                      PGM_Energy_motor,initial_PGM_Energy_pos,
                      Diode_motor,initial_Diode_pos)

        yield from mv(detector.driver.range, initial_det_range,                     # The range to use for the scan
                      detector.driver.values_per_read, initial_det_vals_reading,    # The values per reading to use.
                      detector.driver.averaging_time, initial_det_avg_time,         # The averaging time to use.
                      detector.driver.integration_time, initial_det_int_time)       # The integration time to use.
        
 
    #ADD AN OPEN SHUTTER CALL HERE

    return



def sh_test():
    caput('XF:21ID-PPS{Sh:FE}Cmd:Cls-Cmd', 1)
    #caput XF:21IDC-BI{EM:7}EM180:Acquire 0
    uid=yield from (scan_time([detector],num=1, scan_type=scan_type_str))  
    value = db[-1].table()['qem07-current-2-mean_value']
    caput('XF:21IDC-BI{EM:7}EM180:CurrentOffset1', value)
    caput('XF:21ID-PPS{Sh:FE}Cmd:Opn-Cmd', 1)
    return 



from ophyd import EpicsMotor, PVPositioner, PVPositionerPC, EpicsSignal, EpicsSignalRO, Device
from ophyd import Component as Cpt, FormattedComponent as FmtCpt
#from ophyd import (EpicsMCA, EpicsDXP)
#from ophyd import DeviceStatus
import bluesky.plans as bp

class LinearActOut(PVPositioner):
    readback = Cpt(EpicsSignalRO, 'Pos-Sts')
    setpoint = Cpt(EpicsSignal, 'Cmd:Cls-Cmd')
    done = Cpt(EpicsSignalRO, 'Pos-Sts')
    done_val = 'Not Open' # for some reason this is how the logic .  limit activated and logic turns to 0

class LinearActIn(PVPositioner):
    readback = Cpt(EpicsSignalRO, 'Pos-Sts')
    setpoint = Cpt(EpicsSignal, 'Cmd:Opn-Cmd')
    done = Cpt(EpicsSignalRO, 'Pos-Sts')
    done_val = 'Open'  #for some reason this logic is backwards. need to fix this.


sh_in = LinearActIn('XF:21ID-PPS{Sh:FE}', name='sh_in')
sh_out = LinearActOut('XF:21ID-PPS{Sh:FE}', name='sh_out')

def sh_close():
    #yield from bp.mv(sh_in,1)
    os.system('caput XF:21ID-PPS{Sh:FE}Cmd:Cls-Cmd 1')

def sh_open():
    #yield from bp.mv(sh_out,1)
    os.system('caput XF:21ID-PPS{Sh:FE}Cmd:Opn-Cmd 1')





def macro():

    Und = EPU1.gap
    PGM_Focus_motor = PGM.Focus_Const
    Ex_slt_v_mtr = Exit_SlitA.v_gap

    sh_open()
    
#    Und_gap = 31.53
#    yield from mv( Und,Und_gap)
#    yield from scan_multi_1D([qem12], PGM.Focus_Const, 2.9, 3.4, 0.1, PGM.Energy, 529.75, 533, 0.02, DET_channel=3, snake = True)
    
#    Und_gap = 27.75
#    yield from mv( Und,Und_gap)
#    yield from scan_multi_1D([qem12], PGM.Focus_Const, 2.9, 3.4, 0.1, PGM.Energy, 400, 402.5, 0.02, DET_channel=3, snake = True)


#    c = 3.1
#    yield from mv( PGM.Focus_Const, c)

#    Und_gap = 27.75
#    yield from mv( Und,Und_gap)
#    yield from scan_multi_1D([qem12], Exit_SlitB.v_gap, 10, 55, 5, PGM.Energy, 400.0, 402.5, 0.01, DET_channel=3)
     
#    Und_gap = 31.53
#    yield from mv( Und,Und_gap)
#    yield from scan_multi_1D([qem12], Exit_SlitB.v_gap, 10, 55, 5, PGM.Energy, 529.75, 533.0, 0.01, DET_channel=3, snake = True)

    yield from scan_multi_1D([qem07], EPU1.gap, 16, 65, 2, PGM.Energy, 130, 1500, 1.0, DET_channel=1, snake = True)

#    yield from scan_multi_1D([qem07], PGM.Energy, 130, 300, 10.0, EPU1.gap, 16, 30, .1,  DET_channel=1)
#   yield from scan_multi_1D([qem07], PGM.Energy, 250, 1000, 50.0, EPU1.gap, 20, 50, .1,  DET_channel=1, snake = True)
#    yield from scan_multi_1D([qem07], PGM.Energy, 1000, 1500, 50.0, EPU1.gap, 20, 70, .1,  DET_channel=1, snake = True)
#    yield from scan_multi_1D([qem07],  PGM.Energy, 30, 100, 5.0 , EPU1.gap, 25, 60, .1, DET_channel=1, snake = True)    
    yield from scan_multi_1D([qem07],  PGM.Energy, 160, 400, 20.0, EPU2.gap, 45, 80, .1, DET_channel=1, snake = True)

    sh_close()
    


########################################################
########### GRT #######################################
#######################################################
def esm_grt(grt, ph_en, branch = 'A', EPU = 57):
    ''' set up the position of the gratings. 
    Puts the correct offset for M2 and GRT pitches
    usage: esm_grt(grt (1200, 800, 600, 300, integer), ph_en (in eV, float), branch = ('A' or 'B')'''
    
#    PGM_Grating_motor = PGM.Grating_Trans
#    PGM_Focus_motor = PGM.Focus_Const
#    PGM_Energy_motor = PGM.Energy  
#    PGM_Mirror_motor = PGM.Mirror_Pitch
#    PGM_Grt_Pitch_motor = PGM.Grating_Pitch



    

    GRT_TRANSLATION = {'1200':  -65.36,            # grating traslation position
                       '800':  -205.429,
                       '600': -135.9223,
                       '300':  +5.4758}
                                                  # M2 and GRT pitchs offsets and c-values from gas measuraments  
    if branch == 'A':
        off_M2 = {'1200': '83.7681090000',
                    '800': '83.760909',
                    '600': '83.683109',
                    '300': '83.77470'}
        off_GRT = {'1200': '81.9522200000',
                     '800': '81.93502',
                     '600': '81.874220000',
                     '300': '81.90042'}
        c =      {'1200': 2.05,
                   '800': 3.2,
                   '600': 3.0,
                   '300':2.0 }
    elif branch =='B':
        off_M2 = {'1200': '83.773809',
                    '800': '83.7609090000',
                    '600': '83.776709',
                    '300': '83.7605'}
        off_GRT = {'1200': '81.96202',
                     '800': '81.93972',
                     '600': '81.92532',
                     '300': '81.89542'}
        c =      {'1200': 2.05,
                   '800': 3.1,
                   '600': 3.0,
                   '300':2.0 }
    else:
        print('select the beamline branch (A or B)')
                                                           # check the select energy makes sense
    if (grt == 1200) and (ph_en > 1500 or ph_en < 130):
              print('energy out of range for 1200 l/mm grating (130-1500 eV)')
    elif (grt == 800) and ( ph_en > 1500 or ph_en < 15):
        print('energy out of range for 800 l/mm grating (15-1500 eV)')
    elif (grt == 600) and (ph_en > 420 or ph_en <15):
        print('energy out of range for 600l/mm grating (15 - 420 eV)')
    elif (grt == 300) and (ph_en > 1500 or ph_en <15):
        print('energy out of range for 600l/mm grating (15 - 1500 eV)')
    elif (EPU == 57) and (ph_en > 1390 or ph_en < 140):
        print('EPU57 energy out of range  (140 - 1390 eV)')
    elif (EPU == 105) and (ph_en > 400 or ph_en < 20):
        print('EPU105 energy out of range (20 - 400 eV)')
    else:
        sh_close()

        GRT_name = str(grt)
        yield from mv (PGM.Mirror_Pitch_off, off_M2[GRT_name])  # set offset for M2
#        os.system('caput XF:21IDB-OP{Mono:1-Ax:8_MP}Mtr.OFF '+ off_M2[GRT_name])    # set offset for M2
        yield from mv (PGM.Grating_Pitch_off, off_GRT[GRT_name])  # set offset for GRT
#        os.system('caput XF:21IDB-OP{Mono:1-Ax:8_GP}Mtr.OFF '+ off_GRT[GRT_name])   # set offset for GRT
#        os.system('caput XF:21IDB-OP{Mono:1}:LINES:SET ' + GRT_name)                # tells to the PGM software which grating                                     
        yield from mv(PGM.Grating_Trans, GRT_TRANSLATION[GRT_name])                 # position the GRT translation 

        initial_M2_pitch = PGM.Mirror_Pitch.position         # read in the current M2 pitch
        initial_GRT_pitch = PGM.Grating_Pitch.position       # read in the current GRT pitch

        final_M2_pitch, final_GRT_pitch,C = angles_C(grt, ph_en, c[GRT_name])     #calculate theor. values

        n_steps = 5
        M2_steps=np.linspace(initial_M2_pitch, final_M2_pitch, num=n_steps)  # divide the range of motion of M2 in 'n_steps' even steps
        GRT_steps=np.linspace(initial_GRT_pitch, final_GRT_pitch, num=n_steps) # divide the range of motion of GRT in 'n_steps' even steps

        for i in range(n_steps):
            yield from mv(PGM.Mirror_Pitch, M2_steps[i])                                   # set position M2 pitch step by step in 10 times 
            yield from mv(PGM.Grating_Pitch, GRT_steps[i])                               # set position GRT pitch step by step in 10 times

        yield from mv(PGM.Focus_Const, c[GRT_name], PGM.Energy, ph_en )       # sets c and photon energy using the software of the PGM (this is necessary to set the experimental c) 

        if (EPU == 105):
            yield from mv(EPU105.gap, GAP_105(ph_en) )       # sets EPU2 (i.e. EPU105) gap according to experimental table 
        if (EPU == 57):
            yield from mv(EPU57.gap, GAP_57(ph_en) )        # sets EPU1 (i.e. EPU57) gap according to experimental table 
        
        sh_open()
        
########################################################
########### GRT #######################################
#######################################################


def test(g_o = 300, p_o =  83.7681090000):
    yield from bp.mv(PGM.Grating_lines, g_o)
    yield from bp.mv(PGM.Mirror_Pitch_off, p_o)    



    

def GAP_105(ph_en):
    ''' Given the photon energy (in eV) returns the gap for EPU105 according to measuraments
    with the PGM 800 l/mm grating
    '''
    if ph_en <20 or ph_en > 400:
        print("Out of energy range (30-400 eV)")
        return
    else:
        E_105 = [ 20., 25., 30., 35., 40., 45., 50., 55., 60., 65., 70., 75., 80., 85., 90., 95., 100.,\
                 120., 140., 160., 180., 200., 220., 240., 260., 280., 300., 320., 340., 360., 380., 400.]

        Gap_105 = [ 21.5, 25.1, 28.05, 30.6, 32.8, 34.7, 36.44, 38., 39.5, 40.84, 42.1, 43.24, 44.39, 45.4, 46.39, 47.3, 48.2,\
                   51.5, 54.3, 56.8, 59.1, 61.3, 63.2, 65.1, 66.9, 68.6, 70.3, 71.9, 73.5, 75.1, 76.7, 78.2]

        GAP_105 = interp1d(E_105, Gap_105)
        gap_105 =  (GAP_105(ph_en))
    return gap_105


def GAP_57(ph_en):
    ''' Given the photon energy (in eV) returns the gap for EPU57 according to measuraments
    with the PGM 800 l/mm grating
    '''
    if ph_en <140 or ph_en > 1390:
        print("Out of energy range (140-1390 eV)")
        return
    else:
        Gap_57 = [ 16.1, 16.2, 16.9, 17.6, 18.2, 18.8, 19.4, 19.9, 20.4, 20.9, 21.4, 21.9, 22.3, 22.7, 23.1,\
                 23.5, 23.95, 24.3, 22.3, 24.31, 26.1, 27.7, 29.2, 30.6, 31.9, 33.2, 34.5, 35.7, 37., 38.2,\
                 39.4, 40.6, 42., 43.2, 43.4, 44.7, 46.2, 47.8, 49.8, 52., 54.3, 57.7, 61.5]

        E_57 = [ 130., 140., 150., 160., 170., 180., 190., 200., 210., 220., 230., 240., 250., 260., 270., 280.,\
                 290., 300., 250., 300., 350., 400., 450., 500., 550., 600., 650., 700., 750., 800., 850., 900.,\
                 950., 1000., 1000., 1050., 1100., 1150., 1200., 1250., 1300., 1350., 1400.]

        GAP_57 = interp1d(E_57, Gap_57)
        gap_57 =  (GAP_57(ph_en))
    return gap_57







from ophyd import (EpicsSignal, Device, Component as Cpt)

import os
os.environ['EPICS_CA_ADDR_LIST']='10.16.2.59 10.16.2.60 10.16.2.61'

class FastShutter(Device):
    OPEN_SHUTTER = "open"
    CLOSE_SHUTTER = "closed"
    SETTLE_TIME = 0.1  # seconds
    output = Cpt(EpicsSignal,'{shutter:1}sts', string=True, put_complete=True)

    def open(self):
        self.output.set(FastShutter.OPEN_SHUTTER, settle_time=FastShutter.SETTLE_TIME)

    def close(self):
        self.output.set(FastShutter.CLOSE_SHUTTER, settle_time=FastShutter.SETTLE_TIME)


fast_shutter = FastShutter('XF:21IDB-BI', name='fast_shutter')


class Scintillator(Device):
    OPEN_SHUTTER = "open"
    CLOSE_SHUTTER = "closed"
    SETTLE_TIME = 0.1  # seconds
    output = Cpt(EpicsSignal,'{shutter:2}sts', string=True, put_complete=True)

    def open(self):
        self.output.set(Scintillator.OPEN_SHUTTER, settle_time=Scintillator.SETTLE_TIME)

    def close(self):
        self.output.set(Scintillator.CLOSE_SHUTTER, settle_time=Scintillator.SETTLE_TIME)


scintillator_shutter = Scintillator('XF:21IDB-BI', name='scintillator_shutter')

class PhotonShutter(Device):
    OPEN_SHUTTER = "Open"
    CLOSE_SHUTTER = "Not Open"
    output = Cpt(EpicsSignal, '{PSh}Pos-Sts', string=True, put_complete=True)
    
    def open(self):
        self.output.set(Scintillator.OPEN_SHUTTER)
        
    def close(self):
        self.output.set(Scintillator.CLOSE_SHUTTER)
            
photon_shutter = PhotonShutter('XF:21IDA-PPS', name='photon_shutter')


class FastShutter2(Device):
    OPEN_SHUTTER = "Force High"
    CLOSE_SHUTTER = "Force Low"
    SETTLE_TIME = 0.1  # seconds
    delay = Cpt(EpicsSignal, '-DlyGen:0}Delay-SP')
    width = Cpt(EpicsSignal, '-DlyGen:0}Width-SP')
    output = Cpt(EpicsSignal,'-Out:FP0}Src:Scale-SP', string=True, put_complete=True)

    
    def open(self):
        self.output.set(FastShutter2.OPEN_SHUTTER, settle_time=FastShutter2.SETTLE_TIME)

    def close(self):
        self.output.set(FastShutter2.CLOSE_SHUTTER, settle_time=FastShutter2.SETTLE_TIME)


fast_shutter2 = FastShutter2('XF:21ID-TS{EVR:C1', name='fast_shutter2')
