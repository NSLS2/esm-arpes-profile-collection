import IPython
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.optimize as opt
import os
from bluesky.plans import scan, adaptive_scan, spiral_fermat, spiral,scan_nd  
from bluesky.plan_stubs import abs_set, mv 
from bluesky.preprocessors import baseline_decorator, subs_decorator 
# from bluesky.callbacks import LiveTable,LivePlot, CallbackBase
#from pyOlog.SimpleOlogClient import SimpleOlogClient
from esm import ss_csv
from cycler import cycler
from collections import ChainMap
import math
import re
from builtins import input as pyinput
from boltons.iterutils import chunked
import sys
ip=IPython.get_ipython()



def macro1():
    #This macro is used to run a series of undulator spectra a different photon energies.

    grating='600' # the grating to use in the scans
    EPU='105'
#    Photon_list=(20,50,100,150,200) # the list of photon energies to use in the scan
    Photon_list=(25,40,60,80,100) # the list of photon energies to use in the scan
    Erange=[0.75,6.5]   #the photon energy %range for the scans, in the form [x,y] where the start
                      #value is x*Eph and the end value is y*Eph or the limit of the grating range.

    yield from Beamline.move_to('Branch_A')
                      
    for Ephoton in Photon_list:
        yield from Eph.move_to(Ephoton,grating=grating,EPU=EPU,branch='A',shutter='open')
        yield from M3_pitch_alignment(Branch='A')
        yield from Eph.move_to(Erange[0]*Ephoton,grating=grating,EPU=None,branch='A',shutter='open')
        yield from mv(ExitSlitA.h_gap,300)
        uid = yield from scan_1D('qem07',PGM.Energy,Erange[0]*Ephoton,
                                 min(Erange[1]*Ephoton,Eph.Range[grating][1]),0.5)

        yield from mv(BTA2diag.trans,-87)
        yield from mv(qem07.driver.range, '12 pC')

        uid = yield from scan_1D('qem07@4',PGM.Energy,Erange[0]*Ephoton,
                                 min(Erange[1]*Ephoton,Eph.Range[grating][1]),0.5)

        yield from mv(BTA2diag.trans,-63)
        yield from mv(qem07.driver.range, '350 pC')


def macro2():

    uid = yield from scan_multi_1D('qem12@2', PGM.Focus_Const, 3.0, 3.3, 0.05, PGM.Energy,64.550 , 64.725, 0.0005)    
    #This macro is used to aquire the image of the beam upstream of M3, it involves moving the the YaG
    #screen and collecting several images that need to be summed together to get the entire image.

    yield from mv(PGM.Focus_Const,3.2)
    
    slit_list=(1,1.25,1.5,1.6,1.7,1.8,1.9,2) # the list of photon energies to use in the scan

    for slit in slit_list:
        yield from mv(FEslit.h_gap, slit, FEslit.v_gap, slit)
        uid = yield from M3_pitch_alignment('B')
        uid = yield from scan_1D('qem12@2', PGM.Energy,64.55 , 64.725, 0.0005)

    yield from mv(shutter_FE,'Close')


def macro3():

    grating='800'
    EPU='105'
    branch='A'
    
#    slit_list=(1,10) # the list of photon energies to use in the scan
#    slit_list=(10) # the list of photon energies to use in the scan
    
    photon_list=(30,40,60,80,100,200) # the list of photon energies to use in the scan
#    photon_list=(40,60,80,100,200) # the list of photon energies to use in the scan

#    pitch_list=(-0.712829,-0.711738,-0.710757,-0.710112,-0.70986,-0.70916)
#    yield from mv(BTA2diag.trans,-63)
#    yield from mv(ExitSlitA.h_gap,50 , ExitSlitA.v_gap,30) 
    
#    for i,photon in enumerate(photon_list):
#        yield from mv(FEslit.h_gap, slit_list[0], FEslit.v_gap, slit_list[0])       
#        yield from Eph.move_to(photon,grating=grating,EPU=EPU,branch=branch,shutter='open')
#        yield from mv(M3.Ry,pitch_list[i])
#        yield from M3_pitch_alignment(branch)
#        yield from mv(ExitSlitA.h_gap,50 , ExitSlitA.v_gap,30) 
#        EPUgap=Eph.Und_e2g(photon,EPU)
#        yield from scan_1D('qem07',EPU105.gap,EPUgap-4,EPUgap+4,0.1)
        
#        for slit in slit_list:        
#            yield from mv(FEslit.h_gap, slit, FEslit.v_gap, slit)

# yield from scan_1D('qem07',EPU105.gap,EPUgap-4,EPUgap+4,0.1)

    yield from Eph.move_to(100,grating=grating,EPU=EPU,branch=branch,shutter='open')
#     yield from mv(FEslit.h_gap, 10, FEslit.v_gap, 10)
 
    uid = yield from scan_1D('qem07',PGM.Energy,90,720,0.25)
    yield from mv(EPU105.phase,34.7)
    uid = yield from scan_1D('qem07',PGM.Energy,90,720,0.25)
    yield from mv(EPU105.phase,0)

    yield from Eph.move_to(60,grating=grating,EPU=EPU,branch=branch,shutter='open')
    yield from M3_pitch_alignment(branch)
    uid = yield from scan_1D('qem07',PGM.Energy,50,520,0.25)

    yield from Eph.move_to(30,grating=grating,EPU=EPU,branch=branch,shutter='open')
    yield from M3_pitch_alignment(branch)
    uid = yield from scan_1D('qem07',PGM.Energy,20,420,0.25)
     

    for photon in photon_list:
        yield from Eph.move_to(photon,grating=grating,EPU=EPU,branch=branch,shutter='open')
        yield from scan_1D('Mir3_Cam10_U_1',M3Udiag.trans,-20,20,0.1)

            
    yield from mv(shutter_FOE,'close')




def macro4():
    #This macro is to run series of 2D scans over 100 um x 100 um with 2 micron step size.

    for i in range(10):
        
        uid=yield from scan_2D('qem08@2',LT.Y,52.45,52.55,.002,LT.X,15.3,15.4,.002,scan_type='XAS')

 
        
def macro5():
    #This macro is used to map the energy of the harmonics of EPU57. Using B-branch. Linear Vertical:phase,-28

    yield from mv(FEslit.h_gap, 1, FEslit.v_gap, 1)
    yield from mv(EPU57.phase,-28)
    yield from mv(ExitSlitA.h_gap,300)
    yield from mv(ExitSlitA.v_gap,10)
    yield from mv(BTA2diag.trans,-63)

    yield from scan_multi_1D('qem07@1', PGM.Energy, 130, 300, 10.0, EPU57.gap, 20, 30, .1, snake = True)    
    yield from scan_multi_1D('qem07@1', PGM.Energy, 250, 1000, 50.0, EPU57.gap, 20, 50, .1, snake = True)  
    yield from scan_multi_1D('qem07@1', PGM.Energy, 1000, 1500, 50.0, EPU57.gap, 20, 70, .11, snake = True)

    yield from mv(shutter_FOE,'close')    

def macro_pol1():
#   yield from mv(Pol.Rz,180)   yield from scan_multi_1D('qem12@3', EPU57.phase, 14, 28, 2, EPU57.gap, 16, 32, .1)
    yield from mv(Pol.Rz,90)
    yield from scan_multi_1D('qem12@3', EPU57.phase, 4, 14, 2, EPU57.gap, 25, 32, .1)
  
        
def macro_pol():

    phase = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    gap = [29.7, 29.6, 29.4, 29.0, 28.4, 27.7, 26.8, 25.9, 24.9, 23.9, 22.9, 22.1, 21.4, 21.0, 20.8]

    
    for i in range(len(gap)):
        yield from mv(EPU57.phase,phase[i])
        yield from mv(EPU57.gap,gap[i])
        yield from scan_1D('qem12@3', Pol.Rz, 0, 360, 5)



def macro_LV_105_maps():

    yield from mv(FEslit.h_gap, 1, FEslit.v_gap, 1)
    yield from mv(EPU105.phase,-52.5)
    yield from mv(ExitSlitB.h_gap,300)
    yield from mv(ExitSlitB.v_gap,10)
    yield from mv(BTB2diag.trans,-63)

    yield from Eph.move_to(80, grating='300', EPU=None, shutter='open')

    yield from scan_multi_1D('qem12@1', PGM.Energy, 80, 500, 10.0, EPU105.gap, 20, 100, .1, snake = True)    
#    yield from scan_multi_1D('qem12@1', PGM.Energy, 250, 1000, 50.0, EPU57.gap, 20, 50, .1, snake = True)  
#    yield from scan_multi_1D('qem12@1', PGM.Energy, 1000, 1500, 50.0, EPU57.gap, 20, 70, .11, snake = True)
    yield from mv(shutter_FE,'Close')




def macro_exposure_V_ast(X_init = 10.8):
    Ast_V = [-2, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    X = np.arange(X_init+2, X_init+0.2, -0.2 )
        
    for i in range(len(X)):
        yield from mv(LT.X ,X[i])
        yield from mv(M4A.VFM_Mirror_Astig,Ast_V[i])
        gv_temp_open('XF:21IDC-VA{BT:A2-GV:A2_D_1}',3)

def macro_exposure_time_H(X_init = 10.8):    
    time = [1,2,3,4,5,7,9,11,13,16]
    step = -0.2
    X = np.arange(X_init+2, X_init+2+step*len(time), step)        
    for i in range(len(X)):
        yield from mv(LT.X ,X[i])
        gv_temp_open('XF:21IDC-VA{BT:A2-GV:A2_D_1}',time[i])

def macro_exposure_square(X_init = 10.8, Y_init = 68.4):
    step = 0.02
    X = np.arange(X_init-step*5,X_init, step)
    Y = np.arange(Y_init, Y_init+step*5, step)
    for i in X:
        for j in Y:
            yield from mv(LT.X ,i)
            yield from mv(LT.Y ,j)
            gv_temp_open('XF:21IDC-VA{BT:A2-GV:A2_D_1}',16)






    
def macro_exposure_H_ast(X_init = 10.8, Y_init=68.2):
    Ast_H = [-2.1, -1.6, -1.1, -0.6, -0.1, 0.4, 0.9, 1.4, 1.9, 2.4 ]
    X = np.arange(X_init, X_init-1.8, -0.2 )
    H_inout= [-2.5, -2.75, -3.0, -3.25, -3.5]
    Y=Y_init
    for j in range(len(H_inout)):
        yield from mv(LT.Y ,Y+j*0.2)
        yield from mv(M4A.HFM_Mirror_InOut, H_inout[j])
        for i in range(len(X)):
            yield from mv(LT.X ,X[i])
            yield from mv(M4A.HFM_Mirror_Astig,Ast_H[i])
            gv_temp_open('XF:21IDC-VA{BT:A2-GV:A2_D_1}',3)



#def macro_exposure(time =  1, V = 10, H = 10, j=0):
    #for i in range(V):
    #    for j in range(H):
    #yield from mv(M4A.HFM_Ry ,j)
    #        yield from mv(M4A.VFM_Rx ,i)
    #        yield from gv_temp_open('XF:21IDD-VA{ANAL:1A-GV:EA1_1}',3)
