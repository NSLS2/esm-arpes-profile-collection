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
#from esm import ss_csv
from cycler import cycler
from collections import ChainMap
import math
import re
from boltons.iterutils import chunked
import sys
try:
    from suitcase import hdf5
except:
    print("Cannot import hdf5 from suitcase")
from builtins import input as pyinput
ip=IPython.get_ipython()
import datetime
import time
from ophyd import EpicsSignal


###Utilities
###    The following set of code is used to create a series of utility functiosn for use at ESM.

def ESM_export(scan_ID,filename):
        '''
        This routine is used to export a data file to an HDF5 file for transfer to other analysis
        software.

        PARAMETERS
        ----------

        scan_ID : list.
            The scan id numbers (as a list) for the data to export, can be -1, -2 to indicate the last
            or second to last scan etc.

        filename : str.
            The filename to use for the output data file.

        filestring : str, output
            The file path, and file name, of the file that was created.

        '''
        #reference the header file for the scan to be saved.
        hdr=db[scan_ID]

        # check if the filename includes the '.h5' extension and add it if not.
        if not filename.endswith('.h5'):
                filename+='.h5'

        # give the filepath of where to store the h5 file
        filepath='/direct/XF21ID1/hdf5_files/'+datetime.today().strftime('%Y_%m')

        #check if a current month directory exists, if not create it.
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        #create the file path and file name string
        filestring=filepath+'/'+filename

        #check if the file already exists, if so ask user if overwrite is necesary.
        if os.path.isfile(filestring):
                if ask_user_continue('The file already exists, continuing will overwrite it')==0:
                    raise RuntimeError('user quit save')

                else:
                    os.remove(filestring)

        # actually write the new file
        hdf5.export(hdr,filestring)

        # return the file path and file name so the user can find it.
        return filestring


def channel_list_unpack(DETS_str, dot=False):
        '''
        This function is used to unpack the detector input string and return a list of channels

        This function is used to unpack the detector list string used as an input into our "scans"
        and "notebook" routines and returns a list of channels that it relates to for plotting and/
        or data analysis.

        PARAMETERS
        ----------

        DETS_str : str
            The input string that is to be "unpacked" into a channel list. This needs to have the
                format definitions:
                    - DETX is the name of 'Xth' detector.
                    - every '@' symbol defines a new channel number for the preceeding detector.
                    - ChX is the channel number of the 'Xth' channel.
                    -    if ChX is 0 then it returns no channels for this detector.
                    -    if ChX is -1 then it returns all channels for this detector.
                    - every '-' symbol defines a new value for the preceeding channel number.
                    - ValX is the name of the 'Xth' value, and can be 'total', 'max' or 'min'.

                format1:  'DET1@Ch1-Val1-Val2-...@Ch2-Val1-...., DET2@Ch1-Val1-...@.... ,....'
                    If no '@' is present for a detector then it reverts to the default of channel 1.
                    If no '-' is present it reverts to the default 'Total'.

                format2: 'DET1,    DET2    ,......,   @Ch1-val1-val2-...@Ch2-val2-...@...'
                    This returns The channels and values defined by the last list entry for all
                    detectors.
                    If no '-' is present it reverts to the default 'Total'.
        dot : Boolean, optional
            A boolean that indicates if the attribute separator in the returned string should be '.' (True)
            or '_' (False). Default is False.



        name_list : list, output
            The output list of channels.
        '''
        name_list=[]
        #define the empty output list

        #split the detectors str into a list of detector strs
        DET_list=DETS_str.split(',')

        # check if format 1 or format 2 is being used.

        if  DET_list[-1].startswith('@'):
            #If format 2 is used
            Channel_list=DET_list[-1][1:].split('@')
            DET_list=DET_list[:-1]
            Format = 2
        else:
            Format = 1

        for i,DET_str in enumerate(DET_list):
            DET=DET_str.partition('@')[0]
            if Format == 1:
            #If format 1 is used
                Channel_list=DET_str.partition('@')[2].split('@')

            if len(Channel_list[0]) == 0:
                #if no channels are defined for this detector.
                name_list.append(format_channel_name(DET,1,Value='total',dot= dot))

            elif '-1' in Channel_list:
                #if all channels are defined. for this detector.
                name_list+= format_channel_name(DET,-1, dot = dot).split(',')

            elif '0' not in Channel_list:
                #if some channels are defined for this detector.
                for j,Channel_str in enumerate(Channel_list):
                    Channel=Channel_str.partition('-')[0]
                    Value_list=Channel_str.partition('-')[2].split('-')

                    if len(Value_list[0]) == 0:
                        #if no valuess are defined for this detector.
                        name_list.append(format_channel_name(DET,Channel,dot = dot))
                    else:
                        #if there are values listed for this channel.
                        for k,Value in enumerate(Value_list):
                            name_list.append(format_channel_name(DET,Channel,Value, dot = dot))



        return name_list

def format_channel_name(DET,Channel,Value='total', dot = False):
        '''
        This function formats the channel name for a given detector type.

        This function takes in the detctor name, channel number, and an optional channel value
        and returns a formated channel name string for this type of detector. If 'Channel' is -1
        then it returns a string containg all the possible channel names for the detector,
        seperated by commas.

        PARAMETERS
        ----------
        DET: str
            The name of the detector for which the channel is to be formatted

        Channel: integer
            The channel number for the  channel to be formatted

        Value: str,optional
            The optional "value" for the channel number.
        dot : Boolean, optional
            A boolean that indicates if the attribute separator in the returned string should be '.' (True)
            or '_' (False). Default is False.

        channel_name : str
            The output string that is the formatted channel name.
        '''
        if dot:
            sep = '.'
        else:
            sep = '_'

        if 'qem' in DET.lower():
            #if the detector is a qem.
            if Channel == -1:
                channel_name=''
                for i in range(1,5):
                    if i > 1: channel_name+=','
                    channel_name+=DET+sep+'current'+str(i)+sep+'mean_value'

            else:
                channel_name=DET+sep+'current'+str(Channel)+sep+'mean_value'

        elif 'cam' in DET.lower():
            #if the detctor is a camera.
            if Channel == -1:
                channel_name=''
                for i in range(1,5):
                    if i > 1: channel_name+=','
                    channel_name+=DET+sep+'stats'+str(i)+sep+'total,'
                    channel_name+=DET+sep+'stats'+str(i)+sep+'max_value,'
                    channel_name+=DET+sep+'stats'+str(i)+sep+'min_value'
            else:
                if 'max' in Value or 'min' in Value:
                    Value+='_value'
                channel_name=DET+sep+'stats'+str(Channel)+sep+Value

#        elif 'mbs' in DET.lower():
#	        channel_name='mbs'
        else:
            #If the detcor type has not been determined.
            raise ValueError("Detector type not recognised, name must contain 'qem' or 'cam'.")



        return channel_name


def ask_user_continue(request_str):
        '''
        This function asks the user to confirm that the current process should be completed.

        This function asks the user, using the request_str to give specifics, if they should continue.

        PARAMETERS
        ----------

        request_str : str
            The output string given with the user prompt.
        '''

        valid={'yes':1,'no':0}
        prompt_str = ', continue(yes or no):'

        while True:
            sys.stdout.write(request_str + prompt_str)
            choice = pyinput().lower()
            if choice in valid:
                return valid[choice]



def SiC2F(photon_energy, current):
        '''
        This routine is used to convert an XUV Si-diode current to flux.

        Given the photon energy (in eV) and the XUV Si-diode current (in Amp), returns the flux
        (ph/sec). It uses the QY for a typical XUV Si-diode.

        PARAMETERS
        ----------

        photon_energy : float
            The photon energy that was incidnent on the diode.

        current : float
            The measured current on the diode.

        flux : float, output
            The flux that is returned.

        '''



        # XUV QY data: number of electrons per 1 photons of a given photon energy
        SiC2F_data={ 'E_eV' : [1,1.25,2,2.75,3,4,5,6,7,8,9,10,20,40,60,80,100,120,140,160,180,200,220,
                               240,260,280,300,320,340,360,380,400,420,440,460,480,500, 520, 540,560,
                               580,600,620,640,740,760,780,800,820,840,860,880,900,920,940,960,980,1000,
                               1200,1400,1600,1800,2000,2200,2400,2600,2800, 3000,3200,3400,3600,3800,
                               4000,4200,4400,4600,4800,5000,5200,5400,5600,5800,6000] ,
                     'QY' : [0.023, 0.32 ,0.64,0.57,0.49,0.432,0.45,0.5,0.7,1,1.05,1.1,3.25,8.38,12.18,
                             17.85,22.72,26.42,31.37,33.5,42.95,50.68,60.61,66.12,71.63,77.13, 82.64,
                             88.15,93.66,99.17,104.68,110.19,115.7,121.21,126.72,132.23,137.74,143.25,
                             148.76,154.27,159.78,165.29,170.8,176.31,203.86,209.37, 214.88,220.39,
                             225.9,231.41,236.91,242.42,247.94,253.44,258.95,264.46,269.97,275.48,
                             330.58,385.67,440.77,495.87,550.96,606.06,661.18,716.35, 771.35,826.45,
                             881.54,936.64,991.74,1046.83,1101.93,1157.02,1212.12,1267.22,1322.31,
                             1377.41,1432.51,1487.6,1542.7,1597.8,1652.89]}
        SiCtoF = interp1d(SiC2F_data['E_eV'],SiC2F_data['QY'])

        if (photon_energy < min(SiC2F_data['E_eV']) or photon_energy > max(SiC2F_data['E_eV'])):
                raise RuntimeError('photon energy outside of range of conversion data')
        else:
            flux =  (current)/(1.6E-19)/SiCtoF(photon_energy)
            return '{:.3e}'.format(flux)


def gaussian_1D(x,params):
    '''This function defines a 2D gaussian that is used for fitting.
    Parameters
    ----------
    x : variable
        the axis variable for the 1D gaussian.
    params : variables
        the list of "fitting" variables for the 2D gaussian
            Parameters:

            1. amp  =  the amplitude of the 1D guassian
            2. cen =  the centre of the first axis gaussian
            4. std = the width of the first axis gaussian
            6. bkg  = height of the constant background for the gaussian
    '''
    amp,cen,std,bkg = params
    return amp*np.exp(-(x-cen)**2/2/std**2)+bkg

def gaussian_1D_error(params,y,x):
    '''This function defines the difference between a fitted gaussian and the raw data.
    Parameters
    ----------
    params : variables
        the list of "fitting" variables for the 2D gaussian
            Parameters:

            1. amp  =  the amplitude of the 1D guassian
            2. cen =  the centre of the first axis gaussian
            4. std = the width of the first axis gaussian
            6. bkg  = height of the constant background for the gaussian

    y  : variable
        the value of the raw data at (x)
    x : variable
        the axis variable for the 1D gaussian.
    '''

    return gaussian_1D(x, params)-y


def gaussian_2D(x1,x2,params):
    '''This function defines a 2D gaussian that is used for fitting.
    Parameters
    ----------
    x1 : variable
        the first axis variable for the 2D gaussian.
    x2 : variable
        the second axis variable for the 2D gaussian.
    params : variables
        the list of "fitting" variables for the 2D gaussian
            Parameters:

            1. amp  =  the amplitude of the 2D guassian
            2. cen1 =  the centre of the first axis gaussian
            3. cen2 = the centre of the 2nd axis guassian
            4. std1 = the width of the first axis gaussian
            5. std2 = the width of the second axis gaussian
            6. bkg  = height of the constant background for the gaussian
    '''
    amp,cen1,cen2,std1,std2,bkg = params
    return amp*np.exp(-(x1-cen1)**2/2/std1**2-(x2-cen2)**2/2/std2**2)+bkg

def gaussian_2D_error(params,y,x1,x2):
    '''This function defines the difference between a fitted gaussian and the raw data.
    Parameters
    ----------
    params : variables
        the list of "fitting" variables for the 2D gaussian
            Parameters:

            1. amp  =  the amplitude of the 2D guassian
            2. cen1 =  the centre of the first axis gaussian
            3. cen2 = the centre of the 2nd axis guassian
            4. std1 = the width of the first axis gaussian
            5. std2 = the width of the second axis gaussian
            6. bkg  = height of the constant background for the gaussian
    y  : variable
        the value of the raw data at (x1, y1)
    x1 : variable
        the first axis variable for the 2D gaussian.
    x2 : variable
        the second axis variable for the 2D gaussian.
    '''

    return gaussian_2D(x1, x2, params)-y


def fit_Gauss_1Dseries(uid,initial_guess):
    '''
    This scan fits 1D Gaussian curves o each line in a 2D dataset giben by uid.

    This scan is used to fit to a 1D Gaussian to each line in a 2D dataset, the output is then a set of 1D data corresponding to the
    amplitude, position and linewidth as a fucntion of the Y axis of the dataset. The function returns a list with 4 items, the items
    being the data for amplitude,position, linewidth, background offset and y_sequence number.

    Parameters
    ----------
    uid : number
        This is the uid used to extract the data from the databroker.

    initial_guess : list
        This is the initial guess for the amplitude, centre, width and background offset, in a list in this order.

    '''

    #reference the data
    hdr=db[uid]

    #find out the shape of the data
    x_num = hdr.start['X_num']
    y_num = hdr.start['Y_num']

    #Load the data from the databroker.

    amplitude=[]
    linewidth=[]
    position=[]
    background=[]
    y_seq=[]

    #step through each "row" and fit a gaussian.
    for y_step in range(0,y_num):
        x = hdr.table()[hdr.start['plot_Xaxis']][x_num*y_step:x_num*(y_step+1)-1]
        y_seq.append(y_step)
        data = hdr.table()[hdr.start['plot_Zaxis']][x_num*y_step:x_num*(y_step+1)-1]

        popt_row, pcov_row = opt.leastsq(gaussian_1D_error,x0=initial_guess,
                                         args=(data[hdr.start['plot_Zaxis']],
                                               x[hdr.start['plot_Xaxis']]))
        amplitude.append(popt_row[0])
        position.append(popt_row[1])
        linewidth.append(popt_row[2])
        background.append(popt_row[3])

        initial_guess[0]=popt_row[0]
        initial_guess[1]=popt_row[1]
        initial_guess[2]=popt_row[2]
        initial_guess[3]=popt_row[3]


    return [amplitude,linewidth,position,background,y_seq]


def max_in_1D(scan_id):
    '''
    This scan is used to find the maximum value in a 1D data set and return the max value, and the x co-ordinate.

    This scan is used to find the maximum value in a 1D dataset and return the max value, and the x co-ordinate.
    It returns a list containing the x and y values for the maximum y value in the dataset.

    Parameters
    ----------
    scan_id : number
        This is the uid used to extract the data from the databroker.

    '''

    scan = db[scan_id]
    if scan.start['plot_Xaxis'][0].startswith('FE'):
        Xname = scan.start['plot_Xaxis'][0].replace('_readback', '_setpoint')
    else:
        Xname = scan.start['plot_Xaxis'][0]

    data2D = scan.table()[[Xname, scan.start['plot_Yaxis'][0]]]
#    del data2D['time']

    max_idx = np.argmax(data2D[scan.start['plot_Yaxis'][0]], axis=None)

    return [data2D[Xname][max_idx],data2D[scan.start['plot_Yaxis'][0]][max_idx]]


def max_in_2D(scan_id):
    '''
    This scan is used to find the maximum value in a 2D data set and return the max value, and the x and y co-ordinates.

    This scan is used to find the maximum value in a 2D dataset and return the max value, and the x and y co-ordinates.
    It returns a list containg the x, y and z values for the maximum z value in the dataset.

    Parameters
    ----------
    scan_id : number
        This is the uid used to extract the data from the databroker.

    '''

    scan = db[scan_id]
    if scan.start.plot_Xaxis[0].startswith('FE'):
        Xname = scan.start['plot_Xaxis'][0].replace('_readback', '_setpoint')
        Yname = scan.start['plot_Yaxis'][0].replace('_readback', '_setpoint')
    else:
        Xname = scan.start['plot_Xaxis'][0]+'_user_setpoint'
        Yname = scan.start['plot_Yaxis'][0]+'_user_setpoint'

    data3D = scan.table()[[Xname, Yname, scan.start['plot_Zaxis'][0]]]
    del data3D['time']

    max_idx = np.argmax(data3D[scan.start['plot_Zaxis'][0]], axis=None)

    return [data3D[Xname][max_idx],data3D[Yname][max_idx],data3D[scan.start['plot_Zaxis'][0]][max_idx]]


def scan_info(scan_id_list,Baseline=False,Detector=False):
    '''
    This routine is used to return a formatted string containing the relevant information from the
    for the scan defined by scan_id.

    This scan looks up metadata and prints out some relevant info from this, it can also optionally
    include the baseline state stream from scan_id and removes everything but the readback
    values. It then returns a formatted string containing these values to make finding the information
    required easier.

    Parameters
    ----------
    scan_id_list : number
        This is a list of the uid, scan id or previous scan number (-1,-2 etc.) used to extract the
        data from the databroker.

     Baseline : Boolean, optional
            Used to include baseline readings (i.e. data recorded that is not directly part of the
            scan). to include baseline data use Baseline=True)

     detector : Boolean, optional
            Used to include detectro configuration readings (i.e. detector settings that is not directly
            part ofset prior to the scan). to include detector settings use Baseline=True)

     f_string : str
         Possible output string for formatting.

    '''

    #define the output string
    f_string=''

    for scan_id in scan_id_list:
        #Load up the baseline data stream and extract out the list of keys that are not setpoints or
        #done indicators
        data = db[scan_id].table(stream_name='baseline')
        key_list=[key for key in data.keys() if '_setpoint' not in key and '_done' not in key ]

        #Extract out a list of devices from the list of keys.
        temp_list=key_list
        if 'time' in temp_list: temp_list.remove('time')
        device_dict={}
        exit_val=0

        #continue looping over the list of remaining axes until none exist.
        while len(temp_list)>0 and exit_val<=200:
            device_name,_,channel = temp_list[0].partition('_')
            axis_list=list(key for key in temp_list if key.startswith(device_name))
            temp_device_dict = {key:data[key] for  key in axis_list}
            device_dict[device_name]=temp_device_dict

            temp_list = list(dev for dev in temp_list if not dev.startswith(device_name) )
            exit_val+=1

        # calculate how long the scan took
        scan_time=(db[scan_id].stop['time']-db[scan_id].start['time'])

        #create the formatted string heading
        f_string+='\n\n************************************************************\n'
        f_string+='Scan id '+str(db[scan_id].start['scan_id'])+' ,  '
        f_string+=datetime.datetime.fromtimestamp(int(db[scan_id].start['time'])).strftime('%Y-%m-%d %H:%M:%S') +'\n'
        f_string+='************************************************************\n\n'

        f_string+='***** Scan Data *****\n\n'
        f_string+='Scan type : '+str(db[scan_id].start['scan_type'])
        f_string+=' , Scan name : '+str(db[scan_id].start['scan_name'])
        f_string+=' , Plan name : '+str(db[scan_id].start['plan_name'])+'\n'
        f_string+='Detector(s) : '+str(db[scan_id].start['detectors'])+'\n'
        f_string+='num_points : '+str(db[scan_id].start['num_points'])
        f_string+=', scan time : '+time.strftime("%H:%M:%S", time.gmtime(scan_time))
        f_string+=', time/point : '+time.strftime("%H:%M:%S", time.gmtime(scan_time/db[scan_id].start['num_points']))+'\n'

        if db[scan_id].start['scan_name'].endswith('_2D'):
            f_string+='Plotted channel(s) : '+str(db[scan_id].start['plot_Zaxis'])+'\n'
            f_string+='X axis : '+str(db[scan_id].start['plot_Xaxis'])+'( '
            f_string+= str(db[scan_id].start['X_start'])+' , '+str(db[scan_id].start['X_stop'])
            f_string+=' , '+str(db[scan_id].start['X_delta'])+' )'+'\n'
            f_string+='Y axis : '+str(db[scan_id].start['plot_Yaxis'])+' ( '
            f_string+= str(db[scan_id].start['Y_start'])+' , '+str(db[scan_id].start['Y_stop'])
            f_string+=' , '+str(db[scan_id].start['Y_delta'])+' )'+'\n\n'
        elif db[scan_id].start['scan_name'].endswith('_1D'):
            f_string+='Plotted channel(s) : '+str(db[scan_id].start['plot_Yaxis'])+'\n'
            f_string+='X axis : '+str(db[scan_id].start['plot_Xaxis'])+' ( '
            f_string+= str(db[scan_id].start['plan_args']['args'][-2])+' , '
            f_string+= str(db[scan_id].start['plan_args']['args'][-1])+' , '
            f_string+= str(db[scan_id].start['delta'])+' ) '+'\n\n'


        if Detector is True:
            #If the user asked for detector info
            dets=db[scan_id].start['detectors']
            f_string+='***** Detector Settings *****\n\n'
            hdr=db[scan_id]

            for det in dets:
                f_string+='    '+det+':\n'
                keys=list(hdr.config_data(det)['primary'][0].keys())
                #extract out the list of configuration items for this detector and step through them.
                for key in keys:
                    set_list = key.split('_')
                    del set_list[0]
                    f_string+='\t '
                    n_string=''
                    for set in set_list:
                            n_string+=set+' '
                    f_string+=n_string.ljust(25)
                    f_string+=':\t '+ str(hdr.config_data(det)['primary'][0][key])+'\n'

            f_string+='\n\n'


        if Baseline is True:
            f_string+='***** Baseline Readings *****\n\n'
            #step through each of the 'devices' and print out the axes associated with it
            device_list=list(device_dict.keys())
            device_list.sort()

            for dev in device_list:
                f_string+='    '+dev+':\n'
                axis_dict = device_dict[dev]
                for axis in axis_dict:
                    f_string+='\t '+axis.ljust(30)+':  start = %f\t , end = %f\n ' % (axis_dict[axis][1],axis_dict[axis][2])

                f_string+='\n'



    print (f_string)




#import time

fug_current = EpicsSignal(name="fug_current",read_pv="XF:21IDA-ES{FUG:Fil}I-I_units",write_pv="XF:21IDA-ES{FUG:Fil}I:OutMain-SP_units")
fug_voltage = EpicsSignal(name="fug_voltage",read_pv="XF:21IDA-ES{FUG:Fil}E-I_units",write_pv="XF:21IDA-ES{FUG:Fil}E:OutMain-SP_units")
fug_enable = EpicsSignal(name="fug_enable",read_pv="XF:21IDA-ES{FUG:Fil}Enbl:OutMain-Sts",write_pv="XF:21IDA-ES{FUG:Fil}Enbl:OutMain-Cmd")
#prep_pressure = EpicsSignal(name="prep_pressure",read_pv="XF:21IDA-VA{BT:5-IP:5_1}P-I")
prep_pressure = EpicsSignal(name="prep_pressure",read_pv="XF:21IDD-VA{PREP:2A-CCG:EA5_1}P-I")



def fug_fil_degas(I_target, pressure_pv, max_pressure, max_voltage, current=fug_current, voltage=fug_voltage, enable=fug_enable):
    """
    fug_fil_degas(1 (in A), prep_pressure, 1*10**-8, 10 (in Volts))
    """

#    pressure_pv_dict = {
#            ['Load_Lock_1']= 'XF:21IDD-VA{LOCK1:4A-CCG:EA4_1}P-I',
#            ['Load_Lock_2']= 'XF:21IDD-VA{LOCK2:3A-CCG:EA3_1}P-I',
#            ['Preparation']= 'XF:21IDD-VA{PREP:2A-CCG:EA5_1}P-I',
#            ['Low_Temp']= 'XF:21IDD-VA{LOWT:5A-CCG:EA5_1}P-I',
#            ['Analysis']= 'XF:21IDD-VA{ANAL:1A-CCG:EA1_1}P-I' }


    pressure_pv = EpicsSignal(pressure_pv_dict[pressure_pv])

    voltage.put(max_voltage)
    current.put(0)
    enable.put(1)
    while(I_target > current.get()):
        incr = .1
        print("Current value: " + str(round(current.get(),2)) + ". Going to " + str(round(current.get()+incr,2))) 
        while(pressure_pv.get() >= max_pressure):
            time.sleep(10)
        current.put(current.get() + incr)
        time.sleep(10)
    current.put(0)
    voltage.put(0)
    enable.put(0)
    
    

#fil_degas(fug_current, prep_pressure, 10, 1*10**-8)




from ophyd import EpicsSignal
import time

def gv_temp_open(sysdev, duration):
    """
    gv_temp_open("XF:21IDB-VA{BT:10-GV:10_D_1}", 1.5 (seconds))
    """
    open_pv_str = sysdev + "Cmd:Opn-Cmd"
    close_pv_str = sysdev + "Cmd:Cls-Cmd"
    open_cmd = EpicsSignal(name="open_cmd",read_pv=open_pv_str,write_pv=open_pv_str)
    close_cmd = EpicsSignal(name="close_cmd",read_pv=close_pv_str,write_pv=close_pv_str)
    open_cmd.put(1)
    time.sleep(duration)
    close_cmd.put(1)



def XY_max(scan_id):
    ''' 
    finds the max value (Y) of a scan (scan_id) and returns it toghether with the correponding X value
    '''
    hdr = db[scan_id]
    Y = hdr.table(fields = hdr.start['plot_Yaxis'])
    Y_max = Y.max()[1]
    index_max = Y.loc[ Y[ hdr.start['plot_Yaxis'][0] ]== Y_max ].index[0]
    X = hdr.table(fields = hdr.start['plot_Xaxis'])
    X_max = X.loc[index_max][1]
    return(X_max, Y_max) 
    

from six.moves import cPickle as pickle #for performance

def save_dict(di_, filename_):
    try:
        with open(filename_, 'wb') as f:
            pickle.dump(di_, f)
    except IOError as e:
        print(e)
def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di



def pv_time(t = '2019-02-15 08:25:03',
            pv = 'XF:21IDB-OP{Mono:1-Ax:8_Eng}Mtr.VAL', 
            archiver_addr = "http://xf21id1-ca1.cs.nsls2.local:17668/retrieval/data/getData.json"):
    
    
    import pytz
    import time
    import datetime as dt
    import pandas as pd
    import requests
    import numpy as np


    since = dt.datetime.strptime(t , '%Y-%m-%d %H:%M:%S') - dt.timedelta(seconds=0)
    until = dt.datetime.strptime(t , '%Y-%m-%d %H:%M:%S')


# request data from archiver

    timezone = 'US/Eastern'
    since = pytz.timezone(timezone).localize(since).replace(microsecond=0).isoformat()
    until = pytz.timezone(timezone).localize(until).replace(microsecond=0).isoformat()

    params = {'pv': pv, 'from': since, 'to': until}


    req = requests.get(archiver_addr, params=params, stream=True)
    req.raise_for_status() 

# process data

    raw, = req.json()

    secs = [x['secs'] for x in raw['data']]
    nanos = [x['nanos'] for x in raw['data']]
    data = [x['val'] for x in raw['data']]

    asecs = np.asarray(secs)
    ananos = np.asarray(nanos)

    times = asecs*1.0e+3 + ananos*1.0e-6

    datetimes = pd.to_datetime(times, unit='ms')

 

# create and print the DataFrame
 
    df = pd.DataFrame()

    df['time'] = datetimes
    df['data'] = data

    df.time = df.time.dt.tz_localize('UTC').dt.tz_convert('US/Eastern')

                            
    return df['data'].values[-1]


def esm_status(t= '', comp=False):
    ''' time format= 'YYYY-m-d H:M:S'
    usage: 
    1) esm_status() --> prints the current status
    2) esm_status(t='2020-01-15 15:30:00') --> prints status at specified time
    3) esm_status(t='2020-05-17 19:30:45', comp=True) --> prints pvs with more than 5% difference between now and at time t 
    '''
    DT_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if t == '':
        DT = DT_now

    else:
        DT = t
        
    print('Now: {}, then:{}'.format(DT_now, DT))

    pv_dict = load_dict('/home2/xf21id1/.ipython/profile_collection/startup/status_files/mtr_pv_dict.npy') 

    for k in pv_dict.keys():
        try:
                value = pv_time(t = DT, pv = pv_dict[k],
                                archiver_addr = "http://xf21id1-ca1.cs.nsls2.local:17668/retrieval/data/getData.json")
        except:
                value = pv_time(t = DT, pv = pv_dict[k],
                                archiver_addr = 'http://arcapp01.cs.nsls2.local:17668/retrieval/data/getData.json')
        try:
                value = value[-1]
        except:
                None

        if comp:
            try:
                    value_now = pv_time(t = DT_now, pv = pv_dict[k],
                                        archiver_addr = "http://xf21id1-ca1.cs.nsls2.local:17668/retrieval/data/getData.json")
            except:
                    value_now = pv_time(t = DT_now, pv = pv_dict[k],
                                        archiver_addr = 'http://arcapp01.cs.nsls2.local:17668/retrieval/data/getData.json')

            try:
                value_now = value_now[-1]
            except:
                None

            if value == None and value_now == None:
                print('No values for {}'.format(k))
            elif np.abs((value - value_now)) > 0.01:
                print('{0:20}: then = {1:10.4f}\t now = {2:10.4f}\t diff = {3:10.4f}'.format(k, value, value_now, np.abs((value - value_now))))
        else:
            print('{0:20} {1:10.4f}'.format(k, value))
