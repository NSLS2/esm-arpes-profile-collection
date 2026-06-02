ui = self.ui
from bsstudio.functions import widgetValue
from ophyd import EpicsSignal
from PyQt5.QtWidgets import QWidget, QGroupBox
from bsstudio.widgets.MBSTable import MBSTable
import nexusformat.nexus as nx 
from nexusformat.nexus import NXfield, NXsource, NXmonochromator, NXpositioner, NXuser, NXnote



# log file
import csv
with open('/home2/xf21id1/hv_log.csv', mode='a') as f:
	csv_writer = csv.writer(f, delimiter=',')
	csv_writer.writerow([ ' '])

branch = 'A'

#### read in parameters

if widgetValue(ui().ckb_M3_adj):
	M3_Ry = eval('M3.Ry.user_readback.get()')

grt = ui().parentUi().embedframe_16.frameUi().cb_grt.currentText()
#print('epu=', ui().parentUi().embedframe_16.frameUi().cb_epu.currentText())
pol = ui().Light_pol_CB.currentText()
epu = ui().epu_cb.currentText()	

if widgetValue(ui().ckb_hv_ls):
	hv_ls= ui().en_ls_LE.text().split(',')
	en_ls =  [i.split('_')[0] for i in hv_ls] 
	pol_ls = [i.split('_')[1] for i in hv_ls]
else:
	en_start = float(ui().PGM_start_LE.text())
	en_stop = float(ui().PGM_stop_LE.text())
	en_step = float(ui().PGM_step_LE.text())
	en_ls =[i for i in np.arange(en_start, en_stop+1, en_step)]
	pol_ls = [pol]*len(en_ls)
tot = str(len(en_ls)) 
en_step = float(ui().PGM_step_LE.text())
print(en_ls)
print(pol_ls)


#center = float(ui().MBS_center_LE.text())
#iterations = ui().MBS_iterations_LE.text()

center = mbs.center_ke.get()
iterations =mbs.num_scans.get()
if iterations ==0 or widgetValue(ui().ckb_3D):
	iterations = 1
	mbs.num_scans.set(1).wait()

mbs.dith_steps.set(18).wait()

Ai = float(ui().Ai_LE.text())
Af = float(ui().Af_LE.text())
Ad = float(ui().Ad_LE.text())

# sets the PV in MBS according to the selected scan 
	# set the optional param (the same in all MBS scans)


#if width != '': mbs.width.put(float(width))
#if iterations != '': mbs.num_scans.put(int(iterations))
#if dwell_time != '': mbs.frames.put(float(dwell_time))
#if en_delta != '': mbs.step_size.put(float(en_delta))



if widgetValue(ui().ckb_RE):
	ui().ckb_RE.setCheckState(False)
	
	if widgetValue(ui().ckb_I0):
		#RE(mv(BTA2diag.trans, -85)) 
		Io = np.zeros(len(en_ls))
	else:
		RE(mv(BTA2diag.trans, 0))


	mbs_tb_wdg = ui().parentUi().embedframe_5
	mbs_tb_wdg = mbs_tb_wdg.findChild(QWidget, 'Form')
	mbs_tb_wdg = mbs_tb_wdg.findChild(QGroupBox, 'groupBox') 
	mbs_tb_wdg = mbs_tb_wdg.findChild(MBSTable, 'mbstable')    # this is an handle to the MBS seq. widget


#	print('child list', mbs_tb_wdg.children()[0].children()[0].children()[1].children())
#	print(mbs_tb_wdg.children())
#	print(type( mbs_tb_wdg.children()[1] ) )
#	print('child list', mbs_tb_wdg.children()[1].objectName())
	#print('child list', mbs_tb_wdg.children()[0].children()[0].objectName())
	#print('child list', mbs_tb_wdg.children()[0].children()[0].children()[1].objectName())
	mbs_tb_wdg = ui().parentUi().embedframe_5.children()[0].children()[0].children()[1]
	#.GroupBox.mbstable	# this is an handle to the MBS seq. widget

#	print('dir: ', "Z:\\\\csv_files\\XPS\\2025\\"+mbs_tb_wdg.dir_le.text().replace('/','\\') )
#	mbs.file_path.set("Z:\\\\image_files\\spectrum_analyzer\\"+mbs_tb_wdg.dir_le.text()).wait()

#	mbs.file_path.set("Z:\\\\csv_files\\XPS\\2025\\"+mbs_tb_wdg.dir_le.text().replace('/','\\') ).wait()

	print(mbs_tb_wdg.file_le.text(), mbs_tb_wdg.dir_le.text())

	mbs.file_prefix.set(mbs_tb_wdg.file_le.text()).wait()
	RE.md["sample_date"] = mbs_tb_wdg.date_le.text()
	RE.md['sample_name']=mbs_tb_wdg.dir_le.text()

	for indx, eng in enumerate(en_ls):
		ui().MBS_scan_LE.setText(str(indx+1)+' of '+tot)
		if indx > 0:      # assume the mono is already in the lowest photon energy
			print('pol=', pol_ls[indx])
			RE(Eph.move_to(float(eng), grating= grt, EPU=epu, LP= pol_ls[indx], branch=branch , shutter='open'))

#		if widgetValue(ui().ckb_M3_adj) and indx > 0 :      #adjust M3 on the go (if it is not the starting energy) 
		if widgetValue(ui().ckb_M3_adj) :      #adjust M3 on the go 
			M3_Ry = eval('M3.Ry.user_readback.get()')
			RE(mv(M3.Ry, M3_Ry - 2*0.00005))
			RE(mv(M3.Ry, M3_Ry +2*0.00005))   # to establish that it always start arriving from the positive direction, for backslash issues
			sleep(3)
			M3_Ry_0 = eval('M3.Ry.user_readback.get()')
			print(eval('M3.Ry.user_readback.get()'))
			M3_Ry = M3_Ry_0
			RE(mv(M4AUdiag.trans, -6))
			Au0_values, Au0_avg, Au0_sigma,  =np.array([0.0]*10), 0.0, 0.0
			Au1_values, Au1_avg, Au1_sigma,  =np.array([0.0]*10), 0.0, 0.0

			for i in range(10):
				time.sleep(0.1)
				Au0_values[i] = qem08.current1.mean_value.get()
				#print(qem08.current1.mean_value.get())
				#print(Au0_values[i])
			Au0_avg, Au0_std = np.mean(Au0_values), np.std(Au0_values)
			
			
			
			M3_Ry = M3_Ry + 0.00005
			RE(mv(M3.Ry, M3_Ry))
			sleep(3)
			print(eval('M3.Ry.user_readback.get()'))
			for i in range(10):
				time.sleep(0.1)
				Au1_values[i] = qem08.current1.mean_value.get()
			Au1_avg, Au1_std = np.mean(Au1_values), np.std(Au1_values)
			

			dir_found = False
			max_found = False
			insignificant_step = 0

			while not dir_found:       #loop to determine direction
				print( 'av0 = {}, av1 = {}, diff = {}'.format(Au0_avg, Au1_avg, (Au1_avg-Au0_avg) ))
				if abs((Au1_avg-Au0_avg)) > (Au0_std+Au1_std)/2:   # significant change 
					if (Au1_avg-Au0_avg) >0 :     
						direction = +1.0
					else :
						direction = -1.0
					dir_found = True
					
				else:  # not significant step 
					M3_Ry = M3_Ry + 0.00005
					sleep(3)
					RE(mv(M3.Ry, M3_Ry))
					print(eval('M3.Ry.user_readback.get()'))
					Au0_avg, Au0_std = Au1_avg, Au1_std 
					for i in range(10):
						time.sleep(0.1)
						Au1_values[i] = qem08.current1.mean_value.get()
					Au1_avg, Au1_std = np.mean(Au1_values), np.std(Au1_values)
					insignificant_step +=1
					print('one more insignificant step during direction search, tot: ', insignificant_step)

				if insignificant_step == 5:
					RE(mv(M3.Ry, M3_Ry_0))
					print('could not adjust M3')
					dir_found = True
					max_found = True
					#return
			

			print('determined direction: ', direction)				
			insignificant_step = 0

			for i in range(10):
				time.sleep(0.1)
				Au1_values[i] = qem08.current1.mean_value.get()
			Au0_avg, Au0_std = np.mean(Au1_values), np.std(Au1_values)
			
				# now that dir of increase is known, move one time in that direction
			if direction == 1.0:
				M3_Ry = M3_Ry +direction* 0.00005
			else:
				M3_Ry = M3_Ry +2*direction* 0.00005
			RE(mv(M3.Ry, M3_Ry))
			sleep(3)
			print(eval('M3.Ry.user_readback.get()'))

#			Au0_avg, Au0_std = Au1_avg, Au1_std 
			for i in range(10):
				time.sleep(0.1)
				Au1_values[i] = qem08.current1.mean_value.get()
			Au1_avg, Au1_std = np.mean(Au1_values), np.std(Au1_values)
			print("extra step in the direction of increased signal")
			print( 'av1 = {}, av0 = {}, diff = {}'.format(Au1_avg, Au0_avg, (Au1_avg-Au0_avg) ))
			while not max_found:
				if abs((Au1_avg-Au0_avg)) > (Au0_std+Au1_std)/2:      # significant change  
					print( 'av1 = {}, av0 = {}, diff = {}'.format(Au1_avg, Au0_avg, (Au1_avg-Au0_avg) ))			
					if (Au1_avg-Au0_avg) >0:  #same direction
						print('significant, positive step, continue')
						M3_Ry = M3_Ry +direction* 0.00005
						RE(mv(M3.Ry, M3_Ry))
						sleep(3)
						Au0_avg, Au0_std = Au1_avg, Au1_std 
						for i in range(10):
							time.sleep(0.1)
							Au1_values[i] = qem08.current1.mean_value.get()
						Au1_avg, Au1_std = np.mean(Au1_values), np.std(Au1_values)
					else:
						print('significant, negative step, reached max: step back')
						max_found = True
						M3_Ry = M3_Ry -direction* 0.00005
						RE(mv(M3.Ry, M3_Ry))
				else:
					print('insignificant, do nothing, go out')
					max_found = True
		
			# after while-loop, pull out the diagnostic			
			RE(mv(M4AUdiag.trans, 2))




				#log file
			with open('/home2/xf21id1/hv_log.csv', mode='a') as f:
				csv_writer = csv.writer(f, delimiter=',')
				csv_writer.writerow([ eng, eval('{:.2f}'.format(PGM.Energy.position)), eval('{:.2f}'.format(PGM.Focus_Const.position)),  Au1_avg, eval('{:.5f}'.format(M3.Ry.user_readback.get()))])			

		if widgetValue(ui().ckb_hv_ls) and indx >0:			
			c = center +float(en_ls[indx])-float(en_ls[0])
		else:
			c = center + indx*en_step
		mbs.center_ke.put(c)

		if widgetValue(ui().ckb_3D):
			RE(scan([mbs], mbs.deflX, Ai,Af, 1+int( abs(Af-Ai) /Ad ) ))
		else:	  
			RE(count([mbs], num=1))
		

		if widgetValue(ui().ckb_I0):
#			Io[indx] = eval( '{:.2e}'.format(qem07.current4.mean_value.get())) 
#			Io[indx] = eval( '{:.2e}'.format(xqem01.current3.mean_value.get())) 
			Io[indx] = Au1_avg

		








if widgetValue(ui().ckb_SH):
	shutter_FE.close_cmd.put(1)
	EpicsSignal('XF:21IDD-VA{ANAL:1A-GV:EA1_1}Cmd:Cls-Cmd').put(1)

		

if widgetValue(ui().T_OFF_CB):
	EpicsSignal('XF:21IDD-ES{PS:Heat3}STBY_SP.PROC').put(1)
#	lakeshore2.ctrl1.range_sel.put(0)


if widgetValue(ui().ckb_I0):
	#print('saving Io file')
	directory = ui().dir_LE.text()
	filename = directory+'/'+ ui().csv_LE.text()
	with open(filename+'.csv', mode='w') as f:
		csv_writer = csv.writer(f, delimiter=',')
		csv_writer.writerow([ 'hv',  'Io' ])
		for indx, eng in enumerate(en_ls):
			csv_writer.writerow([ eng, Io[indx]])
	print("saved: ", filename)	

# remove the Au mech in case it was in for measuring the Io
if widgetValue(ui().ckb_I0):
	RE(mv(BTA2diag.trans, 0)) 
		