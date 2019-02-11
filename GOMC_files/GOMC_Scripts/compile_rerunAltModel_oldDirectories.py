'''
Compiles the histograms for model and rerun model
'''

from __future__ import division
import numpy as np 
import os, sys, argparse, shutil

compound = 'C6H14'
model = 'Potoff'

root_path = '/home/ram9/'+compound+'/GOMC/GCMC/'+model+'/'

Temps = np.array([330,360,390,420,430,450,470,480,510])                                     
                                     
NTemps = len(Temps)
NHists = 2000

for iTemp, Temp_i in enumerate(Temps):
    
    Temp_path = root_path+str(Temp_i)+'/'

    f = open(root_path+'/histfiles/his'+str(iTemp+1)+'a.dat','w')

    try:

        Hist_data = np.loadtxt(Temp_path+'his_rr1_model',skiprows=1)
        NSnapshots = len(Hist_data)
        Hist_header = np.genfromtxt(Temp_path+'his_rr1_model',skip_footer=NSnapshots)

        for header_i in Hist_header:

            f.write(str(header_i)+'\t')

        f.write('\n')

    except:

        print('No file for T= '+str(Temp_i))
            
    for iHist in np.arange(1,NHists+1):

        try:
        
            Hist_model_i = np.loadtxt(Temp_path+'his_rr'+str(iHist)+'_model',skiprows=1)
            Hist_rerun_model_i = np.loadtxt(Temp_path+'his_rr'+str(iHist)+'_rerun_model',skiprows=1)
        
            N_model = Hist_model_i[:,0]
            u_model = Hist_model_i[:,1]
            N_rerun_model = Hist_rerun_model_i[1:,0]
            u_rerun_model = Hist_rerun_model_i[1:,1]
        
            assert len(u_model) == len(u_rerun_model), 'Number of frames are different, model has '+str(len(u_model))+' frames and rerun has '+str(len(u_rerun_model))+' frames.'   
            assert N_model[0] == N_rerun_model[0], 'First frame N are different, model has '+str(N_model[0])+' and rerun has '+str(N_rerun_model[0])    
            assert N_model[-1] == N_rerun_model[-1], 'Last frame N are different, model has '+str(N_model[-1])+' and rerun has '+str(N_rerun_model[-1])

            for N_j, u_model_j, u_rr_model_j in zip(N_model,u_model,u_rerun_model):

                f.write(str(int(N_j))+'\t'+str(u_model_j)+'\t'+str(u_rr_model_j)+'\n')

        except:

            pass

    f.close()