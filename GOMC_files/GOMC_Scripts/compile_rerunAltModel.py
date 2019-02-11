'''
Compiles the histograms for model and rerun model
'''

from __future__ import division
import numpy as np 
import os, sys, argparse, shutil

compound = '22DMButane'
model = 'PotofftoPotoff_S'

root_path = '/home/ram9/'+compound+'/GOMC/GCMC/'+model+'/'

NStates = 9                                    
NHists = 2000

for iState in range(NStates):
    
    State_path = root_path+'State'+str(iState)+'/'

    try:

        f = open(root_path+'histfiles/his'+str(iState+1)+'a.dat','w')

    except:

        os.mkdir(root_path+'histfiles')
        f = open(root_path+'histfiles/his'+str(iState+1)+'a.dat','w')

    try:

        Hist_data = np.loadtxt(State_path+'his_rr1_model',skiprows=1)
        NSnapshots = len(Hist_data)
        Hist_header = np.genfromtxt(State_path+'his_rr1_model',skip_footer=NSnapshots)

        for header_i in Hist_header:

            f.write(str(header_i)+'\t')

        f.write('\n')

    except:

        print('No file for State'+str(iState))
            
    for iHist in np.arange(1,NHists+1):

        try:
        
            Hist_model_i = np.loadtxt(State_path+'his_rr'+str(iHist)+'_model',skiprows=1)
            Hist_rerun_model_i = np.loadtxt(State_path+'his_rr'+str(iHist)+'_rerun_model',skiprows=1)
        
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
