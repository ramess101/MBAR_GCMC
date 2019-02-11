# -*- coding: utf-8 -*-
"""
Converts TraPPE to Potoff
"""

from __future__ import division
import numpy as np
from MBAR_rerunModel import MBAR_GCMC
import os.path
import glob

nhists_max = 7

compound_list = ['CYC6']
directory_dic = {'CYC6':'CYC6/'}
experimental_dic = {'CYC6':'REFPROP_values'}

trim_data = False
remove_low_high_Tsat = False
bootstrap = False
nBoots = 30
thin_Tsat = 1

#lam_range = [12,14,16,18,20]
#
#lam = 12
#isig = 12
#
#eps_range = np.linspace(40,60,41)
#neps = len(eps_range)

### For referenceFF = 'TraPPE'
eps_low_dic = {12:45,14:55,16:65,18:70,20:80}
eps_high_dic = {12:60,14:70,16:80,18:85,20:95}
lam_range = [12,14,16,18,20]
neps = 31
referenceFF = 'TraPPE'

### For referenceFF = 'lam16'
eps_low_dic = {16:65}
eps_high_dic = {16:75}
lam_range = [16]
neps = 21
referenceFF = 'lam16'

### For referenceFF = 'lam14'
eps_low_dic = {14:55}
eps_high_dic = {14:70}
lam_range = [14]
neps = 31
referenceFF = 'lam14'

### For referenceFF = 'lam18'
eps_low_dic = {18:72}
eps_high_dic = {18:82}
lam_range = [18]
neps = 21
referenceFF = 'lam18'

### For referenceFF = 'lam20'
#eps_low_dic = {20:80}
#eps_high_dic = {20:90}
#lam_range = [20]
#neps = 21
#referenceFF = 'lam20'

### For referenceFF = 'lam16' refined grid
#eps_low_dic = {16:69}
#eps_high_dic = {16:71}
#lam_range = [16]
#neps = 21
#referenceFF = 'lam16'

compound = 'CYC6'

isig_low = int(np.loadtxt('isig_low'))
nsig = 1

for lam in lam_range:

    eps_range = np.linspace(eps_low_dic[lam],eps_high_dic[lam],neps)
    neps = len(eps_range)
    
    for isig in np.arange(isig_low,isig_low+nsig):

        print('refFF: '+referenceFF+', lam = '+str(lam)+', isig: '+str(isig))

        output_root_path = '/home/ram9/'+compound+'/GOMC/GCMC/'+referenceFF+'_basis'+'/MBAR_results/'

        if not os.path.exists(output_root_path):
                        
            os.mkdir(output_root_path)
            
        lam_sig_processed = False

        for file_name in glob.glob(output_root_path+'lam'+str(lam)+'sig'+str(isig)+'eps*'):

            lam_sig_processed = True
            print(file_name)
    
        #if not os.path.exists(output_root_path+'lam'+str(lam)+'sig'+str(isig)+'eps'+str(ieps+1)+'/MBAR_VLE.txt'): 
                               
        if lam_sig_processed:

           print('lam = '+str(lam)+' sig = '+str(isig)+' already processed')

        else:
            
            filepaths = []
                            
            root_path = '/home/ram9/'+compound+'/GOMC/GCMC/'+referenceFF+'_basis'+'/lam'+str(lam)+'recompute_sig'+str(isig)+'/'
                                                                           
            for iT in np.arange(0,nhists_max+1):
                            
                hist_name='/his'+str(iT)+'a.dat'
                    
                if os.path.exists(root_path+hist_name):
                    
                    filepaths.append(root_path+hist_name)
                                
            VLE_RP = np.loadtxt('/home/ram9/'+compound+'/REFPROP_values/VLE_'+compound+'.txt',skiprows=1)
            
            Tsat_RP = VLE_RP[::thin_Tsat,0] #[K]
            rhol_RP = VLE_RP[::thin_Tsat,1] #[kg/m3]
            rhov_RP = VLE_RP[::thin_Tsat,2] #[kg/m3]
            Psat_RP = VLE_RP[::thin_Tsat,3] #[bar]
            DeltaHv_RP = VLE_RP[::thin_Tsat,4] #[kJ/mol]
    
            if remove_low_high_Tsat:  #Remove the low T and high T ends for more stable results
                Tsat_RP = Tsat_RP[2:-2]
                rhol_RP = rhol_RP[2:-2]
                rhov_RP = rhov_RP[2:-2]
                Psat_RP = Psat_RP[2:-2]
                DeltaHv_RP = DeltaHv_RP[2:-2]
            
            Mw = np.loadtxt('/home/ram9/'+compound+'/REFPROP_values/MW_'+compound+'.txt')
            
            print('Performing MBAR-GCMC analysis for '+compound)
            
            MBAR_GCMC_trial = MBAR_GCMC(root_path,filepaths,Mw,trim_data=trim_data,compare_literature=True)
            MBAR_GCMC_trial.solve_VLE(Tsat_RP)
            rhol_ref_model = MBAR_GCMC_trial.rholiq
            rhov_ref_model = MBAR_GCMC_trial.rhovap
            Psat_ref_model = MBAR_GCMC_trial.Psat
            DeltaHv_ref_model = MBAR_GCMC_trial.DeltaHv
            Neffliq_model = MBAR_GCMC_trial.Neffliq
            Neffvap_model = MBAR_GCMC_trial.Neffvap
            Ntotvap_model = len(MBAR_GCMC_trial.Nmol_flat[MBAR_GCMC_trial.Nmol_flat<=MBAR_GCMC_trial.Ncut])
            Ntotliq_model = len(MBAR_GCMC_trial.Nmol_flat[MBAR_GCMC_trial.Nmol_flat>MBAR_GCMC_trial.Ncut])
            
            eps_sig_lam = np.loadtxt(root_path+'eps_sig_lam.txt',skiprows=1)
                
            eps_rr = eps_sig_lam[0]
            sig_rr = eps_sig_lam[1]
            lam_rr = eps_sig_lam[2]
            
            eps_old = eps_rr
            
            for ieps, eps_new in enumerate(eps_range):
                
                eps_scaled = eps_new / eps_old
    #        
                MBAR_GCMC_trial.solve_VLE(Tsat_RP,eps_scaled)
                rhol_rr_model = MBAR_GCMC_trial.rholiq
                rhov_rr_model = MBAR_GCMC_trial.rhovap
                Psat_rr_model = MBAR_GCMC_trial.Psat
                DeltaHv_rr_model = MBAR_GCMC_trial.DeltaHv
                Neffliq_rr_model = MBAR_GCMC_trial.Neffliq
                Neffvap_rr_model = MBAR_GCMC_trial.Neffvap
                Ntotvap_rr_model = len(MBAR_GCMC_trial.Nmol_flat[MBAR_GCMC_trial.Nmol_flat<=MBAR_GCMC_trial.Ncut])
                Ntotliq_rr_model = len(MBAR_GCMC_trial.Nmol_flat[MBAR_GCMC_trial.Nmol_flat>MBAR_GCMC_trial.Ncut])
                                    
                ### Compile the rerun model results
                
                if not os.path.exists(output_root_path+'lam'+str(lam)+'sig'+str(isig)+'eps'+str(ieps+1)+'/MBAR_VLE.txt'):
                    
                    os.mkdir(output_root_path+'lam'+str(lam)+'sig'+str(isig)+'eps'+str(ieps+1))
                    
                output_path = output_root_path+'lam'+str(lam)+'sig'+str(isig)+'eps'+str(ieps+1)+'/'
                
                out_file =open(output_path+'MBAR_VLE.txt','w')
                out_file.write('T (K) rhol (kg/m3) rhov (kg/m3) P (bar) DeltaHv (kJ/mol)'+'\n')
                for Tsat, rhol, rhov, Psat, DeltaHv in zip(Tsat_RP,rhol_rr_model,rhov_rr_model,Psat_rr_model,DeltaHv_rr_model):
                    out_file.write(str(Tsat)+'\t'+str(rhol)+'\t'+str(rhov)+'\t'+str(Psat)+'\t'+str(DeltaHv)+'\n')
                out_file.close()  
                
                ### Compile the rerun model number of effective samples
                out_file =open(output_path+'MBAR_Neff.txt','w')
                out_file.write('T (K) Neffliq Neffvap Percentliq Percentvap Ratioliq Ratiovap'+'\n')
                for Tsat, Neffliq_rr, Neffvap_rr in zip(Tsat_RP,Neffliq_rr_model,Neffvap_rr_model):
                    out_file.write(str(Tsat)+'\t'+str(Neffliq_rr)+'\t'+str(Neffvap_rr)+'\n')
                out_file.close()  
                            
                f = open(output_path+'eps_sig_lam.txt','w')
                f.write('eps'+'\t'+'sig'+'\t'+'lam'+'\n')
                f.write(str(eps_new)+'\t'+str(sig_rr)+'\t'+str(lam_rr))
                f.close()