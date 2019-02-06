# -*- coding: utf-8 -*-
"""
Converts TraPPE to Potoff
"""

from __future__ import division
import numpy as np
from MBAR_rerunModel import MBAR_GCMC
import os.path
import matplotlib.pyplot as plt

nhists_max = 7

compound_list = ['CYC6']
directory_dic = {'CYC6':'CYC6/'}
experimental_dic = {'CYC6':'REFPROP_values'}
reference_list = ['TraPPE']

referenceFF_directory_dic = {'TraPPE':'TraPPE/','lam16':'lam16/','lam16sig20eps1':'lam16sig20eps1/'}

trim_data = True
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

eps_low_dic = {12:45,14:55,16:65,18:70,20:80}
eps_high_dic = {12:60,14:70,16:80,18:85,20:95}

lam_range = [12,14,16,18,20]
#lam_range = [16]
neps = 31

for lam in lam_range:

    eps_range = np.linspace(eps_low_dic[lam],eps_high_dic[lam],neps)
    neps = len(eps_range)
    
    for isig in np.arange(1,22):
        
        fig, axarr = plt.subplots(nrows=3,ncols=1,figsize=[6,18])
        
        for referenceFF in reference_list: 
                
            print('refFF: '+referenceFF+', rrFF: '+str(isig))
               
            for compound in compound_list:
                        
                filepaths = []
                
                ### The full approach
                root_path = 'H:/GCMC_basis_functions/'+referenceFF+'/'+compound+'/lam'+str(lam)+'recompute_sig'+str(isig)+'/'
                ### For just the ref and rr basis function
#                root_path = 'H:/GCMC_basis_functions/'+referenceFF+'/'+compound+'/histfiles/'
                
                for iT in np.arange(0,nhists_max+1):
                                
                    hist_name='/his'+str(iT)+'a.dat'
                        
                    if os.path.exists(root_path+hist_name):
                        
                        filepaths.append(root_path+hist_name)
                                    
                VLE_RP = np.loadtxt('H:/MBAR_GCMC/Experimental_values/REFPROP_values/VLE_'+compound+'.txt',skiprows=1)
                
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
                
                Mw = np.loadtxt('H:/MBAR_GCMC/Experimental_values/MW_'+compound+'.txt')
                
                print('Performing MBAR-GCMC analysis for '+compound)
                
                MBAR_GCMC_trial = MBAR_GCMC(root_path,filepaths,Mw,trim_data=trim_data,compare_literature=True)
        #        MBAR_GCMC_trial.plot_histograms()
        #        MBAR_GCMC_trial.plot_2dhistograms()
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
                    
                    color_scaled = (1.*ieps/neps,0,1.-ieps/neps,1)
                    
    #                print(ieps/neps)
                    
                    neps_minus1 = neps-1.
                    
    #                print(ieps/neps_minus1)
                    
                    if ieps < neps_minus1/2.:
                        
                        color_scaled = (0,2.*ieps/neps_minus1,1.-2.*ieps/neps_minus1,1)
    #                    print('case0')
                        
                    else:
    #                    print('case1')
                        color_scaled = (2.*(ieps/neps_minus1-0.5),1.-2.*(ieps/neps_minus1-0.5),0,1)
                    
                    axarr[0].plot(rhol_rr_model,Tsat_RP,color=color_scaled,linestyle='',marker='o',mfc='None',label='Rerun')
                    axarr[0].plot(rhov_rr_model,Tsat_RP,color=color_scaled,linestyle='',marker='o',mfc='None',label='Rerun')
        
                    axarr[1].plot(1000./Tsat_RP,np.log10(Psat_rr_model),color=color_scaled,linestyle='',marker='o',mfc='None',label='Rerun')
        
                    axarr[2].plot(Tsat_RP,DeltaHv_rr_model,color=color_scaled,linestyle='',marker='o',mfc='None',label='Rerun')
                            
                ### Compile the reference model results
        #                out_file =open('H:/MBAR_GCMC/refFF_to_rrFF/'+referenceFF+'/'+compound+'.txt','w')
        #                out_file.write('T (K) rhol (kg/m3) rhov (kg/m3) P (bar) DeltaHv (kJ/mol)'+'\n')
        #                for Tsat, rhol, rhov, Psat, DeltaHv in zip(Tsat_RP,rhol_ref_model,rhov_ref_model,Psat_ref_model,DeltaHv_ref_model):
        #                    out_file.write(str(Tsat)+'\t'+str(rhol)+'\t'+str(rhov)+'\t'+str(Psat)+'\t'+str(DeltaHv)+'\n')
        #                out_file.close()  
        #                
                    ### Compile the rerun model results
                    
                    if not os.path.exists(root_path+'eps'+str(ieps+1)+'/MBAR_VLE.txt'):
                        
                        os.mkdir(root_path+'eps'+str(ieps+1))
                        
                    output_path = root_path+'eps'+str(ieps+1)+'/'
                    
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
                    
        axarr[0].plot(rhol_ref_model,Tsat_RP,'ks',mfc='None',label='Ref.')
        axarr[0].plot(rhov_ref_model,Tsat_RP,'ks',mfc='None',label='Ref.')
        
        axarr[0].plot(rhol_RP,Tsat_RP,'k-',label='REFPROP')
        axarr[0].plot(rhov_RP,Tsat_RP,'k-',label='REFPROP')
        
        axarr[1].plot(1000./Tsat_RP,np.log10(Psat_ref_model),'ks',mfc='None',label='Ref.')
        axarr[1].plot(1000./Tsat_RP,np.log10(Psat_RP),'k-',label='REFPROP')
                    
        axarr[2].plot(Tsat_RP,DeltaHv_ref_model,'ks',mfc='None',label='Ref.')
        axarr[2].plot(Tsat_RP,DeltaHv_RP,'k-',label='REFPROP')
        
        axarr[0].set_xlabel(r'$\rho$ (kg/m$^3$)')
        axarr[0].set_ylabel(r'$T$ (K)')
        
        axarr[1].set_ylabel(r'$\log_{10}(P^{\rm sat}_{\rm vap}$/bar)')
        axarr[1].set_xlabel(r'$1000/T$ (K)')
        
        axarr[2].set_ylabel(r'$\Delta H_{\rm v}$ (kJ/mol)')
        axarr[2].set_xlabel(r'$T$ (K)')
        
        plt.tight_layout()
        
        fig.savefig(root_path+'eps_scan.pdf')