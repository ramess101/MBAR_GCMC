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

compound = 'CYC6'
directory_dic = {'CYC6':'CYC6/'}
experimental_dic = {'CYC6':'REFPROP_values'}
referenceFF = 'TraPPE'

referenceFF_directory_dic = {'TraPPE':'TraPPE/'}

reprocess = True
remove_low_high_Tsat = True
use_Neff_cutoff = False
include_liquid = True
include_vapor = True
remove_outliers = True
thin_Tsat = 1

Score_w8 = np.array([0.6135,0.0123,0.2455,0.0245,0.0613,0.0061,0.0245,0.0123])
#Score_w8 = np.array([0.6135,0,0,0,0.0613,0,0,0])

if not include_liquid:

    ### Include rhovap and Psat
#    Score_w8[1] += Score_w8[3]
#    Score_w8[2] += Score_w8[0]
#    Score_w8[5] += Score_w8[7]
#    Score_w8[6] += Score_w8[4]
#            
#    Score_w8[0] = 0
#    Score_w8[3] = 0
#    Score_w8[4] = 0
#    Score_w8[7] = 0

    ### Include rhovap Psat and DeltaHv
    Score_w8[2] += Score_w8[0]
    Score_w8[6] += Score_w8[4]
            
    Score_w8[0] = 0
    Score_w8[4] = 0
            
if not include_vapor:

    ### Both rhol and DeltaHv
#    Score_w8[3] += Score_w8[1]
#    Score_w8[0] += Score_w8[2]
#    Score_w8[7] += Score_w8[5]
#    Score_w8[4] += Score_w8[6]
#            
#    Score_w8[1] = 0
#    Score_w8[2] = 0
#    Score_w8[5] = 0
#    Score_w8[6] = 0

    ### Just rhol                 
    Score_w8[0] += Score_w8[2] + Score_w8[1] + Score_w8[3]
    Score_w8[4] += Score_w8[6] + Score_w8[5] + Score_w8[7]
            
    Score_w8[1] = 0
    Score_w8[2] = 0
    Score_w8[3] = 0     
    Score_w8[5] = 0
    Score_w8[6] = 0
    Score_w8[7] = 0 
            
    ### Just DeltaHv
#    Score_w8[3] += Score_w8[2] + Score_w8[1] + Score_w8[0]
#    Score_w8[7] += Score_w8[6] + Score_w8[5] + Score_w8[4]
#    
#    Score_w8[0] = 0         
#    Score_w8[1] = 0
#    Score_w8[2] = 0
#    Score_w8[4] = 0     
#    Score_w8[5] = 0
#    Score_w8[6] = 0

compute_MAPD = lambda yhat,yset: np.mean(np.abs((yhat - yset)/yset*100.))
compute_APD = lambda yhat,ydata: np.abs((yhat - ydata)/ydata*100.)

VLE_RP = np.loadtxt('H:/MBAR_GCMC/Experimental_values/REFPROP_values/VLE_'+compound+'.txt',skiprows=1)

Tsat_RP = VLE_RP[::thin_Tsat,0] #[K]
rhol_RP = VLE_RP[::thin_Tsat,1] #[kg/m3]
rhov_RP = VLE_RP[::thin_Tsat,2] #[kg/m3]
Psat_RP = VLE_RP[::thin_Tsat,3] #[bar]
DeltaHv_RP = VLE_RP[::thin_Tsat,4] #[kJ/mol]

if remove_low_high_Tsat:  #Remove the low T and high T ends for more stable results
    Tsat_RP = Tsat_RP[2:-6]
    rhol_RP = rhol_RP[2:-6]
    rhov_RP = rhov_RP[2:-6]
    Psat_RP = Psat_RP[2:-6]
    DeltaHv_RP = DeltaHv_RP[2:-6]

Neff_cutoff = 50

def Scoring_function(Temp_VLE,rholiq,rhovap,Psat,DeltaHv,rhol_RP,rhov_RP,Psat_RP,DeltaHv_RP,Neff_liq,use_Neff_cutoff=False,include_liquid=True,include_vapor=True):
                           
    if use_Neff_cutoff:
        
        rholiq = rholiq[Neff_liq>Neff_cutoff]
        rhol_RP = rhol_RP[Neff_liq>Neff_cutoff]
        DeltaHv = DeltaHv[Neff_liq>Neff_cutoff]
        DeltaHv_RP = DeltaHv_RP[Neff_liq>Neff_cutoff]

    if remove_outliers: # Sometimes when MBAR fails we get negative values
        len_original = len(Temp_VLE)
        Temp_VLE = Temp_VLE[DeltaHv > 0]        
        rholiq = rholiq[DeltaHv > 0]
        rhol_RP = rhol_RP[DeltaHv > 0]
        rhovap = rhovap[DeltaHv > 0]
        rhov_RP = rhov_RP[DeltaHv > 0]
        Psat = Psat[DeltaHv > 0]
        Psat_RP = Psat_RP[DeltaHv > 0]
        DeltaHv_RP = DeltaHv_RP[DeltaHv > 0]
        DeltaHv = DeltaHv[DeltaHv > 0]

        Temp_VLE = Temp_VLE[rhovap > 0]        
        rholiq = rholiq[rhovap > 0]
        rhol_RP = rhol_RP[rhovap > 0]
        Psat = Psat[rhovap > 0]
        Psat_RP = Psat_RP[rhovap > 0]
        DeltaHv_RP = DeltaHv_RP[rhovap > 0]
        DeltaHv = DeltaHv[rhovap > 0]
        rhov_RP = rhov_RP[rhovap > 0]
        rhovap = rhovap[rhovap > 0]
        
        Temp_VLE = Temp_VLE[Psat > 0]
        rholiq = rholiq[Psat > 0]
        rhol_RP = rhol_RP[Psat > 0]
        rhovap = rhovap[Psat > 0]
        rhov_RP = rhov_RP[Psat > 0]
        DeltaHv_RP = DeltaHv_RP[Psat > 0]
        DeltaHv = DeltaHv[Psat > 0]
        Psat_RP = Psat_RP[Psat > 0]
        Psat = Psat[Psat > 0]
        
        #if len(Temp_VLE) < len_original: print('Removed outliers that were negative')

#    APD_rhol = compute_APD(rholiq,rhol_RP)
#    APD_DeltaHv = compute_APD(DeltaHv,DeltaHv_RP)
#    APD_rhov = compute_APD(rhovap,rhov_RP)
#    APD_Psat = compute_APD(Psat,Psat_RP)    
#    
#    MAPD_rhol = compute_MAPD(rholiq,rhol_RP)
#    MAPD_DeltaHv = compute_MAPD(DeltaHv,DeltaHv_RP)
#    MAPD_rhov = compute_MAPD(rhovap,rhov_RP)
#    MAPD_Psat = compute_MAPD(Psat,Psat_RP)
#

#    for iAPD, APD_rhol_i in enumerate(APD_rhol):
#        
#        if APD_rhol_i > 1.1*MAPD_rhol:
            
            
            

    nTemp_VLE = len(Temp_VLE)
    nNeff_cutoff = len(rholiq)  
      
    if nNeff_cutoff > 0:
    
        MAPD_rhol = compute_MAPD(rholiq,rhol_RP)
        MAPD_DeltaHv = compute_MAPD(DeltaHv,DeltaHv_RP)
        
        APD_rhol = compute_APD(rholiq,rhol_RP)
        APD_DeltaHv = compute_APD(DeltaHv,DeltaHv_RP)
                
    else:
        
        MAPD_rhol = 100
        MAPD_DeltaHv = 100
        APD_rhol = 100
        APD_DeltaHv = 100

    MAPD_rhov = compute_MAPD(rhovap,rhov_RP)
    MAPD_Psat = compute_MAPD(Psat,Psat_RP)
    
    APD_rhov = compute_APD(rhovap,rhov_RP)
    APD_Psat = compute_APD(Psat,Psat_RP)    

### An attempt to remove outliers to smooth the heat maps

    if remove_outliers:    

        outliers = np.zeros(len(Temp_VLE))
        iAPD = 0
        outlier_factor = 6.

        for Neff_i, APD_rhol_i, APD_rhov_i, APD_Psat_i, APD_DeltaHv_i in zip(Neff_liq, APD_rhol,APD_rhov,APD_Psat,APD_DeltaHv):
            
#            if Neff_i < Neff_cutoff and (APD_rhol_i > outlier_factor*MAPD_rhol or APD_rhov_i > outlier_factor*MAPD_rhov or APD_Psat_i > outlier_factor*MAPD_Psat or APD_DeltaHv_i > outlier_factor*MAPD_DeltaHv): outliers[iAPD] = 1.
            if APD_rhol_i > outlier_factor*MAPD_rhol or APD_rhov_i > outlier_factor*MAPD_rhov or APD_Psat_i > outlier_factor*MAPD_Psat or APD_DeltaHv_i > outlier_factor*MAPD_DeltaHv: outliers[iAPD] = 1.
            #if Neff_i < np.mean(Neff_liq) / outlier_factor: outliers[iAPD] = 1.
            iAPD += 1

        if np.sum(outliers > 0):
    
#            print('Removed outliers that deviated by a factor of '+str(outlier_factor))
            
            Temp_VLE = Temp_VLE[outliers < 1]
            rholiq = rholiq[outliers < 1]
            rhol_RP = rhol_RP[outliers < 1]
            rhovap = rhovap[outliers < 1]
            rhov_RP = rhov_RP[outliers < 1]
            DeltaHv_RP = DeltaHv_RP[outliers < 1]
            DeltaHv = DeltaHv[outliers < 1]
            Psat_RP = Psat_RP[outliers < 1]
            Psat = Psat[outliers < 1]
            
            nTemp_VLE = len(Temp_VLE)
            nNeff_cutoff = len(rholiq)  
              
            MAPD_rhol = compute_MAPD(rholiq,rhol_RP)
            MAPD_DeltaHv = compute_MAPD(DeltaHv,DeltaHv_RP)
            
            APD_rhol = compute_APD(rholiq,rhol_RP)
            APD_DeltaHv = compute_APD(DeltaHv,DeltaHv_RP)
                
            MAPD_rhov = compute_MAPD(rhovap,rhov_RP)
            MAPD_Psat = compute_MAPD(Psat,Psat_RP)
            
            APD_rhov = compute_APD(rhovap,rhov_RP)
            APD_Psat = compute_APD(Psat,Psat_RP)   
    
    dAPD_rhov = np.zeros(nTemp_VLE-1)
    dAPD_Psat = np.zeros(nTemp_VLE-1)
    
    if nNeff_cutoff > 1:
        
        dAPD_rhol = np.zeros(nNeff_cutoff-1)
        dAPD_DeltaHv = np.zeros(nNeff_cutoff-1)
    
    else:
    
        dAPD_rhol = np.array([100])
        dAPD_DeltaHv = np.array([100])    
    
    for iT in range(nTemp_VLE-1):
        
        dAPD_rhov[iT] = (APD_rhov[iT+1]-APD_rhov[iT])/(Temp_VLE[iT+1]-Temp_VLE[iT])
        dAPD_Psat[iT] = (APD_Psat[iT+1]-APD_Psat[iT])/(Temp_VLE[iT+1]-Temp_VLE[iT])
        
    if nNeff_cutoff > 1:
        
        for iT in range(nNeff_cutoff-1):
            
            dAPD_rhol[iT] = (APD_rhol[iT+1]-APD_rhol[iT])/(Temp_VLE[iT+1]-Temp_VLE[iT])
            dAPD_DeltaHv[iT] = (APD_DeltaHv[iT+1]-APD_DeltaHv[iT])/(Temp_VLE[iT+1]-Temp_VLE[iT])    
    
    MdAPD_rhol = np.mean(dAPD_rhol) 
    MdAPD_rhov = np.mean(dAPD_rhov)   
    MdAPD_Psat = np.mean(dAPD_Psat)               
    MdAPD_DeltaHv = np.mean(dAPD_DeltaHv)

    Score = Score_w8[0]*MAPD_rhol + Score_w8[1]*MAPD_rhov + Score_w8[2]*MAPD_Psat + Score_w8[3]*MAPD_DeltaHv
    Score += Score_w8[4]*MdAPD_rhol + Score_w8[5]*MdAPD_rhov + Score_w8[6]*MdAPD_Psat + Score_w8[7]*MdAPD_DeltaHv
                
    return Score

nsig = 21
neps = 31

lam_range = [12]

if reprocess:

    Score_matrix_all = {}
    eps_matrix_all = {}
    sig_matrix_all = {}
    Neff_matrix_all = {}
    eps_unique_all = {}
    sig_unique_all = {}
    
    for lam in lam_range:
        
        Score_flat = np.zeros(nsig*neps)
        eps_flat = np.zeros(nsig*neps)
        sig_flat = np.zeros(nsig*neps)
        Neff_flat = np.zeros(nsig*neps)
        
        Score_matrix = np.zeros([neps,nsig])
        eps_matrix = np.zeros([neps,nsig])
        sig_matrix = np.zeros([neps,nsig])
        Neff_matrix = np.zeros([neps,nsig])
        
        iFF = 0
        
        print('lam = '+str(lam))
        
        for isig in range(nsig):
            
            root_path = 'H:/GCMC_basis_functions/'+referenceFF+'/'+compound+'/MBAR_results/lam'+str(lam)+'sig'+str(isig+1)
            
            for ieps in range(neps):
                
    #            print('refFF: '+referenceFF+', isig: '+str(isig+1)+', ieps: '+str(ieps+1))
                output_path = root_path+'eps'+str(ieps+1)+'/'
    #        
                VLE_MBAR = np.loadtxt(output_path+'MBAR_VLE.txt',skiprows=1)
                Tsat_MBAR = VLE_MBAR[::thin_Tsat,0]
                rhol_MBAR = VLE_MBAR[::thin_Tsat,1]
                rhov_MBAR = VLE_MBAR[::thin_Tsat,2]
                Psat_MBAR = VLE_MBAR[::thin_Tsat,3]
                DeltaHv_MBAR = VLE_MBAR[::thin_Tsat,4]
                
                if remove_low_high_Tsat:  #Remove the low T and high T ends for more stable results
                    Tsat_MBAR = Tsat_MBAR[2:-6]
                    rhol_MBAR = rhol_MBAR[2:-6]
                    rhov_MBAR = rhov_MBAR[2:-6]
                    Psat_MBAR = Psat_MBAR[2:-6]
                    DeltaHv_MBAR = DeltaHv_MBAR[2:-6]
                
                Neff_MBAR = np.loadtxt(output_path+'MBAR_Neff.txt',skiprows=1)
                Neff_liq = Neff_MBAR[:,1]
                Neff_vap = Neff_MBAR[:,2]
    
                Score_ieps_isig = Scoring_function(Tsat_MBAR,rhol_MBAR,rhov_MBAR,Psat_MBAR,DeltaHv_MBAR,rhol_RP,rhov_RP,Psat_RP,DeltaHv_RP,Neff_liq,use_Neff_cutoff,include_liquid,include_vapor)
            
                eps_sig_lam = np.loadtxt(output_path+'eps_sig_lam.txt',skiprows=1)
                
                eps_i = eps_sig_lam[0]
                sig_i = eps_sig_lam[1]
                
                Score_flat[iFF] = Score_ieps_isig
                eps_flat[iFF] = eps_i
                sig_flat[iFF] = sig_i
                Neff_flat[iFF] = np.mean(Neff_liq)
                        
                Score_matrix[ieps,isig] = Score_ieps_isig
                eps_matrix[ieps,isig] = eps_i
                sig_matrix[ieps,isig] = sig_i
                Neff_matrix[ieps,isig] = np.mean(Neff_liq)
    
                iFF +=1
    
        eps_unique = np.unique(eps_flat)
        sig_unique = np.unique(sig_flat)
                
        Score_matrix_all[str(lam)] = Score_matrix.copy()
        eps_unique_all[str(lam)] = eps_unique.copy()
        sig_unique_all[str(lam)] = sig_unique.copy()
        Neff_matrix_all[str(lam)] = Neff_matrix.copy()
        eps_matrix_all[str(lam)] = eps_matrix.copy()
        sig_matrix_all[str(lam)] = sig_matrix.copy()
            
eps_ref = 52.5
sig_ref = 0.391

#Score_ceil = Score_matrix.copy()
#Neff_ceil = Neff_matrix.copy()
#
#Score_ceil[Score_ceil > 20] = 20
#Neff_ceil[Neff_ceil > 100] = 100
#            
##plt.contour(sig_unique/10.,eps_unique,Score_matrix)
##plt.plot(sig_ref,eps_ref,'kx',markersize=10)
#plt.contour(sig_unique/10.,eps_unique,Score_ceil)
#plt.colorbar()
#plt.xlabel(r'$\sigma_{\rm CH_2}$ (nm)')
#plt.ylabel(r'$\epsilon_{\rm CH_2}$ (K)')
#plt.show()
#
#plt.contour(sig_unique/10.,eps_unique,Neff_matrix)
#plt.colorbar()
##plt.plot(sig_ref,eps_ref,'kx',markersize=10)
#plt.xlabel(r'$\sigma_{\rm CH_2}$ (nm)')
#plt.ylabel(r'$\epsilon_{\rm CH_2}$ (K)')
#plt.show()

fig, axarr = plt.subplots(2,2,figsize=[12,12])
plt.tight_layout(pad=3)
fig2, axarr2 = plt.subplots(2,2,figsize=[12,12])
plt.tight_layout(pad=3)
fig3, axarr3 = plt.subplots(1,1,figsize=[6,6])
plt.tight_layout(pad=3)

axarr_dic = {'12':(0,0),'14':(0,1),'16':(1,0),'18':(1,1),'20':(0,0)}

#for lam, Score_matrix_lam, Neff_matrix_lam in zip(Score_matrix_all.keys(),Score_matrix_all,Neff_matrix_all):
#    
#    Score_ceil = Score_matrix_lam.copy()
#    Neff_ceil = Neff_matrix_lam.copy()
    
levels = np.linspace(0,20,100)
levels_Neff = np.linspace(0,500,1000)

levels_Neff = np.linspace(0,5,100)

# Did not fix problem for highest tick
#levels = np.append(levels,[20.001])
#levels_Neff = np.append(levels_Neff,[5.001])

eps_opt_all = {}
ieps_opt_all = {}
sig_opt_all = {}
isig_opt_all = {}
Score_opt_all = {}
lam_exclude = '20'

for ilam, lam in enumerate(Score_matrix_all.keys()):
    if not lam == lam_exclude:
        Score_ceil = Score_matrix_all[lam].copy()
        Neff_ceil = Neff_matrix_all[lam].copy()
    
        Score_ceil[Score_ceil > 20] = 20
        Neff_ceil[Neff_ceil > 10**5] = 10**5 
          
    #    Score_ceil_lower = Score_ceil.copy()
    #    Score_ceil_lower[Score_ceil_lower > 5] = 5
                 
        eps_plot = eps_unique_all[lam]
        sig_plot = sig_unique_all[lam]/10.
    
    #    if ilam == 0: CSref = axarr[axarr_dic[lam]].contourf(sig_plot,eps_plot,Score_ceil,cmap='jet')
                    
        #plt.contour(sig_unique/10.,eps_unique,Score_matrix)
#        if lam == '12': 
#            axarr[axarr_dic[lam]].plot(sig_ref,eps_ref,'kx',markersize=15,markeredgewidth=3)
#            axarr2[axarr_dic[lam]].plot(sig_ref,eps_ref,'kx',markersize=15,markeredgewidth=3)
        if lam == '12': 
            axarr[axarr_dic[lam]].plot(sig_ref,eps_ref,'w^',markersize=15,markeredgecolor='k',markeredgewidth=2)
            axarr2[axarr_dic[lam]].plot(sig_ref,eps_ref,'w^',markersize=15,markeredgecolor='k',markeredgewidth=2)
            
        CS = axarr[axarr_dic[lam]].contourf(sig_plot,eps_plot,Score_ceil,cmap='jet_r',levels=levels)
    #    axarr[axarr_dic[lam]].colorbar()
        axarr[axarr_dic[lam]].set_xlabel(r'$\sigma_{\rm CH_2}$ (nm)')
        axarr[axarr_dic[lam]].set_ylabel(r'$\epsilon_{\rm CH_2}$ (K)')
        axarr[axarr_dic[lam]].set_title(r'$\lambda_{\rm CH_2} = $'+lam,y=1.02)
        
        sig_opt = sig_plot[np.argwhere(Score_ceil == np.min(Score_ceil))[0,1]]
        eps_opt = eps_plot[np.argwhere(Score_ceil == np.min(Score_ceil))[0,0]]
        
        ### Hard coded these optimals since we simulated these points and
        ### they are equally justified given the uncertainty
        if lam == '18':
            eps_opt = 77.0
            sig_opt = 0.389
            
        elif lam == '20':
            
            eps_opt = 84.0
            sig_opt = 0.388
        
        eps_opt_all[lam] = eps_opt
        sig_opt_all[lam] = sig_opt
        ieps_opt_all[lam] = np.argwhere(eps_opt == eps_plot)[0][0]
        isig_opt_all[lam] = np.argwhere(sig_opt == sig_plot)[0][0]
        Score_opt_all[lam] = np.min(Score_ceil)
        
        if lam == '12':
            ieps_ref = np.argwhere(eps_ref == eps_plot)[0][0]
            isig_ref = np.argwhere(sig_ref == sig_plot)[0][0]
            Score_ref = Score_matrix_all[lam][ieps_ref,isig_ref]
        
#        axarr[axarr_dic[lam]].plot(sig_opt,eps_opt,'w*',markersize=20,markeredgecolor='k',markeredgewidth=2)
#        axarr2[axarr_dic[lam]].plot(sig_opt,eps_opt,'w*',markersize=20,markeredgecolor='k',markeredgewidth=2)

        axarr[axarr_dic[lam]].plot(sig_opt,eps_opt,'wX',markersize=15,markeredgecolor='k',markeredgewidth=3)
        axarr2[axarr_dic[lam]].plot(sig_opt,eps_opt,'wX',markersize=15,markeredgecolor='k',markeredgewidth=3)
        
    #    CS2 = axarr2[axarr_dic[lam]].contourf(sig_plot,eps_plot,Neff_matrix_all[lam],cmap='jet_r',levels=levels_Neff)
    #    CS2 = axarr2[axarr_dic[lam]].contourf(sig_plot,eps_plot,Neff_ceil,cmap='jet',levels=levels_Neff)
        CS2 = axarr2[axarr_dic[lam]].contourf(sig_plot,eps_plot,np.log10(Neff_ceil),cmap='jet',levels=levels_Neff)
    #    axarr2[axarr_dic[lam]].colorbar()
#        if lam == '12': axarr[axarr_dic[lam]].plot(sig_ref,eps_ref,'kx',markersize=10)
        axarr2[axarr_dic[lam]].set_xlabel(r'$\sigma_{\rm CH_2}$ (nm)')
        axarr2[axarr_dic[lam]].set_ylabel(r'$\epsilon_{\rm CH_2}$ (K)')
        axarr2[axarr_dic[lam]].set_title(r'$\lambda_{\rm CH_2} = $'+lam,y=1.02)
        
        if lam == '12': axarr3.plot(sig_ref,eps_ref,'kx',markersize=10)
        CS3 = axarr3.contour(sig_plot,eps_plot,Score_ceil)
        axarr3.set_xlabel(r'$\sigma_{\rm CH_2}$ (nm)')
        axarr3.set_ylabel(r'$\epsilon_{\rm CH_2}$ (K)')
        axarr3.set_title(r'$\lambda_{\rm CH_2} = $'+lam)
### Initial format       
#axarr[0,0].plot([],[],'kx',markersize=20,markeredgecolor='k',markeredgewidth=2,label=r'TraPPE: $\theta^{\langle0\rangle}, \lambda_{\rm CH_2} = 12$')
#axarr[0,0].plot([],[],'w*',markersize=20,markeredgecolor='k',markeredgewidth=2,label=r'$\theta^{\langle1\rangle}, \lambda_{\rm CH_2} = 12, 14, 16, 18$')
#axarr[0,0].legend(loc='lower left')

axarr[0,0].plot([],[],'w^',markersize=15,markeredgecolor='k',markeredgewidth=2,label=r'TraPPE: $\theta^{\langle0\rangle}, \lambda_{\rm CH_2} = 12$')
axarr[0,0].plot([],[],'wX',markersize=15,markeredgecolor='k',markeredgewidth=3,label=r'$\theta^{\langle1\rangle}, \lambda_{\rm CH_2} = 12, 14, 16, 18$')
axarr[0,0].legend(loc='lower left')

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.95,0.05,0.025,0.875])
cbar = fig.colorbar(CS,cbar_ax)

### Initial format
#axarr2[0,0].plot([],[],'kx',markersize=20,markeredgecolor='k',markeredgewidth=2,label=r'TraPPE: $\theta^{\langle0\rangle}, \lambda_{\rm CH_2} = 12$')
#axarr2[0,0].plot([],[],'w*',markersize=20,markeredgecolor='k',markeredgewidth=2,label=r'$\theta^{\langle1\rangle}, \lambda_{\rm CH_2} = 12, 14, 16, 18$')
#axarr2[0,0].legend(loc='lower left')

axarr2[0,0].plot([],[],'w^',markersize=15,markeredgecolor='k',markeredgewidth=2,label=r'TraPPE: $\theta^{\langle0\rangle}, \lambda_{\rm CH_2} = 12$')
axarr2[0,0].plot([],[],'wX',markersize=15,markeredgecolor='k',markeredgewidth=3,label=r'$\theta^{\langle1\rangle}, \lambda_{\rm CH_2} = 12, 14, 16, 18$')
axarr2[0,0].legend(loc='lower left')

fig2.subplots_adjust(right=0.9)
cbar_ax2 = fig2.add_axes([0.95,0.05,0.025,0.875])
cbar2 = fig2.colorbar(CS2,cbar_ax2)

#cbar = fig.colorbar(CS)
#cbar2 = fig2.colorbar(CS2)
fig3.colorbar(CS3)

### Be careful with these, they just apply labels to contour segments, but can be totally wrong if the user chooses
cbar.set_ticklabels([0,2,4,6,8,10,12,14,16,18,20])
#cbar2.set_ticklabels([0,50,100,150,200,250,300,350,400,450,500])
cbar2.set_ticklabels([0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0])

cbar.set_label(r'Scoring function')
#cbar2.set_label(r'$K^{\rm eff}_{\rm snaps}$)
cbar2.set_label(r'$\log_{10}(\bar K^{\rm eff, liq}_{\rm snaps})$')

fig.savefig('CYC6_scoring_function_lam.pdf',bbox_inches='tight')
fig2.savefig('CYC6_Neff_lam.pdf',bbox_inches='tight')
fig3.savefig('CYC6_scoring_function_combined.pdf')

plt.show()

symbol_list = {12:'ro',14:'bs',16:'gd',18:'m^',20:'cv','ref':'k<'}
line_list = {12:'r--',14:'b:',16:'g-.',18:'m--',20:'c:','ref':'k-'}

fig, axarr = plt.subplots(nrows=3,ncols=1,figsize=[6,18])
fig2, axarr2 = plt.subplots(nrows=2,ncols=2,figsize=[12,12])

### First plot the reference results

isig = isig_ref
ieps = ieps_ref
Score_opt = Score_ref
lam = 12

root_path = 'H:/GCMC_basis_functions/'+referenceFF+'/'+compound+'/MBAR_results/lam'+str(lam)+'sig'+str(isig+1)

output_path = root_path+'eps'+str(ieps+1)+'/'
    
VLE_MBAR = np.loadtxt(output_path+'MBAR_VLE.txt',skiprows=1)
Tsat_MBAR = VLE_MBAR[::thin_Tsat,0]
rhol_MBAR = VLE_MBAR[::thin_Tsat,1]
rhov_MBAR = VLE_MBAR[::thin_Tsat,2]
Psat_MBAR = VLE_MBAR[::thin_Tsat,3]
DeltaHv_MBAR = VLE_MBAR[::thin_Tsat,4]

if remove_low_high_Tsat:  #Remove the low T and high T ends for more stable results
    Tsat_MBAR = Tsat_MBAR[2:-6]
    rhol_MBAR = rhol_MBAR[2:-6]
    rhov_MBAR = rhov_MBAR[2:-6]
    Psat_MBAR = Psat_MBAR[2:-6]
    DeltaHv_MBAR = DeltaHv_MBAR[2:-6]

Score_ieps_isig = Scoring_function(Tsat_MBAR,rhol_MBAR,rhov_MBAR,Psat_MBAR,DeltaHv_MBAR,rhol_RP,rhov_RP,Psat_RP,DeltaHv_RP,Neff_liq,use_Neff_cutoff,include_liquid,include_vapor)

assert Score_ieps_isig == Score_opt, 'Not the correct isig and ieps, Score_opt = '+str(Score_opt)+' while Score_ieps_isig = '+str(Score_ieps_isig)

axarr[0].plot(rhol_MBAR,Tsat_MBAR,symbol_list['ref'],mfc='None',label=r'$\theta_{\rm TraPPE}$')
axarr[0].plot(rhov_MBAR,Tsat_MBAR,symbol_list['ref'],mfc='None',label=r'$\theta_{\rm TraPPE}$')
axarr[1].plot(1000./Tsat_MBAR,np.log10(Psat_MBAR),symbol_list['ref'],mfc='None',label=r'$\theta_{\rm TraPPE}$')
axarr[2].plot(Tsat_MBAR,DeltaHv_MBAR,symbol_list['ref'],mfc='None',label=r'$\theta_{\rm TraPPE}$')

rhol_PD = (rhol_MBAR - rhol_RP)/rhol_RP * 100.
rhov_PD = (rhov_MBAR - rhov_RP)/rhov_RP * 100.
Psat_PD = (Psat_MBAR - Psat_RP)/Psat_RP * 100.
DeltaHv_PD = (DeltaHv_MBAR - DeltaHv_RP)/DeltaHv_RP * 100.

axarr2[0,0].plot(Tsat_MBAR,rhol_PD,line_list['ref'],linewidth=3,label=r'$\theta_{\rm TraPPE}$')
axarr2[0,1].plot(Tsat_MBAR,rhov_PD,line_list['ref'],linewidth=3,label=r'$\theta_{\rm TraPPE}$')
axarr2[1,0].plot(Tsat_MBAR,Psat_PD,line_list['ref'],linewidth=3,label=r'$\theta_{\rm TraPPE}$')
axarr2[1,1].plot(Tsat_MBAR,DeltaHv_PD,line_list['ref'],linewidth=3,label=r'$\theta_{\rm TraPPE}$')
        
### Loop over the optimal for each lambda
                
for lam in lam_range:
    
    isig = isig_opt_all[str(lam)]
    ieps = ieps_opt_all[str(lam)]
    Score_opt = Score_opt_all[str(lam)]
          
    root_path = 'H:/GCMC_basis_functions/'+referenceFF+'/'+compound+'/MBAR_results/lam'+str(lam)+'sig'+str(isig+1)
    
    output_path = root_path+'eps'+str(ieps+1)+'/'
        
    VLE_MBAR = np.loadtxt(output_path+'MBAR_VLE.txt',skiprows=1)
    Tsat_MBAR = VLE_MBAR[::thin_Tsat,0]
    rhol_MBAR = VLE_MBAR[::thin_Tsat,1]
    rhov_MBAR = VLE_MBAR[::thin_Tsat,2]
    Psat_MBAR = VLE_MBAR[::thin_Tsat,3]
    DeltaHv_MBAR = VLE_MBAR[::thin_Tsat,4]
    
    if remove_low_high_Tsat:  #Remove the low T and high T ends for more stable results
        Tsat_MBAR = Tsat_MBAR[2:-6]
        rhol_MBAR = rhol_MBAR[2:-6]
        rhov_MBAR = rhov_MBAR[2:-6]
        Psat_MBAR = Psat_MBAR[2:-6]
        DeltaHv_MBAR = DeltaHv_MBAR[2:-6]
    
    Score_ieps_isig = Scoring_function(Tsat_MBAR,rhol_MBAR,rhov_MBAR,Psat_MBAR,DeltaHv_MBAR,rhol_RP,rhov_RP,Psat_RP,DeltaHv_RP,Neff_liq,use_Neff_cutoff,include_liquid,include_vapor)
    
    assert Score_ieps_isig == Score_opt, 'Not the correct isig and ieps, Score_opt = '+str(Score_opt)+' while Score_ieps_isig = '+str(Score_ieps_isig)
    
    axarr[0].plot(rhol_MBAR,Tsat_MBAR,symbol_list[lam],mfc='None',label=r'$\lambda_{\rm CH_2} = '+str(lam))
    axarr[0].plot(rhov_MBAR,Tsat_MBAR,symbol_list[lam],mfc='None',label=r'$\lambda_{\rm CH_2} = '+str(lam))
    axarr[1].plot(1000./Tsat_MBAR,np.log10(Psat_MBAR),symbol_list[lam],mfc='None',label=r'$\lambda_{\rm CH_2} = '+str(lam))
    axarr[2].plot(Tsat_MBAR,DeltaHv_MBAR,symbol_list[lam],mfc='None',label=r'$\lambda_{\rm CH_2} = '+str(lam))
    
    rhol_PD = (rhol_MBAR - rhol_RP)/rhol_RP * 100.
    rhov_PD = (rhov_MBAR - rhov_RP)/rhov_RP * 100.
    Psat_PD = (Psat_MBAR - Psat_RP)/Psat_RP * 100.
    DeltaHv_PD = (DeltaHv_MBAR - DeltaHv_RP)/DeltaHv_RP * 100.
    
    axarr2[0,0].plot(Tsat_MBAR,rhol_PD,line_list[lam],linewidth=3,label=r'$\lambda_{\rm CH_2} = '+str(lam))
    axarr2[0,1].plot(Tsat_MBAR,rhov_PD,line_list[lam],linewidth=3,label=r'$\lambda_{\rm CH_2} = '+str(lam))
    axarr2[1,0].plot(Tsat_MBAR,Psat_PD,line_list[lam],linewidth=3,label=r'$\lambda_{\rm CH_2} = '+str(lam))
    axarr2[1,1].plot(Tsat_MBAR,DeltaHv_PD,line_list[lam],linewidth=3,label=r'$\theta_{\rm opt}$, $\lambda = $'+str(lam))
    
axarr[0].plot(rhol_RP,Tsat_RP,'k-',label='REFPROP')
axarr[0].plot(rhov_RP,Tsat_RP,'k-',label='REFPROP')
axarr[1].plot(1000./Tsat_RP,np.log10(Psat_RP),'k-',label='REFPROP')         
axarr[2].plot(Tsat_RP,DeltaHv_RP,'k-',label='REFPROP')

axarr[0].set_xlabel(r'$\rho$ (kg/m$^3$)')
axarr[0].set_ylabel(r'$T$ (K)')

axarr[1].set_ylabel(r'$\log_{10}(P^{\rm sat}_{\rm vap}$/bar)')
axarr[1].set_xlabel(r'$1000/T$ (K)')

axarr[2].set_ylabel(r'$\Delta H_{\rm v}$ (kJ/mol)')
axarr[2].set_xlabel(r'$T$ (K)')

fig.savefig('H:/MBAR_GCMC/VLE_optimal_lam.pdf',bbox_inches='tight')

for iax in range(2):
    for jax in range(2):
        axarr2[iax,jax].set_xlabel(r'$T$ (K)')
        axarr2[iax,jax].set_ylabel(r'$\frac{X_{\rm MBAR} - X_{\rm REFPROP}}{X_{\rm REFPROP}} \times 100$ %',fontsize=24)
        axarr2[iax,jax].set_xticks([350,400,450,500,550])

axarr2[0,0].set_title(r'$X = \rho^{\rm sat}_{\rm liq}$')
axarr2[0,1].set_title(r'$X = \rho^{\rm sat}_{\rm vap}$')
axarr2[1,0].set_title(r'$X = P^{\rm sat}_{\rm vap}$')
axarr2[1,1].set_title(r'$X = \Delta H_{\rm v}$')

#for rr in ['ref','12','14','16',18']

fig2.tight_layout()

lgd = axarr2[1,1].legend(loc='lower center', bbox_to_anchor=(-0.3, -0.41),
          ncol=3,numpoints=1,handlelength=5,handletextpad=0.5,columnspacing=0.5,frameon=True,borderaxespad=0)
         
fig2.savefig('H:/MBAR_GCMC/deviation_plots_optimal_lam.pdf',bbox_extra_artists=(lgd,),bbox_inches='tight')