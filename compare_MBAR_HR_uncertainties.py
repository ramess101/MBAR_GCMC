# -*- coding: utf-8 -*-
"""
Analyze branched alkanes
"""

from __future__ import division
import numpy as np
from optimize_epsilon import MBAR_GCMC
import os.path
from extract_Potoff import extract_Potoff
import matplotlib.pyplot as plt
from CriticalPoint_RectilinearScaling import ITIC_VLE

font = {'size' : '24'}
plt.rc('font',**font)

nhists_max = 15
ncompounds_max = 16
icompound = 0

scaled_pu = False

Rg = 8.3144598e-5 #[bar m3 / mol / K]

group_list = ['C-CH-group','C-group','CH-group','alkynes']

compound_list = {'C-CH-group':['223TMButane','223TMPentane','224TMPentane','233TMPentane'],'C-group':['3M3EPentane','22DMButane','22DMHexane','22DMPentane','22DMPropane','33DMHexane','33DMPentane','2233TetraMButane'],'CH-group':['2M3EPentane','2MButane','2MHeptane','2MHexane','2MPentane','2MPropane','3EHexane','3EPentane','3MHeptane','3MHexane','3MPentane','4MHeptane','23DMBUtane','23DMHexane','23DMPentane','24DMHexane','24DMPentane','25DMHexane','34DMHexane','234TMPentane'],'alkynes':['ethyne','propyne','1butyne','2butyne','1pentyne','2pentyne','1hexyne','2hexyne','1heptyne','1octyne','1nonyne','propadiene']}  #{'C-group':['22DMPropane'],'C-CH-group':['224TMPentane']}

Mw_list = {'223TMButane':100.205,'223TMPentane':114.22852,'224TMPentane':114.22852,'233TMPentane':114.22852,'3M3EPentane':114.22852,'22DMButane':86.17536,'22DMHexane':114.22852,'22DMPentane':100.205,'22DMPropane':72.14878,'33DMHexane':114.22852,'33DMPentane':100.205,'2233TetraMButane':114.22852,'2M3EPentane':114.22852,'2MButane':72.14878,'2MHeptane':114.22852,'2MHexane':100.205,'2MPentane':86.17536,'2MPropane':58.1222,'3EHexane':114.22852,'3EPentane':100.205,'3MHeptane':114.22852,'3MHexane':100.205,'3MPentane':86.17536,'4MHeptane':114.22852,'23DMBUtane':86.17536,'23DMHexane':114.22852,'23DMPentane':100.205,'24DMHexane':114.22852,'24DMPentane':100.205,'25DMHexane':114.22852,'34DMHexane':114.22852,'234TMPentane':114.22852,'ethyne':26.04,'propyne':40.0639,'1butyne':54.091,'2butyne':54.091,'1pentyne':68.12,'2pentyne':68.12,'1hexyne':82.14,'2hexyne':82.14,'1heptyne':96.17,'1octyne':110.2,'1nonyne':124.22,'propadiene':40.06} #[gm/mol]

color_list = ['b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y']
symbol_list = {'C-CH-group':'o','C-group':'s','CH-group':'^','alkynes':'d'}
line_list = ['-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.']

directory_dic = {'C-CH-group':'branched-Alkanes/C-CH-group/','C-group':'branched-Alkanes/C-group/','CH-group':'branched-Alkanes/CH-group/','alkynes':'alkynes/'}
axarr_dic = {0:(0,0),1:(0,1),2:(1,0),3:(1,1),4:(2,0),5:(2,1)}

pu_rhol_bins = {0.65:[],0.7:[],0.75:[],0.8:[],0.85:[],0.9:[],0.95:[]}
pu_rhov_bins = {0.65:[],0.7:[],0.75:[],0.8:[],0.85:[],0.9:[],0.95:[]}
pu_Psat_bins = {0.65:[],0.7:[],0.75:[],0.8:[],0.85:[],0.9:[],0.95:[]}
pu_DeltaHv_bins = {0.65:[],0.7:[],0.75:[],0.8:[],0.85:[],0.9:[],0.95:[]}
pu_DeltaUv_bins = {0.65:[],0.7:[],0.75:[],0.8:[],0.85:[],0.9:[],0.95:[]}
pu_PV_bins = {0.65:[],0.7:[],0.75:[],0.8:[],0.85:[],0.9:[],0.95:[]}
pu_Z_bins = {0.65:[],0.7:[],0.75:[],0.8:[],0.85:[],0.9:[],0.95:[]}
Tr_bins = np.array([0.65,0.7,0.75,0.8,0.85,0.9,0.95])

pu_rhol_bins_Potoff = {0.675:[],0.725:[],0.775:[],0.825:[],0.875:[],0.925:[]}
pu_rhov_bins_Potoff = {0.675:[],0.725:[],0.775:[],0.825:[],0.875:[],0.925:[]}
pu_Psat_bins_Potoff = {0.675:[],0.725:[],0.775:[],0.825:[],0.875:[],0.925:[]}
pu_DeltaHv_bins_Potoff = {0.675:[],0.725:[],0.775:[],0.825:[],0.875:[],0.925:[]}
pu_DeltaUv_bins_Potoff = {0.675:[],0.725:[],0.775:[],0.825:[],0.875:[],0.925:[]}
pu_PV_bins_Potoff = {0.675:[],0.725:[],0.775:[],0.825:[],0.875:[],0.925:[]}
pu_Z_bins_Potoff = {0.675:[],0.725:[],0.775:[],0.825:[],0.875:[],0.925:[]}
Tr_bins_Potoff = np.array([0.675,0.725,0.775,0.825,0.875,0.925])

fig, axarr = plt.subplots(nrows=2,ncols=2,figsize=[12,12])

for group in group_list:
       
    for compound in compound_list[group]: # ['224-trimethylpentane','neopentane','isobutane']:
             
            Mw = Mw_list[compound]
        
            filepaths = []
            
            root_path = 'H:/GCMC_histFiles/'+directory_dic[group]+compound
            
            for iT in np.arange(1,nhists_max+1):
                            
                hist_name='/his'+str(iT)+'a.dat'
                    
                if os.path.exists(root_path+hist_name):
                    
                    filepaths.append(root_path+hist_name)
            
            if not os.path.exists('H:/MBAR_GCMC/Potoff_literature/'+compound+'_withuncertainties.txt'):
                
                input_path = 'H:/MBAR_GCMC/Potoff_literature/'+compound+'_rawdata.txt'
                output_path = 'H:/MBAR_GCMC/Potoff_literature/'+compound+'_withuncertainties.txt'
                
                extract_Potoff(input_path,output_path,with_uncertainties=True)
                
            VLE_Potoff = np.loadtxt('H:/MBAR_GCMC/Potoff_literature/'+compound+'_withuncertainties.txt',skiprows=1)
                
            Tsat_Potoff = VLE_Potoff[:,0] #[K]
            rhol_Potoff = VLE_Potoff[:,1]*1000. #[kg/m3]
            urhol_Potoff = VLE_Potoff[:,2]*1000. #[kg/m3]
            rhov_Potoff = VLE_Potoff[:,3]*1000. #[kg/m3]
            urhov_Potoff = VLE_Potoff[:,4]*1000. #[kg/m3]
            Psat_Potoff = VLE_Potoff[:,5] #[bar]
            uPsat_Potoff = VLE_Potoff[:,6] #[bar]
            DeltaHv_Potoff = VLE_Potoff[:,7] #[kJ/mol]
            uDeltaHv_Potoff = VLE_Potoff[:,8] #[kJ/mol]
            Z_Potoff = VLE_Potoff[:,9]
            uZ_Potoff = VLE_Potoff[:,10]
            
            Vl_Potoff = Mw / rhol_Potoff / 1000. #[m3/mol]
            Vv_Potoff = Mw / rhov_Potoff / 1000. #[m3/mol]
            
            PV_Potoff = Psat_Potoff * (Vv_Potoff - Vl_Potoff) / 1000. / 1000. #[kJ/mol]
            uPV_Potoff = PV_Potoff * np.sqrt((urhol_Potoff/rhol_Potoff)**2.+(urhov_Potoff/rhov_Potoff)**2.+(uPsat_Potoff/Psat_Potoff)**2.)
            
            DeltaUv_Potoff = DeltaHv_Potoff - PV_Potoff #[kJ/mol]
            uDeltaUv_Potoff = np.sqrt(uDeltaHv_Potoff**2. + uPV_Potoff**2.) 
            
            t90_to_t95 = 2.78 / 2.13 #Uncertainties were reported at 90% confidence level
            N5_to_N1 = np.sqrt(5) # Five replicates compared to 1 replicate
            
            pu_rhol_Potoff = t90_to_t95 * N5_to_N1 * urhol_Potoff/rhol_Potoff * 100.
            pu_rhov_Potoff = t90_to_t95 * N5_to_N1 * urhov_Potoff/rhov_Potoff * 100.
            pu_Psat_Potoff = t90_to_t95 * N5_to_N1 * uPsat_Potoff/Psat_Potoff * 100.
            pu_DeltaHv_Potoff = t90_to_t95 * N5_to_N1 * uDeltaHv_Potoff/DeltaHv_Potoff * 100.
            pu_Z_Potoff = t90_to_t95 * N5_to_N1 * uZ_Potoff/Z_Potoff * 100.
            pu_PV_Potoff = t90_to_t95 * N5_to_N1 * uPV_Potoff/PV_Potoff * 100.
            pu_DeltaUv_Potoff = t90_to_t95 * N5_to_N1 * uDeltaUv_Potoff/DeltaUv_Potoff * 100.
            
            
            if not os.path.exists('H:/MBAR_GCMC/Potoff_literature/'+compound+'_Tc.txt'):
    
                if Tsat_Potoff[0] < Tsat_Potoff[-1]:
                    ###Have to make sure that Tsat[0] is the highest value since this code was written for ITIC
                    Tsat_Potoff = Tsat_Potoff[::-1]
                    rhol_Potoff = rhol_Potoff[::-1]
                    rhov_Potoff = rhov_Potoff[::-1]
                    Psat_Potoff = Psat_Potoff[::-1]
                    DeltaHv_Potoff = DeltaHv_Potoff[::-1]
                    Z_Potoff = Z_Potoff[::-1]
            
                Potoff_fit = ITIC_VLE(Tsat_Potoff,rhol_Potoff,rhov_Potoff,Psat_Potoff)
                Tc_Potoff = Potoff_fit.Tc
                
                out_file =open('H:/MBAR_GCMC/Potoff_literature/'+compound+'_Tc.txt','w')
                out_file.write(str(Tc_Potoff))
                out_file.close()
                
            Tc_Potoff = np.loadtxt('H:/MBAR_GCMC/Potoff_literature/'+compound+'_Tc.txt')       
            
            if os.path.exists('H:/MBAR_GCMC/MBAR_values/'+compound+'.txt') and os.path.exists('H:/MBAR_GCMC/MBAR_values/'+compound+'_uncertainties.txt'):
                print('Loading MBAR-GCMC analysis for '+compound)
                MBAR_GCMC_load = np.loadtxt('H:/MBAR_GCMC/MBAR_values/'+compound+'.txt',skiprows=1)
                
                Tsat_MBAR = MBAR_GCMC_load[:,0]
                rhol_MBAR = MBAR_GCMC_load[:,1]
                rhov_MBAR = MBAR_GCMC_load[:,2]
                Psat_MBAR = MBAR_GCMC_load[:,3]
                DeltaHv_MBAR = MBAR_GCMC_load[:,4]
                
                MBAR_GCMC_load = np.loadtxt('H:/MBAR_GCMC/MBAR_values/'+compound+'_uncertainties.txt',skiprows=1)
                
                urhol_MBAR = MBAR_GCMC_load[:,1]
                urhov_MBAR = MBAR_GCMC_load[:,2]
                uPsat_MBAR = MBAR_GCMC_load[:,3]
                uDeltaHv_MBAR = MBAR_GCMC_load[:,4]
                
                assert len(Tsat_MBAR) == len(urhol_MBAR), 'Values and uncertainties dimensions mismatch'
  
            else:

                print('Error when performing MBAR-GCMC analysis for '+compound)
                break
                    
            ### Post-process values that can be derived
            Z_MBAR = Psat_MBAR * Mw / rhov_MBAR / Rg / Tsat_MBAR / 1000.
            
            Vl_MBAR = Mw / rhol_MBAR / 1000. #[m3/mol]
            Vv_MBAR = Mw / rhov_MBAR / 1000. #[m3/mol]
            
            PV_MBAR = Psat_MBAR * (Vv_MBAR - Vl_MBAR) / 1000. / 1000. #[kJ/mol]
            
            DeltaUv_MBAR = DeltaHv_MBAR - PV_MBAR #[kJ/mol]

            if os.path.exists('H:/MBAR_GCMC/MBAR_values/'+compound+'.txt'):  
                
                Tr_Potoff = Tsat_Potoff/Tc_Potoff
                
                pu_rhol = urhol_MBAR / rhol_MBAR * 100.
                pu_rhov = urhov_MBAR / rhov_MBAR * 100.
                pu_Psat = uPsat_MBAR / Psat_MBAR * 100.
                pu_DeltaHv = uDeltaHv_MBAR / DeltaHv_MBAR * 100.
                pu_Z = np.sqrt(pu_Psat**2. + pu_rhov**2.)
                ### I think these two expressions are wrong
                pu_PV = np.sqrt(pu_rhol**2.+pu_rhov**2.+pu_Psat**2.)
                pu_DeltaUv = np.sqrt(pu_DeltaHv**2. + pu_PV**2.) 
                               
                axarr[0,0].errorbar(Tr_Potoff[np.abs(pu_rhol)<5],pu_rhol[np.abs(pu_rhol)<5],fmt=color_list[icompound]+symbol_list[group]+line_list[icompound],mfc='None')
                axarr[0,1].errorbar(Tr_Potoff[np.abs(pu_rhov)<10],pu_rhov[np.abs(pu_rhov)<10],fmt=color_list[icompound]+symbol_list[group]+line_list[icompound],mfc='None')
                axarr[1,0].errorbar(Tr_Potoff[np.abs(pu_Psat)<20],pu_Psat[np.abs(pu_Psat)<20],fmt=color_list[icompound]+symbol_list[group]+line_list[icompound],mfc='None')
                axarr[1,1].errorbar(Tr_Potoff[np.abs(pu_DeltaHv)<20],pu_DeltaHv[np.abs(pu_DeltaHv)<20],fmt=color_list[icompound]+symbol_list[group]+line_list[icompound],mfc='None')
#                axarr[2,0].errorbar(Tr_Potoff[np.abs(pu_Z)<20],pu_Z[np.abs(pu_Z)<20],fmt=color_list[icompound]+symbol_list[group]+line_list[icompound],mfc='None')
#                axarr[2,1].errorbar(Tr_Potoff[np.abs(pu_PV)<20],pu_PV[np.abs(pu_PV)<20],fmt=color_list[icompound]+symbol_list[group]+line_list[icompound],mfc='None')

                for Tr, pu_rhol_i, pu_rhov_i, pu_Psat_i, pu_DeltaHv_i, pu_Z_i, pu_PV_i, pu_DeltaUv_i in zip(Tr_Potoff,pu_rhol,pu_rhov,pu_Psat,pu_DeltaHv,pu_Z,pu_PV,pu_DeltaUv): 
                    
                    iBin = np.argmin(np.abs(Tr-Tr_bins))
                    
                    if np.abs(pu_rhol_i) < 5: pu_rhol_bins[Tr_bins[iBin]].append(pu_rhol_i)
                    if np.abs(pu_rhov_i) < 10: pu_rhov_bins[Tr_bins[iBin]].append(pu_rhov_i)
                    if np.abs(pu_Psat_i) < 20: pu_Psat_bins[Tr_bins[iBin]].append(pu_Psat_i)
                    if np.abs(pu_DeltaHv_i) < 20: pu_DeltaHv_bins[Tr_bins[iBin]].append(pu_DeltaHv_i)
                    if np.abs(pu_Z_i) < 20: pu_Z_bins[Tr_bins[iBin]].append(pu_Z_i)
                    if np.abs(pu_PV_i) < 20: pu_PV_bins[Tr_bins[iBin]].append(pu_PV_i)
                    if np.abs(pu_DeltaUv_i) < 20: pu_DeltaUv_bins[Tr_bins[iBin]].append(pu_DeltaUv_i)
                    
                for Tr, pu_rhol_i, pu_rhov_i, pu_Psat_i, pu_DeltaHv_i, pu_Z_i, pu_PV_i, pu_DeltaUv_i in zip(Tr_Potoff,pu_rhol_Potoff,pu_rhov_Potoff,pu_Psat_Potoff,pu_DeltaHv_Potoff,pu_Z_Potoff,pu_PV_Potoff,pu_DeltaUv_Potoff): 
                    
                    iBin = np.argmin(np.abs(Tr-Tr_bins_Potoff))
                    
                    if np.abs(pu_rhol_i) < 5: pu_rhol_bins_Potoff[Tr_bins_Potoff[iBin]].append(pu_rhol_i)
                    if np.abs(pu_rhov_i) < 10: pu_rhov_bins_Potoff[Tr_bins_Potoff[iBin]].append(pu_rhov_i)
                    if np.abs(pu_Psat_i) < 20: pu_Psat_bins_Potoff[Tr_bins_Potoff[iBin]].append(pu_Psat_i)
                    if np.abs(pu_DeltaHv_i) < 20: pu_DeltaHv_bins_Potoff[Tr_bins_Potoff[iBin]].append(pu_DeltaHv_i)
                    if np.abs(pu_Z_i) < 20: pu_Z_bins_Potoff[Tr_bins_Potoff[iBin]].append(pu_Z_i)
                    if np.abs(pu_PV_i) < 20: pu_PV_bins_Potoff[Tr_bins_Potoff[iBin]].append(pu_PV_i)
                    if np.abs(pu_DeltaUv_i) < 20: pu_DeltaUv_bins_Potoff[Tr_bins_Potoff[iBin]].append(pu_DeltaUv_i)
                    
            icompound += 1
            
            if icompound >= ncompounds_max: #Break out of first for loop
                
                break
            
    if icompound >= ncompounds_max: #Break out of second for loop
                
                break
            
prop_list = [r'$\rho_{\rm liq}^{\rm sat}$',r'$\rho_{\rm vap}^{\rm sat}$',r'$P_{\rm vap}^{\rm sat}$',r'$\Delta H_{\rm v}$'] #,r'$Z^{\rm sat}_{\rm vap}$',r'$P \Delta V$']  #r'$\Delta U_{\rm v}$'] #r'$P \Delta V$'] #r'$Z^{\rm sat}_{\rm vap}$']#r'$\Delta H_{\rm v}$']

if scaled_pu:
    ylim_low = [-6,-6,-6,-6,-6,-6]
    ylim_high = [6,6,6,6,6,6]
else:
    ylim_low = [0,0,0,0,0,0]
    ylim_high = [None,None,None,None,None,None]
        
for iax, prop in enumerate(prop_list):
    axarr[axarr_dic[iax]].set_xlabel(r'$T_{\rm r}$')
    
    if scaled_pu:
        axarr[axarr_dic[iax]].set_ylabel(r'$\frac{X_{\rm MBAR} - X_{\rm HR}}{\delta X_{\rm HR}}$')  
    else:
        axarr[axarr_dic[iax]].set_ylabel(r'$\frac{\delta_{\rm MBAR}}{X_{\rm MBAR}} \times 100 \%$')
    axarr[axarr_dic[iax]].set_title(r'$X = $'+prop)
    axarr[axarr_dic[iax]].set_xlim([0.6,1.])
    axarr[axarr_dic[iax]].set_xticks([0.65,0.75,0.85,0.95])
    axarr[axarr_dic[iax]].set_ylim([ylim_low[iax],ylim_high[iax]])
    
#axarr[0,0].set_ylabel(r'$\frac{X_{\rm MBAR} - X_{\rm HR}}{X_{\rm HR}} \times 100 \%$')
#axarr[1,1].set_xlabel(r'$T_{\rm r}$')
#axarr[1,0].set_ylabel(r'$\frac{X_{\rm MBAR} - X_{\rm HR}}{X_{\rm HR}} \times 100 \%$')
#axarr[1,0].set_xlabel(r'$T_{\rm r}$')

plt.tight_layout(pad=0.5)

fig.savefig('Comparison_MBAR_HR_uncertainties.pdf')

if scaled_pu:
    ylim_low = [-6,-6,-6,-6,-6,-6]
    ylim_high = [6,6,6,6,6,6]
else:        
    ylim_low = [0,0,0,0,0,0]
    ylim_high = [None,None,None,None,None,None]

fig, axarr = plt.subplots(nrows=2,ncols=2,figsize=[12,12])

boxplot_rhol = []
boxplot_rhov = []
boxplot_Psat = []
boxplot_DeltaHv = []
boxplot_Z = []
boxplot_PV = []
boxplot_DeltaUv = []

boxplot_rhol_Potoff = []
boxplot_rhov_Potoff = []
boxplot_Psat_Potoff = []
boxplot_DeltaHv_Potoff = []
boxplot_Z_Potoff = []
boxplot_PV_Potoff = []
boxplot_DeltaUv_Potoff = []

for Tr in Tr_bins:
    
    boxplot_rhol.append(pu_rhol_bins[Tr])
    boxplot_rhov.append(pu_rhov_bins[Tr])
    boxplot_Psat.append(pu_Psat_bins[Tr])
    boxplot_DeltaHv.append(pu_DeltaHv_bins[Tr])
    boxplot_Z.append(pu_Z_bins[Tr])
    boxplot_PV.append(pu_PV_bins[Tr])
    boxplot_DeltaUv.append(pu_DeltaUv_bins[Tr])
    
for Tr in Tr_bins_Potoff:
    
    boxplot_rhol_Potoff.append(pu_rhol_bins_Potoff[Tr])
    boxplot_rhov_Potoff.append(pu_rhov_bins_Potoff[Tr])
    boxplot_Psat_Potoff.append(pu_Psat_bins_Potoff[Tr])
    boxplot_DeltaHv_Potoff.append(pu_DeltaHv_bins_Potoff[Tr])
    boxplot_Z_Potoff.append(pu_Z_bins_Potoff[Tr])
    boxplot_PV_Potoff.append(pu_PV_bins_Potoff[Tr])
    boxplot_DeltaUv_Potoff.append(pu_DeltaUv_bins_Potoff[Tr])

axarr[0,0].boxplot(boxplot_rhol,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False)
axarr[0,1].boxplot(boxplot_rhov,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False)
axarr[1,0].boxplot(boxplot_Psat,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False)
axarr[1,1].boxplot(boxplot_DeltaHv,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False)
#axarr[2,0].boxplot(boxplot_Z,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False)
#axarr[2,1].boxplot(boxplot_PV,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False)
#axarr[1,1].boxplot(boxplot_DeltaUv,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False)

for iax, prop in enumerate(prop_list):
    axarr[axarr_dic[iax]].set_xlabel(r'$T_{\rm r}$')
    
    if scaled_pu:
        axarr[axarr_dic[iax]].set_ylabel(r'$\frac{X_{\rm MBAR} - X_{\rm HR}}{\delta X_{\rm HR}}$')  
    else:
        axarr[axarr_dic[iax]].set_ylabel(r'$\frac{\delta_{\rm MBAR}}{X_{\rm MBAR}} \times 100 \%$')
    axarr[axarr_dic[iax]].set_title(r'$X = $'+prop)
    axarr[axarr_dic[iax]].set_xlim([0.6,1.])
    axarr[axarr_dic[iax]].set_xticks([0.65,0.75,0.85,0.95])
    axarr[axarr_dic[iax]].set_ylim([ylim_low[iax],ylim_high[iax]])
        
#axarr[0].set_ylabel(r'$\frac{X_{\rm MBAR} - X_{\rm HR}}{X_{\rm HR}} \times 100 \%$')

plt.tight_layout(pad=0.5)

fig.savefig('Comparison_MBAR_HR_boxplot_uncertainties.pdf') 

if scaled_pu: 
    ylim_low = [-6,-6,-6,-6,-6,-6]
    ylim_high = [6,6,6,6,6,6]
else:
    ylim_low = [0,0,0,0,0,0]
    ylim_high = [None,None,None,None,None,None]

fig, axarr = plt.subplots(nrows=2,ncols=2,figsize=[12,12])

axarr[0,0].boxplot(boxplot_rhol,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False,whis=[5,95],sym='',patch_artist=True,boxprops=dict(facecolor='None',color='red'),capprops=dict(color='red'),whiskerprops=dict(color='red'),flierprops=dict(color='red', markeredgecolor='red'),medianprops=dict(color='red'))
axarr[0,1].boxplot(boxplot_rhov,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False,whis=[5,95],sym='',patch_artist=True,boxprops=dict(facecolor='None',color='red'),capprops=dict(color='red'),whiskerprops=dict(color='red'),flierprops=dict(color='red', markeredgecolor='red'),medianprops=dict(color='red'))
axarr[1,0].boxplot(boxplot_Psat,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False,whis=[5,95],sym='',patch_artist=True,boxprops=dict(facecolor='None',color='red'),capprops=dict(color='red'),whiskerprops=dict(color='red'),flierprops=dict(color='red', markeredgecolor='red'),medianprops=dict(color='red'))
axarr[1,1].boxplot(boxplot_DeltaHv,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False,whis=[5,95],sym='',patch_artist=True,boxprops=dict(facecolor='None',color='red'),capprops=dict(color='red'),whiskerprops=dict(color='red'),flierprops=dict(color='red', markeredgecolor='red'),medianprops=dict(color='red'))
#axarr[2,0].boxplot(boxplot_Z,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False,whis=[5,95],sym='')
#axarr[2,1].boxplot(boxplot_PV,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False,whis=[5,95],sym='')
#axarr[1,1].boxplot(boxplot_DeltaUv,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False,whis=[5,95],sym='')

axarr[0,0].boxplot(boxplot_rhol_Potoff,positions=Tr_bins_Potoff,widths=0.02,notch=True,manage_xticks=False,whis=[5,95],sym='',patch_artist=True,boxprops=dict(facecolor='None',color='blue'),capprops=dict(color='blue'),whiskerprops=dict(color='blue'),flierprops=dict(color='blue', markebluegecolor='blue'),medianprops=dict(color='blue'))
axarr[0,1].boxplot(boxplot_rhov_Potoff,positions=Tr_bins_Potoff,widths=0.02,notch=True,manage_xticks=False,whis=[5,95],sym='',patch_artist=True,boxprops=dict(facecolor='None',color='blue'),capprops=dict(color='blue'),whiskerprops=dict(color='blue'),flierprops=dict(color='blue', markebluegecolor='blue'),medianprops=dict(color='blue'))
axarr[1,0].boxplot(boxplot_Psat_Potoff,positions=Tr_bins_Potoff,widths=0.02,notch=True,manage_xticks=False,whis=[5,95],sym='',patch_artist=True,boxprops=dict(facecolor='None',color='blue'),capprops=dict(color='blue'),whiskerprops=dict(color='blue'),flierprops=dict(color='blue', markebluegecolor='blue'),medianprops=dict(color='blue'))
axarr[1,1].boxplot(boxplot_DeltaHv_Potoff,positions=Tr_bins_Potoff,widths=0.02,notch=True,manage_xticks=False,whis=[5,95],sym='',patch_artist=True,boxprops=dict(facecolor='None',color='blue'),capprops=dict(color='blue'),whiskerprops=dict(color='blue'),flierprops=dict(color='blue', markebluegecolor='blue'),medianprops=dict(color='blue'))

axarr[0,0].plot([],[],'rs',mfc='None',markersize=10,label='MBAR')
axarr[0,0].plot([],[],'bs',mfc='None',markersize=10,label='HR')
axarr[0,0].legend()
#bp0 = axarr[0,0].boxplot(boxplot_rhol_Potoff,positions=Tr_bins_Potoff,widths=0.02,notch=True,manage_xticks=False,whis=[5,95],sym='')
#bp1 = axarr[0,1].boxplot(boxplot_rhov_Potoff,positions=Tr_bins_Potoff,widths=0.02,notch=True,manage_xticks=False,whis=[5,95],sym='')
#bp2 = axarr[1,0].boxplot(boxplot_Psat_Potoff,positions=Tr_bins_Potoff,widths=0.02,notch=True,manage_xticks=False,whis=[5,95],sym='')
#bp3 = axarr[1,1].boxplot(boxplot_DeltaHv_Potoff,positions=Tr_bins_Potoff,widths=0.02,notch=True,manage_xticks=False,whis=[5,95],sym='')

#for bp in [bp0,bp1,bp2,bp3]:
#
#    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
#        plt.setp(bp[element], color='red')
#    
#    for patch in bp['boxes']:
#        patch.set(facecolor='red')

for iax, prop in enumerate(prop_list):
    axarr[axarr_dic[iax]].set_xlabel(r'$T_{\rm r}$')
    
    if scaled_pu:
        axarr[axarr_dic[iax]].set_ylabel(r'$\frac{X_{\rm MBAR} - X_{\rm HR}}{\delta X_{\rm HR}}$')  
    else:
#        axarr[axarr_dic[iax]].set_ylabel(r'$\frac{\delta_{\rm MBAR}}{X_{\rm MBAR}} \times 100 \%$')
        axarr[axarr_dic[iax]].set_ylabel(r'$\frac{\delta_{95}}{X} \times 100 \%$')
    axarr[axarr_dic[iax]].set_title(r'$X = $'+prop)
    axarr[axarr_dic[iax]].set_xlim([0.6,1.])
    axarr[axarr_dic[iax]].set_xticks([0.65,0.75,0.85,0.95])
    axarr[axarr_dic[iax]].set_ylim([ylim_low[iax],ylim_high[iax]])
        
#axarr[0].set_ylabel(r'$\frac{X_{\rm MBAR} - X_{\rm HR}}{X_{\rm HR}} \times 100 \%$')

plt.tight_layout(pad=0.5)

fig.savefig('Comparison_MBAR_HR_boxplot_CI_uncertainties.pdf')     