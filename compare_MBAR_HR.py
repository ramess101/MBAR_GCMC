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
ncompounds_max = 65
icompound = 0

reprocess = False
trim_data = False
scaled_dev = False

Rg = 8.3144598e-5 #[bar m3 / mol / K]

group_list = ['C-CH-group','C-group','CH-group','alkynes']

compound_list = {'C-CH-group':['223TMButane','223TMPentane','224TMPentane','233TMPentane'],'C-group':['3M3EPentane','22DMButane','22DMHexane','22DMPentane','22DMPropane','33DMHexane','33DMPentane','2233TetraMButane'],'CH-group':['2M3EPentane','2MButane','2MHeptane','2MHexane','2MPentane','2MPropane','3EHexane','3EPentane','3MHeptane','3MHexane','3MPentane','4MHeptane','23DMBUtane','23DMHexane','23DMPentane','24DMHexane','24DMPentane','25DMHexane','34DMHexane','234TMPentane'],'alkynes':['ethyne','propyne','1butyne','2butyne','1pentyne','2pentyne','1hexyne','2hexyne','1heptyne','1octyne','1nonyne']}  #{'C-group':['22DMPropane'],'C-CH-group':['224TMPentane']}

Mw_list = {'223TMButane':100.205,'223TMPentane':114.22852,'224TMPentane':114.22852,'233TMPentane':114.22852,'3M3EPentane':114.22852,'22DMButane':86.17536,'22DMHexane':114.22852,'22DMPentane':100.205,'22DMPropane':72.14878,'33DMHexane':114.22852,'33DMPentane':100.205,'2233TetraMButane':114.22852,'2M3EPentane':114.22852,'2MButane':72.14878,'2MHeptane':114.22852,'2MHexane':100.205,'2MPentane':86.17536,'2MPropane':58.1222,'3EHexane':114.22852,'3EPentane':100.205,'3MHeptane':114.22852,'3MHexane':100.205,'3MPentane':86.17536,'4MHeptane':114.22852,'23DMBUtane':86.17536,'23DMHexane':114.22852,'23DMPentane':100.205,'24DMHexane':114.22852,'24DMPentane':100.205,'25DMHexane':114.22852,'34DMHexane':114.22852,'234TMPentane':114.22852,'ethyne':26.04,'propyne':40.0639,'1butyne':54.091,'2butyne':54.091,'1pentyne':68.12,'2pentyne':68.12,'1hexyne':82.14,'2hexyne':82.14,'1heptyne':96.17,'1octyne':110.2,'1nonyne':124.22,'propadiene':40.06} #[gm/mol]

color_list = ['b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y']
symbol_list = {'C-CH-group':'o','C-group':'s','CH-group':'^','alkynes':'d'}
line_list = ['-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.']

### Old
#directory_dic = {'C-CH-group':'branched-Alkanes/C-CH-group/','C-group':'branched-Alkanes/C-group/','CH-group':'branched-Alkanes/CH-group/','alkynes':'alkynes/'}
directory_dic = {'C-CH-group':'branched-Alkanes/Optimized/C-CH-group/','C-group':'branched-Alkanes/Optimized/C-group/','CH-group':'branched-Alkanes/Optimized/CH-group/','alkynes':'alkynes/'}
axarr_dic = {0:(0,0),1:(0,1),2:(1,0),3:(1,1),4:(2,0),5:(2,1)}

dev_rhol_bins = {0.65:[],0.7:[],0.75:[],0.8:[],0.85:[],0.9:[],0.95:[]}
dev_rhov_bins = {0.65:[],0.7:[],0.75:[],0.8:[],0.85:[],0.9:[],0.95:[]}
dev_Psat_bins = {0.65:[],0.7:[],0.75:[],0.8:[],0.85:[],0.9:[],0.95:[]}
dev_DeltaHv_bins = {0.65:[],0.7:[],0.75:[],0.8:[],0.85:[],0.9:[],0.95:[]}
dev_DeltaUv_bins = {0.65:[],0.7:[],0.75:[],0.8:[],0.85:[],0.9:[],0.95:[]}
dev_PV_bins = {0.65:[],0.7:[],0.75:[],0.8:[],0.85:[],0.9:[],0.95:[]}
dev_Z_bins = {0.65:[],0.7:[],0.75:[],0.8:[],0.85:[],0.9:[],0.95:[]}
Tr_bins = np.array([0.65,0.7,0.75,0.8,0.85,0.9,0.95])

fig, axarr = plt.subplots(nrows=3,ncols=2,figsize=[12,18])

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
            
            if os.path.exists('H:/MBAR_GCMC/MBAR_values/Redo_low_tol/'+compound+'.txt') and not reprocess:
                print('Loading MBAR-GCMC analysis for '+compound)
                MBAR_GCMC_load = np.loadtxt('H:/MBAR_GCMC/MBAR_values/Redo_low_tol/'+compound+'.txt',skiprows=1)
                
                Tsat_MBAR = MBAR_GCMC_load[:,0]
                rhol_MBAR = MBAR_GCMC_load[:,1]
                rhov_MBAR = MBAR_GCMC_load[:,2]
                Psat_MBAR = MBAR_GCMC_load[:,3]
                DeltaHv_MBAR = MBAR_GCMC_load[:,4]
  
            else:
                print('Performing MBAR-GCMC analysis for '+compound)
                try:
                    MBAR_GCMC_analyze = MBAR_GCMC(root_path,filepaths,Mw,trim_data=trim_data,compare_literature=True)
            #        MBAR_GCMC_analyze.plot_histograms()
            #        MBAR_GCMC_analyze.plot_2dhistograms()
                    MBAR_GCMC_analyze.solve_VLE(Tsat_Potoff)
        #            MBAR_GCMC_analyze.plot_VLE([],[],[],Tsat_Potoff,rhol_Potoff,rhov_Potoff)
                    
                    Tsat_MBAR = MBAR_GCMC_analyze.Temp_VLE
                    rhol_MBAR = MBAR_GCMC_analyze.rholiq
                    rhov_MBAR = MBAR_GCMC_analyze.rhovap
                    Psat_MBAR = MBAR_GCMC_analyze.Psat
                    DeltaHv_MBAR = MBAR_GCMC_analyze.DeltaHv
                    
                    out_file =open('H:/MBAR_GCMC/MBAR_values/Redo_low_tol/'+compound+'.txt','w')
                    out_file.write('T (K) rhol (kg/m3) rhov (kg/m3) P (bar) DeltaHv (kJ/mol)'+'\n')
                    for Tsat, rhol, rhov, Psat, DeltaHv in zip(Tsat_Potoff,rhol_MBAR,rhov_MBAR,Psat_MBAR,DeltaHv_MBAR):
                        out_file.write(str(Tsat)+'\t'+str(rhol)+'\t'+str(rhov)+'\t'+str(Psat)+'\t'+str(DeltaHv)+'\n')
                    out_file.close()  
                                       
                except:
                    print('Error when performing MBAR-GCMC analysis for '+compound)
                    
            if False:#os.path.exists('H:/MBAR_GCMC/MBAR_values/'+compound+'_uncertainties.txt'):
                print('Analyzing uncertainties for '+compound)
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
                
                uZ_MBAR = np.zeros(len(urhol_MBAR))
                uDeltaUv_MBAR = np.zeros(len(urhol_MBAR))
                uPV_MBAR = np.zeros(len(urhol_MBAR))
                
                print('Assuming zero MBAR uncertainty for Z, DeltaU, and PV')
                
                assert len(Tsat_MBAR) == len(urhol_MBAR), 'Values and uncertainties dimensions mismatch'
                
                urhol = np.sqrt(urhol_Potoff**2. + urhol_MBAR**2.)
                urhov = np.sqrt(urhov_Potoff**2. + urhov_MBAR**2.)
                uPsat = np.sqrt(uPsat_Potoff**2. + uPsat_MBAR**2.)
                uDeltaHv = np.sqrt(uDeltaHv_Potoff**2. + uDeltaHv_MBAR**2.)
                uZ = np.sqrt(uZ_Potoff**2. + uZ_MBAR**2.)
                uDeltaUv = np.sqrt(uDeltaUv_Potoff**2. + uDeltaUv_MBAR**2.)
                uPV = np.sqrt(uPV_Potoff**2. + uPV_MBAR**2.)
                    
            ### Post-process values that can be derived
            Z_MBAR = Psat_MBAR * Mw / rhov_MBAR / Rg / Tsat_MBAR / 1000.
            
            Vl_MBAR = Mw / rhol_MBAR / 1000. #[m3/mol]
            Vv_MBAR = Mw / rhov_MBAR / 1000. #[m3/mol]
            
            PV_MBAR = Psat_MBAR * (Vv_MBAR - Vl_MBAR) / 1000. / 1000. #[kJ/mol]
            
            DeltaUv_MBAR = DeltaHv_MBAR - PV_MBAR #[kJ/mol]
            ### Only plot compounds with data and uncertainties
#            if os.path.exists('H:/MBAR_GCMC/MBAR_values/'+compound+'.txt') and  os.path.exists('H:/MBAR_GCMC/MBAR_values/'+compound+'_uncertainties.txt'):  
            
            ### Only plot compounds with data
            if os.path.exists('H:/MBAR_GCMC/MBAR_values/Redo_low_tol/'+compound+'.txt'):   
                Tr_Potoff = Tsat_Potoff/Tc_Potoff
                
                if scaled_dev:
                    
                    dev_rhol = np.abs(rhol_MBAR - rhol_Potoff)/urhol
                    dev_rhov = np.abs(rhov_MBAR - rhov_Potoff)/urhov
                    dev_Psat = np.abs(Psat_MBAR - Psat_Potoff)/uPsat
                    dev_DeltaHv = np.abs(DeltaHv_MBAR - DeltaHv_Potoff)/uDeltaHv
                    dev_Z = np.abs(Z_MBAR - Z_Potoff)/uZ
                    dev_DeltaUv = np.abs(DeltaUv_MBAR - DeltaUv_Potoff)/uDeltaUv
                    dev_PV = np.abs(PV_MBAR - PV_Potoff)/uPV
    
                    axarr[0,0].errorbar(Tr_Potoff[np.abs(dev_rhol)<5],dev_rhol[np.abs(dev_rhol)<5],fmt=color_list[icompound]+symbol_list[group]+line_list[icompound],mfc='None')
                    axarr[0,1].errorbar(Tr_Potoff[np.abs(dev_rhov)<10],dev_rhov[np.abs(dev_rhov)<10],fmt=color_list[icompound]+symbol_list[group]+line_list[icompound],mfc='None')
                    axarr[1,0].errorbar(Tr_Potoff[np.abs(dev_Psat)<20],dev_Psat[np.abs(dev_Psat)<20],fmt=color_list[icompound]+symbol_list[group]+line_list[icompound],mfc='None')
                    axarr[1,1].errorbar(Tr_Potoff[np.abs(dev_DeltaHv)<20],dev_DeltaHv[np.abs(dev_DeltaHv)<20],fmt=color_list[icompound]+symbol_list[group]+line_list[icompound],mfc='None')
                    axarr[2,0].errorbar(Tr_Potoff[np.abs(dev_Z)<20],dev_Z[np.abs(dev_Z)<20],fmt=color_list[icompound]+symbol_list[group]+line_list[icompound],mfc='None')
                    axarr[2,1].errorbar(Tr_Potoff[np.abs(dev_PV)<20],dev_PV[np.abs(dev_PV)<20],fmt=color_list[icompound]+symbol_list[group]+line_list[icompound],mfc='None')
    #                axarr[1,1].errorbar(Tr_Potoff[np.abs(dev_DeltaUv)<20],dev_DeltaUv[np.abs(dev_DeltaUv)<20],fmt=color_list[icompound]+symbol_list[group]+line_list[icompound],mfc='None')

                else:

                    dev_rhol = (rhol_MBAR - rhol_Potoff)/rhol_Potoff * 100.
                    dev_rhov = (rhov_MBAR - rhov_Potoff)/rhov_Potoff * 100.
                    dev_Psat = (Psat_MBAR - Psat_Potoff)/Psat_Potoff * 100.
                    dev_DeltaHv = (DeltaHv_MBAR - DeltaHv_Potoff)/DeltaHv_Potoff * 100.
                    dev_Z = (Z_MBAR - Z_Potoff)/Z_Potoff * 100.
                    dev_DeltaUv = (DeltaUv_MBAR - DeltaUv_Potoff)/DeltaUv_Potoff * 100.
                    dev_PV = (PV_MBAR - PV_Potoff)/PV_Potoff * 100.
                               
                    purhol_Potoff = urhol_Potoff / rhol_Potoff * 100.
                    purhov_Potoff = urhov_Potoff / rhov_Potoff * 100.
                    puPsat_Potoff = uPsat_Potoff / Psat_Potoff * 100.
                    puDeltaHv_Potoff = uDeltaHv_Potoff / DeltaHv_Potoff * 100.
                    puZ_Potoff = uZ_Potoff / Z_Potoff * 100.
                    puDeltaUv_Potoff = uDeltaUv_Potoff / DeltaUv_Potoff * 100.
                    puPV_Potoff = uPV_Potoff / PV_Potoff * 100.
                    
                    axarr[0,0].errorbar(Tr_Potoff[np.abs(dev_rhol)<5],dev_rhol[np.abs(dev_rhol)<5],yerr=purhol_Potoff[np.abs(dev_rhol)<5],fmt=color_list[icompound]+symbol_list[group]+line_list[icompound],mfc='None')
                    axarr[0,1].errorbar(Tr_Potoff[np.abs(dev_rhov)<10],dev_rhov[np.abs(dev_rhov)<10],yerr=purhov_Potoff[np.abs(dev_rhov)<10],fmt=color_list[icompound]+symbol_list[group]+line_list[icompound],mfc='None')
                    axarr[1,0].errorbar(Tr_Potoff[np.abs(dev_Psat)<20],dev_Psat[np.abs(dev_Psat)<20],yerr=puPsat_Potoff[np.abs(dev_Psat)<20],fmt=color_list[icompound]+symbol_list[group]+line_list[icompound],mfc='None')
                    axarr[1,1].errorbar(Tr_Potoff[np.abs(dev_DeltaHv)<20],dev_DeltaHv[np.abs(dev_DeltaHv)<20],yerr=puDeltaHv_Potoff[np.abs(dev_DeltaHv)<20],fmt=color_list[icompound]+symbol_list[group]+line_list[icompound],mfc='None')
                    axarr[2,0].errorbar(Tr_Potoff[np.abs(dev_Z)<20],dev_Z[np.abs(dev_Z)<20],yerr=puZ_Potoff[np.abs(dev_Z)<20],fmt=color_list[icompound]+symbol_list[group]+line_list[icompound],mfc='None')
                    axarr[2,1].errorbar(Tr_Potoff[np.abs(dev_PV)<20],dev_PV[np.abs(dev_PV)<20],yerr=puPV_Potoff[np.abs(dev_PV)<20],fmt=color_list[icompound]+symbol_list[group]+line_list[icompound],mfc='None')
    #                axarr[1,1].errorbar(Tr_Potoff[np.abs(dev_DeltaUv)<20],dev_DeltaUv[np.abs(dev_DeltaUv)<20],yerr=puDeltaUv_Potoff[np.abs(dev_DeltaUv)<20],fmt=color_list[icompound]+symbol_list[group]+line_list[icompound],mfc='None')
                        
                for Tr, dev_rhol_i, dev_rhov_i, dev_Psat_i, dev_DeltaHv_i, dev_Z_i, dev_PV_i, dev_DeltaUv_i in zip(Tr_Potoff,dev_rhol,dev_rhov,dev_Psat,dev_DeltaHv,dev_Z,dev_PV,dev_DeltaUv): 
                    
                    iBin = np.argmin(np.abs(Tr-Tr_bins))
                    
                    if np.abs(dev_rhol_i) < 5: dev_rhol_bins[Tr_bins[iBin]].append(dev_rhol_i)
                    if np.abs(dev_rhov_i) < 10: dev_rhov_bins[Tr_bins[iBin]].append(dev_rhov_i)
                    if np.abs(dev_Psat_i) < 20: dev_Psat_bins[Tr_bins[iBin]].append(dev_Psat_i)
                    if np.abs(dev_DeltaHv_i) < 20: dev_DeltaHv_bins[Tr_bins[iBin]].append(dev_DeltaHv_i)
                    if np.abs(dev_Z_i) < 20: dev_Z_bins[Tr_bins[iBin]].append(dev_Z_i)
                    if np.abs(dev_PV_i) < 20: dev_PV_bins[Tr_bins[iBin]].append(dev_PV_i)
                    if np.abs(dev_DeltaUv_i) < 20: dev_DeltaUv_bins[Tr_bins[iBin]].append(dev_DeltaUv_i)
                    
#                    if Tr <= 0.675:
#                        
#                        iBin
#                    
#                    idev_rhol in dev_rhol:
#                    
#                    if idev_rho

            icompound += 1
            
            if icompound >= ncompounds_max: #Break out of first for loop
                
                break
            
    if icompound >= ncompounds_max: #Break out of second for loop
                
                break
            
prop_list = [r'$\rho_{\rm liq}^{\rm sat}$',r'$\rho_{\rm vap}^{\rm sat}$',r'$P_{\rm vap}^{\rm sat}$',r'$\Delta H_{\rm v}$',r'$Z^{\rm sat}_{\rm vap}$',r'$P \Delta V$']  #r'$\Delta U_{\rm v}$'] #r'$P \Delta V$'] #r'$Z^{\rm sat}_{\rm vap}$']#r'$\Delta H_{\rm v}$']

if scaled_dev:
#    ylim_low = [-6,-6,-6,-6,-6,-6]
    ylim_low = [0,0,0,0,0,0]
    ylim_high = [6,6,6,6,6,6]
else:
    ylim_low = [-1,-4.5,-10,-5,-5,-5]
    ylim_high = [1,4.5,10,5,5,5]
        
for iax, prop in enumerate(prop_list):
    axarr[axarr_dic[iax]].set_xlabel(r'$T_{\rm r}$')
    
    if scaled_dev:
        axarr[axarr_dic[iax]].set_ylabel(r'$\frac{X_{\rm MBAR} - X_{\rm HR}}{\delta_X}$')  
    else:
        axarr[axarr_dic[iax]].set_ylabel(r'$\frac{X_{\rm MBAR} - X_{\rm HR}}{X_{\rm HR}} \times 100 \%$')
    axarr[axarr_dic[iax]].set_title(r'$X = $'+prop)
    axarr[axarr_dic[iax]].set_xlim([0.6,1.])
    axarr[axarr_dic[iax]].set_xticks([0.65,0.75,0.85,0.95])
    axarr[axarr_dic[iax]].set_ylim([ylim_low[iax],ylim_high[iax]])
    
#axarr[0,0].set_ylabel(r'$\frac{X_{\rm MBAR} - X_{\rm HR}}{X_{\rm HR}} \times 100 \%$')
#axarr[1,1].set_xlabel(r'$T_{\rm r}$')
#axarr[1,0].set_ylabel(r'$\frac{X_{\rm MBAR} - X_{\rm HR}}{X_{\rm HR}} \times 100 \%$')
#axarr[1,0].set_xlabel(r'$T_{\rm r}$')

plt.tight_layout(pad=0.5)

if scaled_dev:
    fig.savefig('Comparison_MBAR_HR_scaled_dev.pdf')
else:
    fig.savefig('Comparison_MBAR_HR.pdf')   

if scaled_dev:
    #ylim_low = [-6,-6,-6,-6,-6,-6]
    ylim_low = [0,0,0,0,0,0]
    ylim_high = [6,6,6,6,6,6]
else:        
    ylim_low = [-1.5,-6,-11,-5,-5,-5]
    ylim_high = [1.5,6,11,5,5,5]
    ylim_low = [-0.8,-2.2,-2,-1.5,-2,-1]
    ylim_high = [0.8,2.2,2,2,2,1]

fig, axarr = plt.subplots(nrows=3,ncols=2,figsize=[12,18])

boxplot_rhol = []
boxplot_rhov = []
boxplot_Psat = []
boxplot_DeltaHv = []
boxplot_Z = []
boxplot_PV = []
boxplot_DeltaUv = []

for Tr in Tr_bins:
    
    boxplot_rhol.append(dev_rhol_bins[Tr])
    boxplot_rhov.append(dev_rhov_bins[Tr])
    boxplot_Psat.append(dev_Psat_bins[Tr])
    boxplot_DeltaHv.append(dev_DeltaHv_bins[Tr])
    boxplot_Z.append(dev_Z_bins[Tr])
    boxplot_PV.append(dev_PV_bins[Tr])
    boxplot_DeltaUv.append(dev_DeltaUv_bins[Tr])

axarr[0,0].boxplot(boxplot_rhol,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False)
axarr[0,1].boxplot(boxplot_rhov,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False)
axarr[1,0].boxplot(boxplot_Psat,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False)
axarr[1,1].boxplot(boxplot_DeltaHv,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False)
axarr[2,0].boxplot(boxplot_Z,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False)
axarr[2,1].boxplot(boxplot_PV,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False)
#axarr[1,1].boxplot(boxplot_DeltaUv,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False)

for iax, prop in enumerate(prop_list):
    axarr[axarr_dic[iax]].set_xlabel(r'$T_{\rm r}$')
    
    if scaled_dev:
        axarr[axarr_dic[iax]].set_ylabel(r'$\frac{X_{\rm MBAR} - X_{\rm HR}}{\delta_X}$')  
    else:
        axarr[axarr_dic[iax]].set_ylabel(r'$\frac{X_{\rm MBAR} - X_{\rm HR}}{X_{\rm HR}} \times 100 \%$')
    axarr[axarr_dic[iax]].set_title(r'$X = $'+prop)
    axarr[axarr_dic[iax]].set_xlim([0.6,1.])
    axarr[axarr_dic[iax]].set_xticks([0.65,0.75,0.85,0.95])
    axarr[axarr_dic[iax]].set_ylim([ylim_low[iax],ylim_high[iax]])
        
#axarr[0].set_ylabel(r'$\frac{X_{\rm MBAR} - X_{\rm HR}}{X_{\rm HR}} \times 100 \%$')

plt.tight_layout(pad=0.5)

if scaled_dev:
    fig.savefig('Comparison_MBAR_HR_boxplot_scaled_dev.pdf')
else:
    fig.savefig('Comparison_MBAR_HR_boxplot.pdf')    

if scaled_dev: 
#    ylim_low = [-6,-6,-6,-6,-6,-6]
    ylim_low = [0,0,0,0,0,0]
#    ylim_high = [6,6,6,6,6,6]
#    ylim_low = [-2,-2,-2,-2,-2,-2]
    ylim_high = [2,2,2,2,2,2]
else:
    ylim_low = [-0.8,-2.2,-2,-1.5,-2,-1]
    ylim_high = [0.8,2.2,2,2,2,1]

fig, axarr = plt.subplots(nrows=2,ncols=2,figsize=[12,12])

prop_list = [r'$\rho_{\rm liq}^{\rm sat}$',r'$\rho_{\rm vap}^{\rm sat}$',r'$P_{\rm vap}^{\rm sat}$',r'$\Delta H_{\rm v}$']#,r'$Z^{\rm sat}_{\rm vap}$',r'$P \Delta V$']  #r'$\Delta U_{\rm v}$'] #r'$P \Delta V$'] #r'$Z^{\rm sat}_{\rm vap}$']#r'$\Delta H_{\rm v}$']

axarr[0,0].boxplot(boxplot_rhol,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False,whis=[5,95],sym='')
axarr[0,1].boxplot(boxplot_rhov,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False,whis=[5,95],sym='')
axarr[1,0].boxplot(boxplot_Psat,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False,whis=[5,95],sym='')
axarr[1,1].boxplot(boxplot_DeltaHv,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False,whis=[5,95],sym='')
#axarr[2,0].boxplot(boxplot_Z,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False,whis=[5,95],sym='')
#axarr[2,1].boxplot(boxplot_PV,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False,whis=[5,95],sym='')
#axarr[1,1].boxplot(boxplot_DeltaUv,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False,whis=[5,95],sym='')

for iax, prop in enumerate(prop_list):
    axarr[axarr_dic[iax]].set_xlabel(r'$T_{\rm r}$')
    
    if scaled_dev:
        axarr[axarr_dic[iax]].set_ylabel(r'$\frac{|X_{\rm MBAR} - X_{\rm HR}|}{\delta_X}$')  
    else:
        axarr[axarr_dic[iax]].set_ylabel(r'$\frac{X_{\rm MBAR} - X_{\rm HR}}{X_{\rm HR}} \times 100$')
    axarr[axarr_dic[iax]].set_title(r'$X = $'+prop,y=1.02)
    axarr[axarr_dic[iax]].set_xlim([0.6,1.])
    axarr[axarr_dic[iax]].set_xticks([0.65,0.75,0.85,0.95])
    axarr[axarr_dic[iax]].set_ylim([ylim_low[iax],ylim_high[iax]])
        
#axarr[0].set_ylabel(r'$\frac{X_{\rm MBAR} - X_{\rm HR}}{X_{\rm HR}} \times 100 \%$')

plt.tight_layout(pad=0.5)

from scipy.special import erfinv

Q1 = np.sqrt(2)*erfinv(0.25)/1.96
median = np.sqrt(2)*erfinv(0.5)/1.96
Q3 = np.sqrt(2)*erfinv(0.75)/1.96
mean = np.sqrt(2)/np.sqrt(np.pi)/1.96

#for iax, prop in enumerate(prop_list):
#              
##    axarr[axarr_dic[iax]].plot([0.65,0.95],[mean,mean],'k-')
#    axarr[axarr_dic[iax]].plot([0.65,0.95],[Q1,Q1],'r--')
#    axarr[axarr_dic[iax]].plot([0.65,0.95],[Q3,Q3],'r--')
#    axarr[axarr_dic[iax]].plot([0.65,0.95],[median,median],'r-')

if scaled_dev:
    fig.savefig('Comparison_MBAR_HR_boxplot_CI_scaled_dev.pdf')
else:
    fig.savefig('Comparison_MBAR_HR_boxplot_CI.pdf')        