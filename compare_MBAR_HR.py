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
ncompounds_max = 44
icompound = 0

reprocess = False

group_list = ['C-CH-group','C-group','CH-group','alkynes']

compound_list = {'C-CH-group':['223TMButane','223TMPentane','224TMPentane','233TMPentane'],'C-group':['3M3EPentane','22DMButane','22DMHexane','22DMPentane','22DMPropane','33DMHexane','33DMPentane','2233TetraMButane'],'CH-group':['2M3EPentane','2MButane','2MHeptane','2MHexane','2MPentane','2MPropane','3EHexane','3EPentane','3MHeptane','3MHexane','3MPentane','4MHeptane','23DMBUtane','23DMHexane','23DMPentane','24DMHexane','24DMPentane','25DMHexane','34DMHexane','234TMPentane'],'alkynes':['ethyne','propyne','1butyne','2butyne','1pentyne','2pentyne','1hexyne','2hexyne','1heptyne','1octyne','1nonyne','propadiene']}  #{'C-group':['22DMPropane'],'C-CH-group':['224TMPentane']}

Mw_list = {'223TMButane':100.205,'223TMPentane':114.22852,'224TMPentane':114.22852,'233TMPentane':114.22852,'3M3EPentane':114.22852,'22DMButane':86.17536,'22DMHexane':114.22852,'22DMPentane':100.205,'22DMPropane':72.14878,'33DMHexane':114.22852,'33DMPentane':100.205,'2233TetraMButane':114.22852,'2M3EPentane':114.22852,'2MButane':72.14878,'2MHeptane':114.22852,'2MHexane':100.205,'2MPentane':86.17536,'2MPropane':58.1222,'3EHexane':114.22852,'3EPentane':100.205,'3MHeptane':114.22852,'3MHexane':100.205,'3MPentane':86.17536,'4MHeptane':114.22852,'23DMBUtane':86.17536,'23DMHexane':114.22852,'23DMPentane':100.205,'24DMHexane':114.22852,'24DMPentane':100.205,'25DMHexane':114.22852,'34DMHexane':114.22852,'234TMPentane':114.22852,'ethyne':26.04,'propyne':40.0639,'1butyne':54.091,'2butyne':54.091,'1pentyne':68.12,'2pentyne':68.12,'1hexyne':82.14,'2hexyne':82.14,'1heptyne':96.17,'1octyne':110.2,'1nonyne':124.22,'propadiene':40.06} #[gm/mol]

color_list = ['b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y']
symbol_list = {'C-CH-group':'o','C-group':'s','CH-group':'^','alkynes':'d'}
line_list = ['-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.']

directory_dic = {'C-CH-group':'branched-Alkanes/C-CH-group/','C-group':'branched-Alkanes/C-group/','CH-group':'branched-Alkanes/CH-group/','alkynes':'alkynes/'}

dev_rhol_bins = {0.65:[],0.7:[],0.75:[],0.8:[],0.85:[],0.9:[],0.95:[]}
dev_rhov_bins = {0.65:[],0.7:[],0.75:[],0.8:[],0.85:[],0.9:[],0.95:[]}
dev_Psat_bins = {0.65:[],0.7:[],0.75:[],0.8:[],0.85:[],0.9:[],0.95:[]}
Tr_bins = np.array([0.65,0.7,0.75,0.8,0.85,0.9,0.95])

fig, axarr = plt.subplots(nrows=1,ncols=3,figsize=[18,6])

for group in group_list:
       
    for compound in compound_list[group]: # ['224-trimethylpentane','neopentane','isobutane']:
                
            filepaths = []
            
            root_path = 'H:/Mie-swf/histFiles/'+directory_dic[group]+compound
            
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
    
            if not os.path.exists('H:/MBAR_GCMC/Potoff_literature/'+compound+'_Tc.txt'):
    
                if Tsat_Potoff[0] < Tsat_Potoff[-1]:
                    ###Have to make sure that Tsat[0] is the highest value since this code was written for ITIC
                    Tsat_Potoff = Tsat_Potoff[::-1]
                    rhol_Potoff = rhol_Potoff[::-1]
                    rhov_Potoff = rhov_Potoff[::-1]
                    Psat_Potoff = Psat_Potoff[::-1]
            
                Potoff_fit = ITIC_VLE(Tsat_Potoff,rhol_Potoff,rhov_Potoff,Psat_Potoff)
                Tc_Potoff = Potoff_fit.Tc
                
                out_file =open('H:/MBAR_GCMC/Potoff_literature/'+compound+'_Tc.txt','w')
                out_file.write(str(Tc_Potoff))
                out_file.close()
                
            Tc_Potoff = np.loadtxt('H:/MBAR_GCMC/Potoff_literature/'+compound+'_Tc.txt')       
                                                              
            Mw = Mw_list[compound]
            
            if os.path.exists('H:/MBAR_GCMC/MBAR_values/'+compound+'.txt') and not reprocess:
                print('Loading MBAR-GCMC analysis for '+compound)
                MBAR_GCMC_load = np.loadtxt('H:/MBAR_GCMC/MBAR_values/'+compound+'.txt',skiprows=1)
                
                rhol_MBAR = MBAR_GCMC_load[:,1]
                rhov_MBAR = MBAR_GCMC_load[:,2]
                Psat_MBAR = MBAR_GCMC_load[:,3]
  
            else:
                print('Performing MBAR-GCMC analysis for '+compound)
                try:
                    MBAR_GCMC_analyze = MBAR_GCMC(root_path,filepaths,Mw,trim_data=False,compare_literature=True)
            #        MBAR_GCMC_analyze.plot_histograms()
            #        MBAR_GCMC_analyze.plot_2dhistograms()
                    MBAR_GCMC_analyze.solve_VLE(Tsat_Potoff)
        #            MBAR_GCMC_analyze.plot_VLE([],[],[],Tsat_Potoff,rhol_Potoff,rhov_Potoff)
                    
                    rhol_MBAR = MBAR_GCMC_analyze.rholiq
                    rhov_MBAR = MBAR_GCMC_analyze.rhovap
                    Psat_MBAR = MBAR_GCMC_analyze.Psat
                    
                    out_file =open('H:/MBAR_GCMC/MBAR_values/'+compound+'.txt','w')
                    out_file.write('T (K) rhol (kg/m3) rhov (kg/m3) P (bar)'+'\n')
                    for Tsat, rhol, rhov, Psat in zip(Tsat_Potoff,rhol_MBAR,rhov_MBAR,Psat_MBAR):
                        out_file.write(str(Tsat)+'\t'+str(rhol)+'\t'+str(rhov)+'\t'+str(Psat)+'\n')
                    out_file.close()  
                                       
                except:
                    print('Error when performing MBAR-GCMC analysis for '+compound)

            if os.path.exists('H:/MBAR_GCMC/MBAR_values/'+compound+'.txt'):            

                dev_rhol = (rhol_MBAR - rhol_Potoff)/rhol_Potoff * 100.
                dev_rhov = (rhov_MBAR - rhov_Potoff)/rhov_Potoff * 100.
                dev_Psat = (Psat_MBAR - Psat_Potoff)/Psat_Potoff * 100.
                           
                purhol_Potoff = urhol_Potoff / rhol_Potoff * 100.
                purhov_Potoff = urhov_Potoff / rhov_Potoff * 100.
                puPsat_Potoff = uPsat_Potoff / Psat_Potoff * 100.
                
                Tr_Potoff = Tsat_Potoff/Tc_Potoff
                           
                axarr[0].errorbar(Tr_Potoff[np.abs(dev_rhol)<5],dev_rhol[np.abs(dev_rhol)<5],yerr=purhol_Potoff[np.abs(dev_rhol)<5],fmt=color_list[icompound]+symbol_list[group]+line_list[icompound],mfc='None')
                axarr[1].errorbar(Tr_Potoff[np.abs(dev_rhov)<10],dev_rhov[np.abs(dev_rhov)<10],yerr=purhov_Potoff[np.abs(dev_rhov)<10],fmt=color_list[icompound]+symbol_list[group]+line_list[icompound],mfc='None')
                axarr[2].errorbar(Tr_Potoff[np.abs(dev_Psat)<20],dev_Psat[np.abs(dev_Psat)<20],yerr=puPsat_Potoff[np.abs(dev_Psat)<20],fmt=color_list[icompound]+symbol_list[group]+line_list[icompound],mfc='None')

#                for iTr, Tr in enumerate(Tr_Potoff): 
#                    
#                    iBin = np.argmin(np.abs(Tr-Tr_bins))
#                    
#                    if dev_rhol[iTr] dev_rhol_bins[Tr_bins[iBin]].append(dev_rhol[iTr]):
                        
                for Tr, dev_rhol_i, dev_rhov_i, dev_Psat_i in zip(Tr_Potoff,dev_rhol,dev_rhov,dev_Psat): 
                    
                    iBin = np.argmin(np.abs(Tr-Tr_bins))
                    
                    if np.abs(dev_rhol_i) < 5: dev_rhol_bins[Tr_bins[iBin]].append(dev_rhol_i)
                    if np.abs(dev_rhov_i) < 10: dev_rhov_bins[Tr_bins[iBin]].append(dev_rhov_i)
                    if np.abs(dev_Psat_i) < 20: dev_Psat_bins[Tr_bins[iBin]].append(dev_Psat_i)
                    
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
            
prop_list = [r'$\rho_{\rm liq}^{\rm sat}$',r'$\rho_{\rm vap}^{\rm sat}$',r'$P_{\rm vap}^{\rm sat}$']
ylim_low = [-1,-4.5,-10]
ylim_high = [1,4.5,10]
        
for iax, prop in enumerate(prop_list):
    axarr[iax].set_xlabel(r'$T_{\rm r}$')
#    axarr[iax].set_ylabel()
    axarr[iax].set_title(r'X = '+prop)
    axarr[iax].set_xlim([0.6,1.])
    axarr[iax].set_xticks([0.65,0.75,0.85,0.95])
    axarr[iax].set_ylim([ylim_low[iax],ylim_high[iax]])
    
axarr[0].set_ylabel(r'$\frac{X_{\rm MBAR} - X_{\rm HR}}{X_{\rm HR}} \times 100 \%$')

plt.tight_layout(pad=0)

fig.savefig('Comparison_MBAR_HR.pdf')
        
#        MBAR_GCMC_trial.eps_scan(Tsat_Potoff,rhol_Potoff,rhov_Potoff,rhol_Potoff,rhov_Potoff,1.0,1.0,1,compound,remove_low_high_Tsat=True)

ylim_low = [-1.5,-6,-11]
ylim_high = [1.5,6,11]

fig, axarr = plt.subplots(nrows=1,ncols=3,figsize=[18,6])

boxplot_rhol = []
boxplot_rhov = []
boxplot_Psat = []

for Tr in Tr_bins:
    
    boxplot_rhol.append(dev_rhol_bins[Tr])
    boxplot_rhov.append(dev_rhov_bins[Tr])
    boxplot_Psat.append(dev_Psat_bins[Tr])

axarr[0].boxplot(boxplot_rhol,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False)
axarr[1].boxplot(boxplot_rhov,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False)
axarr[2].boxplot(boxplot_Psat,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False)

for iax, prop in enumerate(prop_list):
    axarr[iax].set_xlabel(r'$T_{\rm r}$')
#    axarr[iax].set_ylabel()
    axarr[iax].set_title(r'X = '+prop)
    axarr[iax].set_xlim([0.6,1.])
    axarr[iax].set_xticks([0.65,0.75,0.85,0.95])
    axarr[iax].set_ylim([ylim_low[iax],ylim_high[iax]])
        
axarr[0].set_ylabel(r'$\frac{X_{\rm MBAR} - X_{\rm HR}}{X_{\rm HR}} \times 100 \%$')

plt.tight_layout(pad=0)

fig.savefig('Comparison_MBAR_HR_boxplot.pdf')    

ylim_low = [-0.8,-2.2,-7]
ylim_high = [0.8,2.2,7]

fig, axarr = plt.subplots(nrows=1,ncols=3,figsize=[18,6])

axarr[0].boxplot(boxplot_rhol,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False,whis=[5,95],sym='')
axarr[1].boxplot(boxplot_rhov,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False,whis=[5,95],sym='')
axarr[2].boxplot(boxplot_Psat,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False,whis=[5,95],sym='')

for iax, prop in enumerate(prop_list):
    axarr[iax].set_xlabel(r'$T_{\rm r}$')
#    axarr[iax].set_ylabel()
    axarr[iax].set_title(r'X = '+prop)
    axarr[iax].set_xlim([0.6,1.])
    axarr[iax].set_xticks([0.65,0.75,0.85,0.95])
    axarr[iax].set_ylim([ylim_low[iax],ylim_high[iax]])
        
axarr[0].set_ylabel(r'$\frac{X_{\rm MBAR} - X_{\rm HR}}{X_{\rm HR}} \times 100 \%$')

plt.tight_layout(pad=0)

fig.savefig('Comparison_MBAR_HR_boxplot_CI.pdf')        