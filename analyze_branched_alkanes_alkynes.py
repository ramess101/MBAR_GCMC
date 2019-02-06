# -*- coding: utf-8 -*-
"""
Determines the optimal epsilon scaling for branched alkanes and alkynes
Reads in the GCMC snapshots that Mohammad provided
Plots the GOLDEN search optimization results in figure
"""

from __future__ import division
import numpy as np
from optimize_epsilon import MBAR_GCMC
import CoolProp.CoolProp as CP
import os.path
from extract_Potoff import extract_Potoff
import matplotlib.pyplot as plt

font = {'size' : '24'}
plt.rc('font',**font)

nhists_max = 15

group_list = ['CH-group','C-group','C-CH-group','alkynes']
compound_list = {'C-CH-group':['223TMButane','223TMPentane','224TMPentane','233TMPentane'],'C-group':['22DMPropane','3M3EPentane','22DMButane','22DMHexane','22DMPentane','33DMHexane','33DMPentane','2233TetraMButane'],'CH-group':['2MPropane','2M3EPentane','2MButane','2MHeptane','2MHexane','2MPentane','3MPentane','3EHexane','3EPentane','3MHeptane','3MHexane','4MHeptane','23DMButane','23DMHexane','23DMPentane','24DMHexane','24DMPentane','25DMHexane','34DMHexane','234TMPentane'],'alkynes':['ethyne','propyne','1butyne','2butyne','1pentyne','2pentyne','1hexyne','2hexyne','1heptyne','1octyne','1nonyne']}  #{'C-group':['22DMPropane'],'C-CH-group':['224TMPentane']}
#REFPROP_list = {'224TMPentane':'Isooctane','2MPropane':'Isobutane','22DMPropane':'Neopentane','2MButane':'IPENTANE','2MPentane':'IHEXANE','1butyne':'1-Butyne','ethyne':'Acetylene','propyne':'Propyne'}#,'3MPentane':'3METHYLPENTANE','22DMButane':'22DIMETHYLBUTANE','23DMButane':'23DIMETHYLBUTANE'}
REFPROP_list = {'3MPentane':'','22DMButane':'','23DMButane':'','224TMPentane':'Isooctane','2MPropane':'Isobutane','22DMPropane':'Neopentane','2MButane':'IPENTANE','2MPentane':'IHEXANE','1butyne':'1-Butyne','ethyne':'Acetylene','propyne':'Propyne'}#,'3MPentane':'3METHYLPENTANE','22DMButane':'22DIMETHYLBUTANE','23DMButane':'23DIMETHYLBUTANE'}

directory_dic = {'C-CH-group':'branched-Alkanes/C-CH-group/','C-group':'branched-Alkanes/C-group/','CH-group':'branched-Alkanes/CH-group/','alkynes':'alkynes/'}
experimental_dic = {'C-CH-group':'REFPROP_values','C-group':'REFPROP_values','CH-group':'REFPROP_values','alkynes':'DIPPR_values'}

referenceFF_directory_dic = {'Potoff_SL':'Optimized/','Potoff_gen':'Generalized/','TraPPE':'TraPPE/'}

eps_low_dic = {'3MPentane':1.00,'2butyne':0.995,'1hexyne':1.,'1butyne':1.,'1pentyne':0.995,'2pentyne':1.,'2hexyne':1.,'1heptyne':0.995,'1octyne':0.995}
eps_high_dic = {'3MPentane':1.015,'2butyne':1.01,'1hexyne':1.02,'1butyne':1.02,'1pentyne':1.015,'2pentyne':1.03,'2hexyne':1.03,'1heptyne':1.015,'1octyne':1.015}
nScan_dic = {'3MPentane':30,'2butyne':25,'1hexyne':20,'1butyne':20,'1pentyne':20,'2pentyne':30,'2hexyne':30,'1heptyne':20,'1octyne':20}

reprocess = False
trim_data = False
remove_low_high_Tsat = False
bootstrap = False
nBoots = 50

opt_or_scan = 'scan'

referenceFF = 'Potoff_SL'

if reprocess:
    eps_opt = {}
    Score_opt = {}
    Score_Potoff = {}
    eps_computed = {}
    Score_computed = {}

for group in ['CH-group']:# ['alkynes']:# ['C-CH-group','C-group','CH-group','alkynes']:
       
    for compound in ['3MPentane']:# ['propyne','1butyne']:# ['2butyne','1pentyne','2pentyne','2hexyne','1heptyne','1octyne','1nonyne']: # compound_list[group]:
        
        filepaths = []
        
        if group == 'alkynes':   ###Alkynes only have Potoff results     
            root_path = 'H:/GCMC_histFiles/'+directory_dic[group]+compound
        else: ### Branched alkanes have three different force field options
#            root_path = 'H:/GCMC_histFiles/'+directory_dic[group]+referenceFF_directory_dic[referenceFF]+compound        
            root_path = 'H:/GCMC_histFiles/branched-Alkanes/'+referenceFF_directory_dic[referenceFF]+group+'/'+compound        
                                                           
        for iT in np.arange(1,nhists_max+1):
                        
            hist_name='/his'+str(iT)+'a.dat'
                
            if os.path.exists(root_path+hist_name):
                
                filepaths.append(root_path+hist_name)
        
        if not os.path.exists('H:/MBAR_GCMC/Potoff_literature/'+compound+'.txt'):
            
            input_path = 'H:/MBAR_GCMC/Potoff_literature/'+compound+'_rawdata.txt'
            output_path = 'H:/MBAR_GCMC/Potoff_literature/'+compound+'.txt'
            
            extract_Potoff(input_path,output_path)
            
        VLE_Potoff = np.loadtxt('H:/MBAR_GCMC/Potoff_literature/'+compound+'.txt',skiprows=1)
            
        Tsat_Potoff = VLE_Potoff[:,0] #[K]
        rhol_Potoff = VLE_Potoff[:,1]*1000. #[kg/m3]
        rhov_Potoff = VLE_Potoff[:,2]*1000. #[kg/m3]
        Psat_Potoff = VLE_Potoff[:,3] #[bar]
        DeltaHv_Potoff = VLE_Potoff[:,4] #[kJ/mol]
        
        if compound in REFPROP_list or group == 'alkynes':
            
            try:
            
                VLE_RP = np.loadtxt('H:/MBAR_GCMC/Experimental_values/'+experimental_dic[group]+'/VLE_'+compound+'.txt',skiprows=1)
                
                Tsat_RP = VLE_RP[:,0] #[K]
                rhol_RP = VLE_RP[:,1] #[kg/m3]
                rhov_RP = VLE_RP[:,2] #[kg/m3]
                Psat_RP = VLE_RP[:,3] #[bar]
                DeltaHv_RP = VLE_RP[:,4] #[kJ/mol]
                
                Mw = np.loadtxt('H:/MBAR_GCMC/Experimental_values/MW_'+compound+'.txt')
                
            except:
                
                Mw = CP.PropsSI('M','REFPROP::'+REFPROP_list[compound])*1000. #[gm/mol]
                            
                Tsat_RP = Tsat_Potoff.copy()
                rhol_RP = CP.PropsSI('D','T',Tsat_RP,'Q',0,'REFPROP::'+REFPROP_list[compound]) #[kg/m3]   
                rhov_RP = CP.PropsSI('D','T',Tsat_RP,'Q',1,'REFPROP::'+REFPROP_list[compound]) #[kg/m3] 
                Psat_RP = CP.PropsSI('P','T',Tsat_RP,'Q',1,'REFPROP::'+REFPROP_list[compound])/100000. #[bar]
                
                f1 = open('H:/MBAR_GCMC/Experimental_values/REFPROP_values/VLE_'+compound+'.txt','w')
                f1.write('Tsat (K) rhol (kg/m3) rhov (kg/m3) Psat (bar)'+'\n')
                for Tsat, rhol, rhov, Psat in zip(Tsat_RP,rhol_RP,rhov_RP,Psat_RP):
                    f1.write(str(Tsat)+'\t')
                    f1.write(str(rhol)+'\t')
                    f1.write(str(rhov)+'\t')   
                    f1.write(str(Psat)+'\n') 
                f1.close()
                
                f2 = open('H:/MBAR_GCMC/Experimental_values/REFPROP_values/MW_'+compound+'.txt','w')
                f2.write(str(Mw))
                f2.close()
                
        else:
            
            print(compound+' not in REFPROP database')
            Tsat_RP = []
            rhol_RP = []
            rhov_RP = []
            Psat_RP = []
            DeltaHv_RP = []
            
            VLE_Potoff = np.loadtxt('H:/MBAR_GCMC/Potoff_literature/'+compound+'_withuncertainties.txt',skiprows=1)
                
            Tsat_RP = VLE_Potoff[:,0] #[K]
            rhol_RP = VLE_Potoff[:,1]*1000. #[kg/m3]
            rhov_RP = VLE_Potoff[:,3]*1000. #[kg/m3]
            Psat_RP = VLE_Potoff[:,5] #[bar]
            DeltaHv_RP = VLE_Potoff[:,7] #[kJ/mol]
            
        if experimental_dic[group] == 'DIPPR_values':
            
            rhov_RP = np.ones(len(rhov_RP)) #Avoid divide by zero
                
#        MBAR_GCMC_trial = MBAR_GCMC(root_path,filepaths,Mw,trim_data=True,compare_literature=True)
#        MBAR_GCMC_trial.plot_histograms()
#        MBAR_GCMC_trial.plot_2dhistograms()
#        MBAR_GCMC_trial.solve_VLE(Tsat_Potoff)
#        MBAR_GCMC_trial.plot_VLE(Tsat_RP,rhol_RP,rhov_RP,Tsat_Potoff,rhol_Potoff,rhov_Potoff)
        
        if (compound in REFPROP_list or group == 'alkynes') and reprocess: # and not os.path.exists('H:/MBAR_GCMC/Epsilon_scaling/'+referenceFF+'/'+compound+'.txt'):# and not os.path.exists('H:/MBAR_GCMC/figures/'+compound+'_AD_eps_scan.pdf'):
            
            print('Performing MBAR-GCMC analysis for '+compound)
            
            MBAR_GCMC_trial = MBAR_GCMC(root_path,filepaths,Mw,trim_data=trim_data,compare_literature=True)
#            MBAR_GCMC_trial.plot_histograms()
#            MBAR_GCMC_trial.plot_2dhistograms()
            MBAR_GCMC_trial.solve_VLE(Tsat_Potoff)
            MBAR_GCMC_trial.plot_VLE(Tsat_RP,rhol_RP,rhov_RP,Tsat_Potoff,rhol_Potoff,rhov_Potoff)
            
#            urholiq_MBAR, urhovap_MBAR, uPsat_MBAR, uDeltaHv_MBAR = MBAR_GCMC_trial.VLE_uncertainty(Tsat_Potoff)

            if opt_or_scan == 'opt':

                print('Performing Golden search optimization')
                
                MBAR_GCMC_trial.eps_optimize(Tsat_RP,rhol_RP,rhov_RP,Psat_RP,DeltaHv_RP,0.99,1.01,compound,remove_low_high_Tsat=remove_low_high_Tsat)
                
                if MBAR_GCMC_trial.eps_opt > 1.008: MBAR_GCMC_trial.eps_optimize(Tsat_RP,rhol_RP,rhov_RP,Psat_RP,DeltaHv_RP,0.99,1.03,compound,remove_low_high_Tsat=remove_low_high_Tsat)
                if MBAR_GCMC_trial.eps_opt < 0.992: MBAR_GCMC_trial.eps_optimize(Tsat_RP,rhol_RP,rhov_RP,Psat_RP,DeltaHv_RP,0.97,1.01,compound,remove_low_high_Tsat=remove_low_high_Tsat)
                
            elif opt_or_scan == 'scan':
                
                print('Performing scan')
                
                try: 
                    
                    eps_low = eps_low_dic[compound]
                    eps_high = eps_high_dic[compound]
                    nScan = nScan_dic[compound]
                    
                except:
                    
                    eps_low = 0.995
                    eps_high = 1.005
                    nScan = 21
        
                MBAR_GCMC_trial.eps_scan(Tsat_RP,rhol_RP,rhov_RP,Psat_RP,DeltaHv_RP,eps_low,eps_high,nScan,compound,remove_low_high_Tsat=True)
            
            else:
                
                print('Optimization or scan must be specified')
            
            eps_opt[compound] = MBAR_GCMC_trial.eps_opt
            Score_opt[compound] = MBAR_GCMC_trial.Score_opt
#            Score_Potoff[compound] = MBAR_GCMC_trial.Score_Potoff
            eps_computed[compound] = MBAR_GCMC_trial.eps_computed
            Score_computed[compound] = MBAR_GCMC_trial.Score_computed
                          
            out_file =open('H:/MBAR_GCMC/Epsilon_scaling/'+referenceFF+'/'+compound+'_'+opt_or_scan+'.txt','w')
            out_file.write('scaling_factor Scoring_function'+'\n')
            for scaling_factor, Score in zip(eps_computed[compound],Score_computed[compound]):
                out_file.write(str(scaling_factor)+'\t'+str(Score)+'\n')
            out_file.close()  
            
        if bootstrap:
            
            print('Performing bootstrap analysis for '+compound)
            
            MBAR_GCMC_trial = MBAR_GCMC(root_path,filepaths,Mw,trim_data=trim_data,compare_literature=True)
#            MBAR_GCMC_trial.plot_histograms()
#            MBAR_GCMC_trial.plot_2dhistograms()
            urholiq_ref, urhovap_ref, uPsat_ref, uDeltaHv_ref, Score_low95_ref, Score_high95_ref = MBAR_GCMC_trial.Score_uncertainty(Tsat_RP,rhol_RP,rhov_RP,Psat_RP,DeltaHv_RP,remove_low_high_Tsat=remove_low_high_Tsat,eps_scaled=1.,nBoots=nBoots)
            ScoreBoots_ref = np.array(MBAR_GCMC_trial.Score_computed)

            out_file =open('H:/MBAR_GCMC/Epsilon_scaling/'+referenceFF+'/'+compound+'_ScoreBoots_ref.txt','w')
            out_file.write('Scoring_function'+'\n')
            for Score in ScoreBoots_ref:
                out_file.write(str(Score)+'\n')
            out_file.close()
            
            out_file =open('H:/MBAR_GCMC/Epsilon_scaling/'+referenceFF+'/'+compound+'_uScore_ref.txt','w')
            out_file.write('Score_low95'+'\t'+'Score_high95'+'\n')
            out_file.write(str(Score_low95_ref)+'\t'+str(Score_high95_ref))
            out_file.close()
            
            out_file =open('H:/MBAR_GCMC/MBAR_values/'+compound+'_uncertainties.txt','w')
            out_file.write('T (K) urhol (kg/m3) urhov (kg/m3) uPsat (bar) uDeltaHv (kJ/mol)'+'\n')
            for Tsat, urhol, urhov, uPsat, uDeltaHv in zip(Tsat_RP,urholiq_ref,urhovap_ref,uPsat_ref,uDeltaHv_ref):
                out_file.write(str(Tsat)+'\t'+str(urhol)+'\t'+str(urhov)+'\t'+str(uPsat)+'\t'+str(uDeltaHv)+'\n')
            out_file.close()  

#            urholiq_opt, urhovap_opt, uPsat_opt, uDeltaHv_opt, Score_low95_opt, Score_high95_opt = MBAR_GCMC_trial.Score_uncertainty(Tsat_RP,rhol_RP,rhov_RP,Psat_RP,DeltaHv_RP,remove_low_high_Tsat=remove_low_high_Tsat,eps_scaled=eps_opt[compound],nBoots=nBoots)
#            ScoreBoots_opt = np.array(MBAR_GCMC_trial.Score_computed)
#            
#            out_file =open('H:/MBAR_GCMC/Epsilon_scaling/'+referenceFF+'/'+compound+'_ScoreBoots_opt.txt','w')
#            out_file.write('Scoring_function'+'\n')
#            for Score in ScoreBoots_opt:
#                out_file.write(str(Score)+'\n')
#            out_file.close()
#            
#            out_file =open('H:/MBAR_GCMC/Epsilon_scaling/'+referenceFF+'/'+compound+'_uScore_opt.txt','w')
#            out_file.write('Score_low95'+'\t'+'Score_high95'+'\n')
#            out_file.write(str(Score_low95_opt)+'\t'+str(Score_high95_opt))
#            out_file.close()
                          
### Plot the results for different families and reference force fields:
    
color_list = ['b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y','b','r','g','c','m','y']
symbol_list = ['o','s','d','^','v','<','>','o','s','d','^','v','<','>','o','s','d','^','v','<','>','o','s','d','^','v','<','>','o','s','d','^','v','<','>','o','s','d','^','v','<','>','o','s','d','^','v','<','>','o','s','d','^','v','<','>','o','s','d','^','v','<','>','o','s','d','^','v','<','>','o','s','d','^','v','<','>','o','s','d','^','v','<','>']
line_list = ['-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.','-','--',':','-.']
#axarr_dic = {0:(0,0),1:(0,1),2:(1,0),3:(1,1)}
#axarr_dic = {'C-CH-group':{'Potoff_SL':(0,0),'Potoff_Gen':(1,0),'TraPPE':(1,1)},'C-group':{'Potoff_SL':(0,0),'Potoff_Gen':(1,0),'TraPPE':(1,1)},'CH-group':{'Potoff_SL':(0,0),'Potoff_Gen':(1,0),'TraPPE':(1,1)},'alkynes':{'Potoff_SL':(0,1),'Potoff_Gen':(0,1),'TraPPE':(0,1)}}
axarr_dic = {'C-CH-group':{'Potoff_SL':(0)},'C-group':{'Potoff_SL':(0)},'CH-group':{'Potoff_SL':(0)},'alkynes':{'Potoff_SL':(1)}}
#title_list = {'C-CH-group','C-group','CH-group','alkynes'}
#title_list = {'Branched alkanes','Alkynes','Branched alkanes','Branched alkanes'}
#title_list = [r'$\epsilon_{\rm ref}=\epsilon_{\rm Potoff, S/L}$',r'$\epsilon_{\rm ref}=\epsilon_{\rm Potoff, S/L}$',r'$\epsilon_{\rm ref}=\epsilon_{\rm Potoff, gen.}$',r'$\epsilon_{\rm ref}=\epsilon_{\rm TraPPE}$']
title_list = [r'$\theta_{\rm ref}=\theta_{\rm Potoff, S/L}$',r'$\theta_{\rm ref}=\theta_{\rm Potoff}$',r'$\theta_{\rm ref}=\theta_{\rm Potoff, gen.}$',r'$\theta_{\rm ref}=\theta_{\rm TraPPE}$']
#xlabel_list = [r'$\psi \equiv \epsilon / \epsilon_{\rm Potoff, S/L}$',r'$\psi \equiv \epsilon / \epsilon_{\rm Potoff, S/L}$',r'$\psi \equiv \epsilon / \epsilon_{\rm Potoff, gen.}$',r'$\psi \equiv \epsilon / \epsilon_{\rm TraPPE}$']
xlabel_list = [r'$\psi$',r'$\psi$',r'$\psi$',r'$\psi$']
label_dic = {'224TMPentane':'224TMC$_5$','22DMButane':'22DMC$_4$','22DMPropane':'22DMC$_3$','2MButane':'2MC$_4$','2MPentane':'2MC$_5$','3MPentane':'3MC$_5$','2MPropane':'2MC$_3$','23DMButane':'23DMC$_4$','ethyne':'C$_2$','propyne':'C$_3$','1butyne':'1C$_4$','2butyne':'2C$_4$','1pentyne':'1C$_5$','2pentyne':'2C$_5$','1hexyne':'1C$_6$','2hexyne':'2C$_6$','1heptyne':'1C$_7$','1octyne':'1C$_8$','1nonyne':'1C$_9$'}
title_list = ['Branched alkanes','Alkynes']
color_dic = {'2MPropane':'r','2MButane':'b','2MPentane':'g','3MPentane':'c','23DMButane':'m','22DMPropane':'y','22DMButane':'r','224TMPentane':'b','ethyne':'r','propyne':'b','1butyne':'g','2butyne':'c','1pentyne':'m','2pentyne':'y','1hexyne':'r','2hexyne':'b','1heptyne':'g','1octyne':'c','1nonyne':'m'}
line_dic = {'2MPropane':'-','2MButane':'--','2MPentane':'-.','3MPentane':':','23DMButane':'-','22DMPropane':'--','22DMButane':'-.','224TMPentane':':','ethyne':'-','propyne':'--','1butyne':'-','2butyne':'--','1pentyne':'-','2pentyne':'--','1hexyne':'-.','2hexyne':':','1heptyne':'-.','1octyne':':','1nonyne':'-.'}
symbol_dic = {'2MPropane':'o','2MButane':'s','2MPentane':'d','3MPentane':'^','23DMButane':'v','22DMPropane':'<','22DMButane':'>','224TMPentane':'o','ethyne':'o','propyne':'s','1butyne':'d','2butyne':'^','1pentyne':'v','2pentyne':'<','1hexyne':'>','2hexyne':'o','1heptyne':'s','1octyne':'d','1nonyne':'^'}

#fig, axarr = plt.subplots(nrows=2,ncols=2,figsize=[12,12])    
fig, axarr = plt.subplots(nrows=1,ncols=2,figsize=[20,10])

counter = 0    

referenceFF_list = ['Potoff_SL'] #,'Potoff_gen','TraPPE']

for referenceFF_i in referenceFF_list:

    for group in group_list:
           
        for compound in compound_list[group]:

            try:
#                print('Plot optimization for '+compound+' with '+referenceFF_i)
                scaling_factor_optimization = np.loadtxt('H:/MBAR_GCMC/Epsilon_scaling/'+referenceFF_i+'/'+compound+'_scan.txt',skiprows=1)
#                if compound == '3MPentane': scaling_factor_optimization = np.loadtxt('H:/MBAR_GCMC/Epsilon_scaling/Potoff_gen/'+compound+'_scan.txt',skiprows=1)

                eps_computed_compound = scaling_factor_optimization[:,0]
                Score_computed_compound = scaling_factor_optimization[:,1]
                
                ### Remove any strange outliers
                eps_computed_compound = eps_computed_compound[Score_computed_compound<5.5]
                Score_computed_compound = Score_computed_compound[Score_computed_compound<5.5]
                
                eps_opt_compound = eps_computed_compound[np.argmin(Score_computed_compound)]
                Score_opt_compound = np.min(Score_computed_compound)
                
                if np.any(eps_computed_compound == 1.): Score_ref_compound = Score_computed_compound[eps_computed_compound==1.][0]
                
#                axarr[axarr_dic[group][referenceFF_i]].plot(np.sort(eps_computed_compound),Score_computed_compound[np.argsort(eps_computed_compound)],color_dic[compound]+symbol_dic[compound]+line_dic[compound],mfc='None')
                ### Old style
#                axarr[axarr_dic[group][referenceFF_i]].plot(np.sort(eps_computed_compound),Score_computed_compound[np.argsort(eps_computed_compound)],color_dic[compound]+line_dic[compound],mfc='None')
#                axarr[axarr_dic[group][referenceFF_i]].plot(eps_opt_compound,Score_opt_compound,color_dic[compound]+'*',markersize=15)
#                axarr[axarr_dic[group][referenceFF_i]].plot([],[],color_dic[compound]+line_dic[compound]+symbol_dic[compound],markersize=15,mfc='None',label=label_dic[compound])
#                axarr[axarr_dic[group][referenceFF_i]].errorbar(1.,Score_ref_compound,fmt=color_dic[compound]+symbol_dic[compound],mfc='None',markersize=15)
                
                axarr[axarr_dic[group][referenceFF_i]].plot(np.sort(eps_computed_compound),Score_computed_compound[np.argsort(eps_computed_compound)],color_dic[compound]+line_dic[compound],mfc='None',linewidth=2)
                axarr[axarr_dic[group][referenceFF_i]].plot(eps_opt_compound,Score_opt_compound,color_dic[compound]+symbol_dic[compound],mfc='None',markersize=12,markeredgewidth=2)
                axarr[axarr_dic[group][referenceFF_i]].plot([],[],color_dic[compound]+line_dic[compound]+symbol_dic[compound],mfc='None',markersize=12,label=label_dic[compound],linewidth=2,markeredgewidth=2)
                
#                try:
##                print('Plot uncertainties for '+compound+' with '+referenceFF_i)
#                    Score_95_ref = np.loadtxt('H:/MBAR_GCMC/Epsilon_scaling/'+referenceFF+'/'+compound+'_uScore_ref.txt',skiprows=1)
#                    Score_95_opt = np.loadtxt('H:/MBAR_GCMC/Epsilon_scaling/'+referenceFF+'/'+compound+'_uScore_opt.txt',skiprows=1)
#                             
#                    Score_low95_ref = Score_95_ref[0]
#                    Score_high95_ref = Score_95_ref[1]
#    
#                    Score_low95_opt = Score_95_opt[0]
#                    Score_high95_opt = Score_95_opt[1]
#                    
#                    uScore_low95_ref = Score_ref_compound - Score_low95_ref
#                    uScore_high95_ref = Score_high95_ref - Score_ref_compound
#    
#                    uScore_low95_opt = Score_opt_compound - Score_low95_opt
#                    uScore_high95_opt = Score_high95_opt - Score_opt_compound
#                    
#                    axarr[axarr_dic[group][referenceFF_i]].errorbar(1.,Score_ref_compound,yerr=[[uScore_low95_ref,],[uScore_high95_ref,]],fmt=color_dic[compound]+symbol_dic[compound],mfc='None',markersize=15)
#                    axarr[axarr_dic[group][referenceFF_i]].errorbar(eps_opt_compound,Score_opt_compound,yerr=[[uScore_low95_opt,],[uScore_high95_opt,]],fmt=color_dic[compound]+'*',markersize=15)
#                    
#                except:
#                    
#                    print('No uncertainty values for '+compound+' with '+referenceFF_i)
                    
            except:
                
                print('Optimization results not available for '+compound+' with '+referenceFF_i)
                            
            counter += 1
               
#axarr_dic = {0:(0,0),1:(0,1),2:(1,0),3:(1,1)}
axarr_dic = {0:(0),1:(1)}

xlim_low = [0.995,0.995,0.99,0.99]
xlim_high = [1.015,1.03,1.01,1.01]
ylim_high = [2,6]
ylim_low = [-0.05,-0.1,None,None]
xticks = [[0.995,1.000,1.005,1.010,1.015],[1.00,1.01,1.02,1.03]]
yticks = [[0.0,0.5,1.0,1.5,2.0],[0,1,2,3,4,5,6]]

for iax in range(len(axarr_dic)): #, group in enumerate(group_list):
#    axarr[axarr_dic[iax]].set_xlabel(r'$\epsilon / \epsilon_{\rm Potoff}$')
    axarr[axarr_dic[iax]].set_xlabel(xlabel_list[iax])
    axarr[axarr_dic[iax]].set_ylabel(r'Scoring Function')
    axarr[axarr_dic[iax]].set_title(title_list[iax],fontsize=36)
    axarr[axarr_dic[iax]].set_xlim([xlim_low[iax],xlim_high[iax]])
    axarr[axarr_dic[iax]].set_xticks(xticks[iax])
    axarr[axarr_dic[iax]].set_yticks(yticks[iax])
    axarr[axarr_dic[iax]].set_ylim([ylim_low[iax],ylim_high[iax]])
    
    axarr[axarr_dic[iax]].plot([1.0,1.0],[ylim_low[iax],ylim_high[iax]],'k--',linewidth=2)
    
axarr[axarr_dic[0]].legend(ncol=1,labelspacing=0,columnspacing=0)  
axarr[axarr_dic[1]].legend(ncol=1,labelspacing=0,columnspacing=0)    

#axarr[axarr_dic[0]].text(0.995,2.2,'Branched alkanes',fontsize=36)
#axarr[axarr_dic[1]].text(1.002,5.8,'Alkynes',fontsize=36)

plt.tight_layout(pad=0.5)

fig.savefig('Optimal_epsilon_scaling.pdf')
        
plt.show()