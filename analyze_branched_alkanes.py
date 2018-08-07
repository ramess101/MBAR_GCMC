# -*- coding: utf-8 -*-
"""
Analyze branched alkanes
"""

from __future__ import division
import numpy as np
from optimize_epsilon import MBAR_GCMC
import CoolProp.CoolProp as CP
import os.path
from extract_Potoff import extract_Potoff

nhists_max = 15

group_list = ['C-CH-group','C-group','CH-group']
compound_list = {'C-CH-group':['223TMButane','223TMPentane','224TMPentane','233TMPentane'],'C-group':['3M3EPentane','22DMButane','22DMHexane','22DMPentane','22DMPropane','33DMHexane','33DMPentane','2233TetraMButane'],'CH-group':['2M3EPentane','2MButane','2MHeptane','2MHexane','2MPentane','2MPropane','3EHexane','3EPentane','3MHeptane','3MHexane','3MPentane','4MHeptane','23DMBUtane','23DMHexane','23DMPentane','24DMHexane','24DMPentane','25DMHexane','34DMHexane','234TMPentane']}  #{'C-group':['22DMPropane'],'C-CH-group':['224TMPentane']}
REFPROP_list = {'224TMPentane':'Isooctane','2MPropane':'Isobutane','22DMPropane':'Neopentane','2MButane':'IPENTANE','2MPentane':'IHEXANE'}#,'3MPentane':'3METHYLPENTANE','22DMButane':'22DIMETHYLBUTANE','23DMButane':'23DIMETHYLBUTANE'}

for group in group_list:
       
    for compound in compound_list[group]: # ['224-trimethylpentane','neopentane','isobutane']:
        
        filepaths = []
        
        root_path = 'H:/Mie-swf/histFiles/'+group+'/'+compound
        
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
        
        if compound in REFPROP_list:
            
            try:
            
                VLE_RP = np.loadtxt('H:/MBAR_GCMC/REFPROP_values/VLE_'+compound+'.txt',skiprows=1)
                
                Tsat_RP = VLE_RP[:,0] #[K]
                rhol_RP = VLE_RP[:,1] #[kg/m3]
                rhov_RP = VLE_RP[:,2] #[kg/m3]
                Psat_RP = VLE_RP[:,3] #[bar]
                
                Mw = np.loadtxt('H:/MBAR_GCMC/REFPROP_values/MW_'+compound+'.txt')
                
            except:
                
                Mw = CP.PropsSI('M','REFPROP::'+REFPROP_list[compound])*1000. #[gm/mol]
                            
                Tsat_RP = Tsat_Potoff.copy()
                rhol_RP = CP.PropsSI('D','T',Tsat_RP,'Q',0,'REFPROP::'+REFPROP_list[compound]) #[kg/m3]   
                rhov_RP = CP.PropsSI('D','T',Tsat_RP,'Q',1,'REFPROP::'+REFPROP_list[compound]) #[kg/m3] 
                Psat_RP = CP.PropsSI('P','T',Tsat_RP,'Q',1,'REFPROP::'+REFPROP_list[compound])/100000. #[bar]
                
                f1 = open('H:/MBAR_GCMC/REFPROP_values/VLE_'+compound+'.txt','w')
                f1.write('Tsat (K) rhol (kg/m3) rhov (kg/m3) Psat (bar)'+'\n')
                for Tsat, rhol, rhov, Psat in zip(Tsat_RP,rhol_RP,rhov_RP,Psat_RP):
                    f1.write(str(Tsat)+'\t')
                    f1.write(str(rhol)+'\t')
                    f1.write(str(rhov)+'\t')   
                    f1.write(str(Psat)+'\n') 
                f1.close()
                
                f2 = open('H:/MBAR_GCMC/REFPROP_values/MW_'+compound+'.txt','w')
                f2.write(str(Mw))
                f2.close()
                
        else:
            
            print(compound+' not in REFPROP database')
            Tsat_RP = []
            rhol_RP = []
            rhov_RP = []
                
#        MBAR_GCMC_trial = MBAR_GCMC(root_path,filepaths,Mw,trim_data=True,compare_literature=True)
#        MBAR_GCMC_trial.plot_histograms()
#        MBAR_GCMC_trial.plot_2dhistograms()
#        MBAR_GCMC_trial.solve_VLE(Tsat_Potoff)
#        MBAR_GCMC_trial.plot_VLE(Tsat_RP,rhol_RP,rhov_RP,Tsat_Potoff,rhol_Potoff,rhov_Potoff)
        
        if compound in REFPROP_list:# and not os.path.exists('H:/MBAR_GCMC/figures/'+compound+'_AD_eps_scan.pdf'):
            
            print('Performing MBAR-GCMC analysis for '+compound)
            
            MBAR_GCMC_trial = MBAR_GCMC(root_path,filepaths,Mw,trim_data=True,compare_literature=True)
            MBAR_GCMC_trial.plot_histograms()
            MBAR_GCMC_trial.plot_2dhistograms()
            MBAR_GCMC_trial.solve_VLE(Tsat_Potoff)
            MBAR_GCMC_trial.plot_VLE(Tsat_RP,rhol_RP,rhov_RP,Tsat_Potoff,rhol_Potoff,rhov_Potoff)
        
            MBAR_GCMC_trial.eps_scan(Tsat_RP,rhol_RP,rhov_RP,Psat_RP,rhol_Potoff,rhov_Potoff,Psat_Potoff,0.98,1.02,5,compound,remove_low_high_Tsat=True)
            
#        else: #Compare with Potoff, not RP
#        
#            MBAR_GCMC_trial.eps_scan(Tsat_Potoff,rhol_Potoff,rhov_Potoff,rhol_Potoff,rhov_Potoff,0.98,1.02,5,compound,remove_low_high_Tsat=True)