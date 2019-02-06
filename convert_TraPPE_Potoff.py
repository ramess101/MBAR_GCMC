# -*- coding: utf-8 -*-
"""
Converts TraPPE to Potoff
"""

from __future__ import division
import numpy as np
from MBAR_rerunModel import MBAR_GCMC
import os.path
import matplotlib.pyplot as plt
from extract_Potoff import extract_Potoff

nhists_max = 13

#group_list = ['C-CH-group','C-group','CH-group','alkynes']
compound_list = ['2MPropane','22DMPropane','224TMPentane','23DMButane','234TMPentane']
#,'C-CH-group':['223TMButane','223TMPentane','224TMPentane','233TMPentane'],'C-group':['3M3EPentane','22DMButane','22DMHexane','22DMPentane','22DMPropane','33DMHexane','33DMPentane','2233TetraMButane'],'CH-group':['2M3EPentane','2MButane','2MHeptane','2MHexane','2MPentane','2MPropane','3EHexane','3EPentane','3MHeptane','3MHexane','3MPentane','4MHeptane','23DMBUtane','23DMHexane','23DMPentane','24DMHexane','24DMPentane','25DMHexane','34DMHexane','234TMPentane'],'alkynes':['ethyne','propyne','1butyne','2butyne','1pentyne','2pentyne','1hexyne','2hexyne','1heptyne','1octyne','1nonyne']}  #{'C-group':['22DMPropane'],'C-CH-group':['224TMPentane']}
REFPROP_list = {'224TMPentane':'Isooctane','2MPropane':'Isobutane','22DMPropane':'Neopentane','2MButane':'IPENTANE','2MPentane':'IHEXANE','1butyne':'1-Butyne','ethyne':'Acetylene','propyne':'Propyne'}#,'3MPentane':'3METHYLPENTANE','22DMButane':'22DIMETHYLBUTANE','23DMButane':'23DIMETHYLBUTANE'}

directory_dic = {'alkanes':'alkanes/','C-CH-group':'branched-Alkanes/C-CH-group/','C-group':'branched-Alkanes/C-group/','CH-group':'branched-Alkanes/CH-group/','alkynes':'alkynes/'}
experimental_dic = {'C-CH-group':'','C-group':'REFPROP_values','CH-group':'REFPROP_values','alkynes':'DIPPR_values'}

referenceFF_directory_dic = {'Potoff_SL':'Optimized/','Potoff_gen':'Generalized/','TraPPE':'TraPPE/','Potoff':'Potoff/'}

### Original
color_list = {'TraPPE':'b','Potoff':'r','NERD':'g','PotoffGen':'r','PotoffSL':'c'}
shape_list = {'TraPPE':'s','Potoff':'o','NERD':'^','PotoffGen':'o','PotoffSL':'v'}
line_list = {'TraPPE':'-','Potoff':'--','NERD':'-.','PotoffGen':':','PotoffSL':'--'}

### Updated
#color_list = {'2MPropane':'b','22DMPropane':'r','224TMPentane':'g','23DMButane':'y','234TMPentane':'m','22DMButane':'c','33DMHexane':'k','3M3EPentane':'y'}
color_list = {'2MPropane':'blue','22DMPropane':'red','224TMPentane':'green','23DMButane':'gold','234TMPentane':'purple','22DMButane':'lime','33DMHexane':'cyan','3M3EPentane':'maroon'}
shape_list = {'TraPPE':'s','Potoff':'o','NERD':'s','PotoffGen':'o','PotoffSL':'o'}
line_list = {'TraPPE':'-','Potoff':'--','NERD':'-','PotoffGen':':','PotoffSL':':'}
mfc_list = {'TraPPE':'None','Potoff':lambda compound: color_list[compound],'NERD':lambda compound: color_list[compound],'PotoffGen':lambda compound: color_list[compound],'PotoffSL':'v'}
label_list = {'TraPPEtoPotoffGen':r'TraPPE $\Rightarrow$ MiPPE-gen','PotoffGentoTraPPE':r'MiPPE-gen $\Rightarrow$ TraPPE','PotoffGentoPotoffSL':r'MiPPE-gen $\Rightarrow$ MiPPE-SL','TraPPEtoNERD':r'TraPPE $\Rightarrow$ NERD','TraPPE':'TraPPE, Mick et al.','PotoffGen':'MiPPE-gen, Mick et al.','PotoffSL':'MiPPE-SL, Mick et al.','NERD':'NERD, Mick et al.','2MPropane':'2-methylpropane','22DMPropane':'2,2-dimethylpropane','224TMPentane':'2,2,4-trimethylpentane','23DMButane':'2,3-dimethylbutane','234TMPentane':'2,3,4-trimethylpentane','22DMButane':'2,2-dimethylbutane','33DMHexane':'3,3-dimethylhexane','3M3EPentane':'3-methyl-3-ethylpentane'}
black_list = ['PotoffGentoPotoffSL_33DMHexane','PotoffGentoPotoffSL_3M3EPentane','TraPPEtoNERD_2MPropane']
legend_shift = {'constant':-0.45,'12to16':-0.41}

reprocess = False
trim_data = False
remove_low_high_Tsat = False
bootstrap = False
nBoots = 30

lam = '12to16'

if lam == 'constant':
    
    reference_list = ['PotoffGen','TraPPE']
    rerun_list = {'TraPPE':['NERD'],'PotoffGen':['PotoffSL']}
    compound_list = ['2MPropane','22DMPropane','224TMPentane','234TMPentane','23DMButane','22DMButane','33DMHexane','3M3EPentane']
    
elif lam == '12to16':
    
    reference_list = ['TraPPE','PotoffGen']
    rerun_list = {'TraPPE':['PotoffGen'],'PotoffGen':['TraPPE']}
    compound_list = ['2MPropane','22DMPropane','224TMPentane','23DMButane','234TMPentane']

else:
    
    print('Must prescribe lambda type')
    reference_list = []
    
#reference_list = ['PotoffGen']
#rerun_list = {'TraPPE':['NERD'],'PotoffGen':['PotoffSL']}
#compound_list = ['22DMButane']

#rerun_list = {'TraPPE':['Potoff','NERD'],'Potoff':['TraPPE','Potoff_SL']}

if not reprocess: 
    
    fig, axarr = plt.subplots(nrows=2,ncols=2,figsize=[12,12])

for referenceFF in reference_list: # ['TraPPE']: # ['TraPPE', 'Potoff']:
    
    for rerunFF in rerun_list[referenceFF]: 
        
        print('refFF: '+referenceFF+', rrFF: '+rerunFF)
       
        for compound in compound_list:# ['2MPropane','22DMPropane','224TMPentane']:# compound_list:
            
            if reprocess:
            
                filepaths = []
                
                root_path = 'H:/GCMC_histFiles/ram9/'+referenceFF+'to'+rerunFF+'/'+compound
                                                               
                for iT in np.arange(0,nhists_max+1):
                                
                    hist_name='/his'+str(iT)+'a.dat'
                        
                    if os.path.exists(root_path+hist_name):
                        
                        filepaths.append(root_path+hist_name)
                        
                if not os.path.exists('H:/MBAR_GCMC/'+rerunFF+'_literature/'+compound+'_withuncertainties.txt'):
                
                    input_path = 'H:/MBAR_GCMC/'+rerunFF+'_literature/'+compound+'_rawdata.txt'
                    output_path = 'H:/MBAR_GCMC/'+rerunFF+'_literature/'+compound+'_withuncertainties.txt'
                    
                    extract_Potoff(input_path,output_path,with_uncertainties=True)
                    
                if not os.path.exists('H:/MBAR_GCMC/'+rerunFF+'_literature/'+compound+'.txt'):
                
                    input_path = 'H:/MBAR_GCMC/'+rerunFF+'_literature/'+compound+'_rawdata.txt'
                    output_path = 'H:/MBAR_GCMC/'+rerunFF+'_literature/'+compound+'.txt'
                    
                    extract_Potoff(input_path,output_path,with_uncertainties=False)
                            
                VLE_Potoff = np.loadtxt('H:/MBAR_GCMC/'+rerunFF+'_literature/'+compound+'.txt',skiprows=1)
                    
                Tsat_Potoff = VLE_Potoff[:,0] #[K]
                rhol_Potoff = VLE_Potoff[:,1]*1000. #[kg/m3]
                rhov_Potoff = VLE_Potoff[:,2]*1000. #[kg/m3]
                Psat_Potoff = VLE_Potoff[:,3] #[bar]
                DeltaHv_Potoff = VLE_Potoff[:,4] #[kJ/mol]
                
                if True: #compound in REFPROP_list or group == 'alkynes':
                    
                    try:
                    
                        VLE_RP = np.loadtxt('H:/MBAR_GCMC/Experimental_values/REFPROP_values/VLE_'+compound+'.txt',skiprows=1)
                        
                        Tsat_RP = VLE_RP[:,0] #[K]
                        rhol_RP = VLE_RP[:,1] #[kg/m3]
                        rhov_RP = VLE_RP[:,2] #[kg/m3]
                        Psat_RP = VLE_RP[:,3] #[bar]
        #                DeltaHv_RP = VLE_RP[:,4] #[kJ/mol]
                        
#                        Mw = np.loadtxt('H:/MBAR_GCMC/Experimental_values/MW_'+compound+'.txt')
                        
                    except:
                        
                        print('Compound not evaluted with REFPROP')
                else:
                    
                    print(compound+' not in REFPROP database')
                    Tsat_RP = []
                    rhol_RP = []
                    rhov_RP = []
                    Psat_RP = []
                    DeltaHv_RP = []
                    
                    VLE_Potoff = np.loadtxt('H:/MBAR_GCMC/'+rerunFF+'_literature/'+compound+'_withuncertainties.txt',skiprows=1)
                        
                    Tsat_RP = VLE_Potoff[:,0] #[K]
                    rhol_RP = VLE_Potoff[:,1]*1000. #[kg/m3]
                    rhov_RP = VLE_Potoff[:,3]*1000. #[kg/m3]
                    Psat_RP = VLE_Potoff[:,5] #[bar]
                    DeltaHv_RP = VLE_Potoff[:,7] #[kJ/mol]       
                
                print('Performing MBAR-GCMC analysis for '+compound)
                
                Mw = np.loadtxt('H:/MBAR_GCMC/Experimental_values/MW_'+compound+'.txt')
                
                MBAR_GCMC_trial = MBAR_GCMC(root_path,filepaths,Mw,trim_data=trim_data,compare_literature=True)
                MBAR_GCMC_trial.plot_histograms()
                MBAR_GCMC_trial.plot_2dhistograms()
                MBAR_GCMC_trial.solve_VLE(Tsat_Potoff)
                rhol_ref_model = MBAR_GCMC_trial.rholiq
                rhov_ref_model = MBAR_GCMC_trial.rhovap
                Psat_ref_model = MBAR_GCMC_trial.Psat
                DeltaHv_ref_model = MBAR_GCMC_trial.DeltaHv
                Neffliq_model = MBAR_GCMC_trial.Neffliq
                Neffvap_model = MBAR_GCMC_trial.Neffvap
                Ntotvap_model = len(MBAR_GCMC_trial.Nmol_flat[MBAR_GCMC_trial.Nmol_flat<=MBAR_GCMC_trial.Ncut])
                Ntotliq_model = len(MBAR_GCMC_trial.Nmol_flat[MBAR_GCMC_trial.Nmol_flat>MBAR_GCMC_trial.Ncut])
                
                MBAR_GCMC_trial.solve_VLE(Tsat_Potoff,1.)
                rhol_rr_model = MBAR_GCMC_trial.rholiq
                rhov_rr_model = MBAR_GCMC_trial.rhovap
                Psat_rr_model = MBAR_GCMC_trial.Psat
                DeltaHv_rr_model = MBAR_GCMC_trial.DeltaHv
                Neffliq_rr_model = MBAR_GCMC_trial.Neffliq
                Neffvap_rr_model = MBAR_GCMC_trial.Neffvap
                Ntotvap_rr_model = len(MBAR_GCMC_trial.Nmol_flat[MBAR_GCMC_trial.Nmol_flat<=MBAR_GCMC_trial.Ncut])
                Ntotliq_rr_model = len(MBAR_GCMC_trial.Nmol_flat[MBAR_GCMC_trial.Nmol_flat>MBAR_GCMC_trial.Ncut])
                
                plt.plot(rhol_ref_model,Tsat_Potoff,'ro',mfc='None',label='Ref.')
                plt.plot(rhov_ref_model,Tsat_Potoff,'ro',mfc='None',label='Ref.')
                plt.plot(rhol_rr_model,Tsat_Potoff,'bs',mfc='None',label='Rerun')
                plt.plot(rhov_rr_model,Tsat_Potoff,'bs',mfc='None',label='Rerun')
                plt.plot(rhol_RP,Tsat_RP,'k-',label='REFPROP')
                plt.plot(rhov_RP,Tsat_RP,'k-',label='REFPROP')
                plt.show()
                
                plt.plot(1000./Tsat_Potoff,np.log10(Psat_ref_model),'ro',mfc='None',label='Ref.')
                plt.plot(1000./Tsat_Potoff,np.log10(Psat_rr_model),'bs',mfc='None',label='Rerun')
                plt.plot(1000./Tsat_RP,np.log10(Psat_RP),'k-',label='REFPROP')
                plt.show()
                
                plt.plot(Tsat_Potoff,DeltaHv_ref_model,'ro',mfc='None',label='Ref.')
                plt.plot(Tsat_Potoff,DeltaHv_rr_model,'bs',mfc='None',label='Rerun')
                plt.show()
                
                ### Compile the reference model results
                out_file =open('H:/MBAR_GCMC/refFF_to_rrFF/'+referenceFF+'/'+compound+'.txt','w')
                out_file.write('T (K) rhol (kg/m3) rhov (kg/m3) P (bar) DeltaHv (kJ/mol)'+'\n')
                for Tsat, rhol, rhov, Psat, DeltaHv in zip(Tsat_Potoff,rhol_ref_model,rhov_ref_model,Psat_ref_model,DeltaHv_ref_model):
                    out_file.write(str(Tsat)+'\t'+str(rhol)+'\t'+str(rhov)+'\t'+str(Psat)+'\t'+str(DeltaHv)+'\n')
                out_file.close()  
                
                ### Compile the rerun model results
                out_file =open('H:/MBAR_GCMC/refFF_to_rrFF/'+referenceFF+'to'+rerunFF+'/'+compound+'.txt','w')
                out_file.write('T (K) rhol (kg/m3) rhov (kg/m3) P (bar) DeltaHv (kJ/mol)'+'\n')
                for Tsat, rhol, rhov, Psat, DeltaHv in zip(Tsat_Potoff,rhol_rr_model,rhov_rr_model,Psat_rr_model,DeltaHv_rr_model):
                    out_file.write(str(Tsat)+'\t'+str(rhol)+'\t'+str(rhov)+'\t'+str(Psat)+'\t'+str(DeltaHv)+'\n')
                out_file.close()  
                
                ### Compile the rerun model number of effective samples
                out_file =open('H:/MBAR_GCMC/refFF_to_rrFF/'+referenceFF+'to'+rerunFF+'/'+compound+'_Neff.txt','w')
                out_file.write('T (K) Neffliq Neffvap Percentliq Percentvap Ratioliq Ratiovap'+'\n')
                for Tsat, Neffliq_rr, Neffvap_rr, Neffliq_ref, Neffvap_ref in zip(Tsat_Potoff,Neffliq_rr_model,Neffvap_rr_model,Neffliq_model,Neffvap_model):
                    out_file.write(str(Tsat)+'\t'+str(Neffliq_rr)+'\t'+str(Neffvap_rr)+'\t'+str(Neffliq_rr/Ntotliq_rr_model*100.)+'\t'+str(Neffvap_rr/Ntotvap_rr_model*100.)+'\t'+str(Neffliq_rr/Neffliq_ref*100.)+'\t'+str(Neffvap_rr/Neffvap_ref*100.)+'\n')
                out_file.close()  
                
            else:
                
                if any(black_i == referenceFF+'to'+rerunFF+'_'+compound for black_i in black_list):
                    print(referenceFF+'to'+rerunFF+'_'+compound)
                    rFF_list = []
                else:
                    rFF_list =  [rerunFF,referenceFF+'to'+rerunFF]
#                if compound == 'C6H14': rFF_list =  [referenceFF,referenceFF+'to'+rerunFF]
                
                for rFF in rFF_list:
                    
                    try:
                        
                        if rFF == rerunFF:
                            
#                            symbol = color_list[rFF]+line_list[rFF]
                            color = color_list[compound]
                            line = line_list[rFF]
                            shape = ''
                            symbol = color + line # color_list[compound]+line_list[rFF]
                            label = 'Ref.'
                            path_tail = rFF+'_literature'
                            rho_conversion = 1000.
                            
                            if compound == 'C6H14': #C6H14 does not actually have literature values
                                path_tail = 'refFF_to_rrFF/'+rFF
                                rho_conversion = 1.
                            
                        elif rFF  == referenceFF:
                            
#                            symbol = color_list[rFF]+line_list[rFF]
                            color = color_list[compound]
                            line = line_list[rFF]
                            symbol = color_list[compound]+line_list[rFF]
                            label = 'Ref.'
                            path_tail = 'refFF_to_rrFF/'+rFF
                            rho_conversion = 1.
                        
                        else:
                        
#                            symbol = color_list[rerunFF]+shape_list[rerunFF]
                            color = color_list[compound]
                            shape = shape_list[rerunFF]
                            line = ''
                            symbol = color + shape #color_list[compound]+shape_list[rerunFF]
                            label = 'Rerun'
                            path_tail = 'refFF_to_rrFF/'+rFF
                            rho_conversion = 1.
                        
                        VLE_load = np.loadtxt('H:/MBAR_GCMC/'+path_tail+'/'+compound+'.txt',skiprows=1)
                                      
                        Tsat_load = VLE_load[:,0] #[K]
                        rhol_load = VLE_load[:,1]*rho_conversion #[kg/m3]
                        rhov_load = VLE_load[:,2]*rho_conversion #[kg/m3]
                        Psat_load = VLE_load[:,3]/10. #[MPa]
                        DeltaHv_load = VLE_load[:,4] #[kJ/mol]
                        
                        ### Old format
#                        axarr[0,0].plot(Tsat_load,rhol_load,symbol,mfc='None',label=label)
#                        axarr[0,1].plot(Tsat_load,np.log10(rhov_load),symbol,mfc='None',label=label)
#                        axarr[1,0].plot(1000./Tsat_load,np.log10(Psat_load),symbol,mfc='None',label=label)
#                        axarr[1,1].plot(Tsat_load,DeltaHv_load,symbol,mfc='None')
                        
                        axarr[0,0].plot(Tsat_load,rhol_load,line+shape,color=color,mfc='None',label=label)
                        axarr[0,1].plot(Tsat_load,np.log10(rhov_load),line+shape,color=color,mfc='None',label=label)
                        axarr[1,0].plot(1000./Tsat_load,np.log10(Psat_load),line+shape,color=color,mfc='None',label=label)
                        axarr[1,1].plot(Tsat_load,DeltaHv_load,line+shape,color=color,mfc='None')
                        
                    except:
                        
                        print(compound+' not processed yet for refFF:'+referenceFF+' and rrFF: '+rerunFF)

prop_list = [r'$\rho_{\rm liq}^{\rm sat}$',r'$\rho_{\rm vap}^{\rm sat}$',r'$P_{\rm vap}^{\rm sat}$',r'$\Delta H_{\rm v}$',r'$Z^{\rm sat}_{\rm vap}$',r'$P \Delta V$']  #r'$\Delta U_{\rm v}$'] #r'$P \Delta V$'] #r'$Z^{\rm sat}_{\rm vap}$']#r'$\Delta H_{\rm v}$']
#axarr_dic = {0:(0,0),1:(0,1),2:(1,0),3:(1,1),4:(2,0),5:(2,1)}
#        
##for iax, prop in enumerate(prop_list):
##    axarr[axarr_dic[iax]].set_xlabel(r'$T_{\rm r}$')
##    
##    axarr[axarr_dic[iax]].set_ylabel(r'$\frac{X_{\rm MBAR} - X_{\rm HR}}{X_{\rm HR}} \times 100 \%$')
##    
##    axarr[axarr_dic[iax]].set_title(r'$X = $'+prop)
##    axarr[axarr_dic[iax]].set_xlim([0.6,1.])
##    axarr[axarr_dic[iax]].set_xticks([0.65,0.75,0.85,0.95])
##    axarr[axarr_dic[iax]].set_ylim([ylim_low[iax],ylim_high[iax]])

axarr[0,0].set_xlabel(r'$T$ (K)')
axarr[0,1].set_xlabel(r'$T$ (K)')
axarr[1,0].set_xlabel(r'$1000/T$ (K)')
axarr[1,1].set_xlabel(r'$T$ (K)')

axarr[0,0].set_ylabel(r'$\rho_{\rm liq}^{\rm sat}$ (kg/m$^3$)')
axarr[0,1].set_ylabel(r'$\log_{10}(\rho_{\rm vap}^{\rm sat}/$(kg/m$^3))$')
axarr[1,0].set_ylabel(r'$\log_{10}(P_{\rm vap}^{\rm sat}/$MPa$)$')
axarr[1,1].set_ylabel(r'$\Delta H_{\rm v}$ (kJ/mol)')

for referenceFF in reference_list:
    
    for rerunFF in rerun_list[referenceFF]: 
        
        axarr[1,1].plot([],[],'k'+shape_list[rerunFF],markersize=10,mfc='None',label=label_list[referenceFF+'to'+rerunFF])
        
for referenceFF in reference_list:
    
    for rerunFF in rerun_list[referenceFF]:  
       
        axarr[1,1].plot([],[],'k'+line_list[rerunFF],linewidth=2,label=label_list[rerunFF])
       
for compound in compound_list:
    
    axarr[1,1].plot([],[],'-',color=color_list[compound],linewidth=12,label=label_list[compound])

lgd = axarr[1,1].legend(loc='lower center', bbox_to_anchor=(-0.3, legend_shift[lam]),
          ncol=3,numpoints=3,handlelength=3,handletextpad=0.3,columnspacing=0.5,frameon=True,borderaxespad=0)
         
plt.tight_layout()           

fig.savefig('refFF_to_rrFF_lam_'+lam+'.pdf',bbox_extra_artists=(lgd,),bbox_inches='tight')   
 