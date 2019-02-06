# -*- coding: utf-8 -*-
"""
Plot the number of effective samples
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
#color_list = {'2MPropane':'b','22DMPropane':'r','224TMPentane':'g','23DMButane':'y','234TMPentane':'m','TraPPE':'b','Potoff':'r','NERD':'g','PotoffGen':'r','PotoffSL':'c'}
color_list = {'2MPropane':'blue','22DMPropane':'red','224TMPentane':'green','23DMButane':'gold','234TMPentane':'purple','22DMButane':'lime','33DMHexane':'cyan','3M3EPentane':'maroon','TraPPE':'blue','Potoff':'red','NERD':'green','PotoffGen':'red','PotoffSL':'purple'}
shape_list = {'TraPPE':'s','Potoff':'o','NERD':'s','PotoffGen':'o','PotoffSL':'o'}
line_list = {'TraPPE':'-','Potoff':'--','NERD':'-','PotoffGen':':','PotoffSL':':'}
mfc_list = {'TraPPE':'None','Potoff':lambda compound: color_list[compound],'NERD':lambda compound: color_list[compound],'PotoffGen':lambda compound: color_list[compound],'PotoffSL':'v'}
label_list = {'TraPPEtoPotoffGen':r'TraPPE $\Rightarrow$ MiPPE-gen','PotoffGentoTraPPE':r'MiPPE-gen $\Rightarrow$ TraPPE','PotoffGentoPotoffSL':r'MiPPE-gen $\Rightarrow$ MiPPE-SL','TraPPEtoNERD':r'TraPPE $\Rightarrow$ NERD','TraPPE':'TraPPE, lit.','PotoffGen':'MiPPE-gen, lit.','PotoffSL':'MiPPE-SL, lit.','NERD':'NERD, lit.','2MPropane':'2-methylpropane','22DMPropane':'2,2-dimethylpropane','224TMPentane':'2,2,4-trimethylpentane','23DMButane':'2,3-dimethylbutane','234TMPentane':'2,3,4-trimethylpentane','22DMButane':'2,2-dimethylbutane','33DMHexane':'3,3-dimethylhexane','3M3EPentane':'3-methyl-3-ethylpentane'}
black_list = ['PotoffGentoPotoffSL_33DMHexane','PotoffGentoPotoffSL_3M3EPentane','TraPPEtoNERD_2MPropane','TraPPEtoNERD_224TMPentane','TraPPEtoNERD_234TMPentane','TraPPEtoNERD_23DMButane']


Neff_liq_bins = {0.65:[],0.7:[],0.75:[],0.8:[],0.85:[],0.9:[],0.95:[]}
Neff_vap_bins = {0.65:[],0.7:[],0.75:[],0.8:[],0.85:[],0.9:[],0.95:[]}
Tr_bins = np.array([0.65,0.7,0.75,0.8,0.85,0.9,0.95])

reprocess = False
trim_data = False
remove_low_high_Tsat = False
bootstrap = False
nBoots = 30

if not reprocess: 
    
    fig, axarr = plt.subplots(nrows=2,ncols=2,figsize=[12,12])
    plt.tight_layout(pad=3) 
    fig2, axarr2 = plt.subplots(nrows=1,ncols=1,figsize=[6,6])
    fig4, axarr4 = plt.subplots(nrows=1,ncols=2,figsize=[12,6])
    plt.tight_layout(pad=1)

lam_list = ['constant','12to16']

for lam in lam_list:

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
    
    for referenceFF in reference_list: # ['TraPPE']: # ['TraPPE', 'Potoff']:
        
        for rerunFF in rerun_list[referenceFF]: 
            
            print('refFF: '+referenceFF+', rrFF: '+rerunFF)
           
            for compound in compound_list:# ['2MPropane','22DMPropane','224TMPentane']:# compound_list:
                
                if any(black_i == referenceFF+'to'+rerunFF+'_'+compound for black_i in black_list):
                    black_listed = True
                else:
                    black_listed = False
                
                if black_listed: #reprocess or (referenceFF == 'TraPPE' and compound == '234TMPentane'):
                    
                    print(compound+' was black listed')
                    
                else:
                    
                    ### First load the literature values
                    path_tail = rerunFF+'_literature'
                    rho_conversion = 1000.
                    
                    VLE_lit = np.loadtxt('H:/MBAR_GCMC/'+path_tail+'/'+compound+'.txt',skiprows=1)
                                  
                    Tsat_lit = VLE_lit[:,0] #[K]
                    rhol_lit = VLE_lit[:,1]*rho_conversion #[kg/m3]
                    rhov_lit = VLE_lit[:,2]*rho_conversion #[kg/m3]
                    Psat_lit = VLE_lit[:,3]/10. #[MPa]
                    DeltaHv_lit = VLE_lit[:,4] #[kJ/mol]
                    
                    Tc_lit = np.loadtxt('H:/MBAR_GCMC/Potoff_literature/'+compound+'_Tc.txt',skiprows=0)
                    
                    ### Load the MBAR estimates
                    
                    color = color_list[compound]
                    shape = shape_list[rerunFF]
                    symbol = color_list[compound]+shape_list[rerunFF]
                    label = 'Rerun'
                    path_tail = 'refFF_to_rrFF/'+referenceFF+'to'+rerunFF
                    rho_conversion = 1.
                                           
                    VLE_load = np.loadtxt('H:/MBAR_GCMC/'+path_tail+'/'+compound+'.txt',skiprows=1)
                                  
                    Tsat_load = VLE_load[:,0] #[K]
                    rhol_load = VLE_load[:,1]*rho_conversion #[kg/m3]
                    rhov_load = VLE_load[:,2]*rho_conversion #[kg/m3]
                    Psat_load = VLE_load[:,3]/10. #[MPa]
                    DeltaHv_load = VLE_load[:,4] #[kJ/mol]
                    
                    rhol_error = (rhol_load-rhol_lit)/rhol_lit*100.
                    rhov_error = (rhov_load-rhov_lit)/rhov_lit*100.
                    Psat_error = (Psat_load-Psat_lit)/Psat_lit*100.
                    rhol_error = (np.log10(rhol_load)-np.log10(rhol_lit))/np.log10(rhol_lit)*100.
                    rhov_error = (np.log10(rhov_load)-np.log10(rhov_lit))/np.log10(rhov_lit)*100.
#                    Psat_error = (np.log10(Psat_load)-np.log10(Psat_lit))
                    DeltaHv_error = (DeltaHv_load-DeltaHv_lit)/DeltaHv_lit*100.             
                    
    #                axarr[0,0].plot(Tsat_load,rhol_error,symbol,mfc='None',label=label)
    #                axarr[0,1].plot(Tsat_load,rhov_error,symbol,mfc='None',label=label)
    #                axarr[1,0].plot(1000./Tsat_load,Psat_error,symbol,mfc='None',label=label)
    #                axarr[1,1].plot(Tsat_load,DeltaHv_error,symbol,mfc='None')
    
                    Tr_load = Tsat_load/Tc_lit
                    
                    Neff_load = np.loadtxt('H:/MBAR_GCMC/'+path_tail+'/'+compound+'_Neff.txt',skiprows=1)
                    Neff_liq = Neff_load[:,1]
                    Neff_vap = Neff_load[:,2]
                    
                    ### Old format
#                    axarr[0,0].plot(np.log10(Neff_liq),rhol_error,symbol,mfc='None',label=label)
#                    axarr[0,1].plot(np.log10(Neff_vap),rhov_error,symbol,mfc='None',label=label)
#                    axarr[1,0].plot(np.log10(Neff_vap),Psat_error,symbol,mfc='None',label=label)
#                    axarr[1,1].plot(np.log10(Neff_liq),DeltaHv_error,symbol,mfc='None')

                    axarr[0,0].plot(np.log10(Neff_liq),rhol_error,shape,color=color,mfc='None',label=label)
                    axarr[0,1].plot(np.log10(Neff_vap),rhov_error,shape,color=color,mfc='None',label=label)
                    axarr[1,0].plot(np.log10(Neff_vap),Psat_error,shape,color=color,mfc='None',label=label)
                    axarr[1,1].plot(np.log10(Neff_liq),DeltaHv_error,shape,color=color,mfc='None')
                    
                    for Tr, Neff_liq_i, Neff_vap_i in zip(Tr_load,Neff_liq,Neff_vap): 
                        
                        iBin = np.argmin(np.abs(Tr-Tr_bins))
                        
                        Neff_liq_bins[Tr_bins[iBin]].append(Neff_liq_i)
                        Neff_vap_bins[Tr_bins[iBin]].append(Neff_vap_i)
                    
                    axarr2.plot(Tr_load,np.log10(Neff_liq),'bs',mfc='None')
                    axarr2.plot(Tr_load,np.log10(Neff_vap),'ro',mfc='None') 
                    
                    ### Old format
#                    axarr4[0].plot(Tr_load,np.log10(Neff_liq),color_list[rerunFF]+shape_list[rerunFF],mfc='None')
#                    axarr4[1].plot(Tr_load,np.log10(Neff_vap),color_list[rerunFF]+shape_list[rerunFF],mfc='None') 
#                    
#                    axarr4[0].plot(Tr_load,np.log10(Neff_liq),shape_list[rerunFF],color=color_list[compound],mfc='None')
#                    axarr4[1].plot(Tr_load,np.log10(Neff_vap),shape_list[rerunFF],color=color_list[compound],mfc='None') 
#                    
                    if (referenceFF == 'TraPPE' and rerunFF == 'PotoffGen') or (referenceFF == 'PotoffGen' and rerunFF == 'TraPPE'):

#                        axarr4[0].plot(Tr_load,np.log10(Neff_liq),color_list[rerunFF]+shape_list[rerunFF],mfc='None')
#                        axarr4[1].plot(Tr_load,np.log10(Neff_vap),color_list[rerunFF]+shape_list[rerunFF],mfc='None') 
                        
                        axarr4[0].plot(Tr_load,np.log10(Neff_liq),shape_list[rerunFF],color=color_list[compound],mfc='None')
                        axarr4[1].plot(Tr_load,np.log10(Neff_vap),shape_list[rerunFF],color=color_list[compound],mfc='None') 
                        
                    else:
#                        
#                        axarr4[0].plot(Tr_load,np.log10(Neff_liq),color_list[rerunFF]+shape_list[rerunFF],mfc=color_list[rerunFF])
#                        axarr4[1].plot(Tr_load,np.log10(Neff_vap),color_list[rerunFF]+shape_list[rerunFF],mfc=color_list[rerunFF])
#                        
                        axarr4[0].plot(Tr_load,np.log10(Neff_liq),shape_list[rerunFF],color=color_list[compound],mfc=color_list[compound])
                        axarr4[1].plot(Tr_load,np.log10(Neff_vap),shape_list[rerunFF],color=color_list[compound],mfc=color_list[compound]) 

prop_list = [r'$\rho_{\rm liq}^{\rm sat}$',r'$\rho_{\rm vap}^{\rm sat}$',r'$P_{\rm vap}^{\rm sat}$',r'$\Delta H_{\rm v}$']  #r'$\Delta U_{\rm v}$'] #r'$P \Delta V$'] #r'$Z^{\rm sat}_{\rm vap}$']#r'$\Delta H_{\rm v}$']
axarr_dic = {0:(0,0),1:(0,1),2:(1,0),3:(1,1),4:(2,0),5:(2,1)}
#        
for iax, prop in enumerate(prop_list):
    axarr[axarr_dic[iax]].set_xlabel(r'$\log_{10}(K^{\rm eff}_{\rm snaps})$')
    
    axarr[axarr_dic[iax]].set_ylabel(r'$\frac{X_{\rm MBAR} - X_{\rm HR}}{X_{\rm HR}} \times 100 \%$')
    
    axarr[axarr_dic[iax]].set_title(r'$X = $'+prop)
    
    current_ylim = axarr[axarr_dic[iax]].get_ylim()
    
    axarr[axarr_dic[iax]].plot([np.log10(50),np.log10(50)],current_ylim,'k--')
    axarr[axarr_dic[iax]].set_ylim(current_ylim)
    
#    axarr[axarr_dic[iax]].set_xlim([0.6,1.])
#    axarr[axarr_dic[iax]].set_xticks([0.65,0.75,0.85,0.95])
#    axarr[axarr_dic[iax]].set_ylim([ylim_low[iax],ylim_high[iax]])

#axarr[0,0].set_xlabel(r'$T$ (K)')
#axarr[0,1].set_xlabel(r'$T$ (K)')
#axarr[1,0].set_xlabel(r'$1000/T$ (K)')
#axarr[1,1].set_xlabel(r'$T$ (K)')
#
#axarr[0,0].set_ylabel(r'$\rho_{\rm liq}^{\rm sat}$ (kg/m$^3$)')
#axarr[0,1].set_ylabel(r'$\rho_{\rm vap}^{\rm sat}$ (kg/m$^3)$')
#axarr[1,0].set_ylabel(r'$P_{\rm vap}^{\rm sat}$ (MPa)')
#axarr[1,1].set_ylabel(r'$\Delta H_{\rm v}$ (kJ/mol)')

#axarr[0,0].set_xlabel(r'$\log_{10}(N_{\rm eff})$')
#axarr[0,1].set_xlabel(r'$\log_{10}(N_{\rm eff})$')
#axarr[1,0].set_xlabel(r'$\log_{10}(N_{\rm eff})$')
#axarr[1,1].set_xlabel(r'$\log_{10}(N_{\rm eff})$')
#
#axarr[0,0].set_title(r'$\rho_{\rm liq}^{\rm sat}$')
#axarr[0,1].set_title(r'$\rho_{\rm vap}^{\rm sat}$')
#axarr[1,0].set_title(r'$P_{\rm vap}^{\rm sat}$')
#axarr[1,1].set_title(r'$\Delta H_{\rm v}$')

axarr2.set_xlabel(r'$T_{\rm r}$')
axarr2.set_ylabel(r'$\log_{10}(K^{\rm eff}_{\rm snaps})$')
axarr2.set_xlim([0.6,1.])
axarr2.set_xticks([0.65,0.75,0.85,0.95])
axarr2.plot([],[],'bs',mfc='None',label='Liquid')
axarr2.plot([],[],'ro',mfc='None',label='Vapor')
axarr2.plot([0.6,1],[np.log10(50),np.log10(50)],'k--',label=r'$N_{\rm eff} = 50$')
axarr2.legend()

axarr2.set_title(r'$\lambda_{\rm ref} \neq \lambda_{\rm rr}$')
axarr2.set_title(r'TraPPE $\Leftrightarrow$ MiPPE-gen')

fig2.savefig('refFF_to_rrFF_lam_'+lam+'_Neff.pdf')  

phase_list = ['Liquid','Vapor']

for iax, phase in enumerate(phase_list):
    axarr4[iax].set_xlabel(r'$T_{\rm r}$')
    
    axarr4[iax].set_ylabel(r'$\log_{10}(K^{\rm eff}_{\rm snaps})$')
    
    axarr4[iax].set_title(phase)

    axarr4[iax].set_xlim([0.6,1.])
    axarr4[iax].set_xticks([0.65,0.75,0.85,0.95])
    
    axarr4[iax].set_ylim([0,5.7])
    axarr4[iax].set_yticks([0,1,2,3,4,5])

    axarr4[iax].plot([0.6,1],[np.log10(50),np.log10(50)],'k--')#,label=r'$N_{\rm eff} = 50$')

axarr4[1].text(0.75,1.85,r'$K^{\rm eff}_{\rm snaps} = 50$')
    
rerun_list = {'TraPPE':['PotoffGen','NERD'],'PotoffGen':['TraPPE','PotoffSL']}

for referenceFF in ['TraPPE','PotoffGen']:
    
    for rerunFF in rerun_list[referenceFF]: 
        
#for refFFtorrFF in ['TraPPEtoPotoffGen','PotoffGentoTraPPE','TraPPEtoNERD','PotoffGentoPotoffSL']
        
        if (referenceFF == 'TraPPE' and rerunFF == 'PotoffGen') or (referenceFF == 'PotoffGen' and rerunFF == 'TraPPE'):
#            axarr4[1].plot([],[],color_list[rerunFF]+shape_list[rerunFF],markersize=10,mfc='None',label=label_list[referenceFF+'to'+rerunFF])
#            axarr4[1].plot([],[],shape_list[rerunFF],color=color_list[rerunFF],markersize=10,mfc='None',label=label_list[referenceFF+'to'+rerunFF])
            axarr4[1].plot([],[],shape_list[rerunFF],color='k',markersize=10,mfc='None',label=label_list[referenceFF+'to'+rerunFF])            
        else:
#            axarr4[1].plot([],[],color_list[rerunFF]+shape_list[rerunFF],markersize=10,mfc=color_list[rerunFF],label=label_list[referenceFF+'to'+rerunFF])
#            axarr4[1].plot([],[],shape_list[rerunFF],color=color_list[rerunFF],markersize=10,mfc=color_list[rerunFF],label=label_list[referenceFF+'to'+rerunFF])
            axarr4[1].plot([],[],shape_list[rerunFF],color='k',markersize=10,mfc='k',label=label_list[referenceFF+'to'+rerunFF])

for compound in ['2MPropane','22DMPropane','224TMPentane','234TMPentane','23DMButane','22DMButane','33DMHexane','3M3EPentane']:
    
    axarr4[1].plot([],[],'-',color=color_list[compound],linewidth=12,label=label_list[compound])

lgd4 = axarr4[1].legend(loc='lower center', bbox_to_anchor=(-0.15, -0.5),
          ncol=3,numpoints=1,handlelength=1,handletextpad=0.5,columnspacing=0.5,frameon=True,borderaxespad=0)
         
fig4.savefig('refFF_to_rrFF_Neff_alt.pdf',bbox_extra_artists=(lgd4,),bbox_inches='tight') 


for referenceFF in reference_list:
    
    for rerunFF in rerun_list[referenceFF]: 
        
        axarr[1,1].plot([],[],'k'+shape_list[rerunFF],markersize=10,mfc='None',label=label_list[referenceFF+'to'+rerunFF])
        
for compound in compound_list:
    
    axarr[1,1].plot([],[],'-',color=color_list[compound],linewidth=12,label=label_list[compound])

lgd = axarr[1,1].legend(loc='lower center', bbox_to_anchor=(-0.3, -0.51),
          ncol=3,numpoints=1,handlelength=2,handletextpad=0.5,columnspacing=0.5,frameon=True,borderaxespad=0)
         
#plt.tight_layout()           

fig.savefig('refFF_to_rrFF_lam_'+lam+'_error.pdf',bbox_extra_artists=(lgd,),bbox_inches='tight')   

fig3, axarr3 = plt.subplots(nrows=1,ncols=2,figsize=[12,6])

boxplot_Neff_liq = []
boxplot_Neff_vap = []

for Tr in Tr_bins:
    
    boxplot_Neff_liq.append(np.log10(Neff_liq_bins[Tr]))
    boxplot_Neff_vap.append(np.log10(Neff_vap_bins[Tr]))

axarr3[0].boxplot(boxplot_Neff_liq,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False)
axarr3[0].boxplot(boxplot_Neff_vap,positions=Tr_bins,widths=0.02,notch=True,manage_xticks=False)
 