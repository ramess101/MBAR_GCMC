# -*- coding: utf-8 -*-
"""
Evaluate argon histograms
"""

import numpy as np
import matplotlib.pyplot as plt
from pymbar import MBAR
from MBAR_GCMC_class import MBAR_GCMC
from scipy import stats

# Physical constants
N_A = 6.02214086e23 #[/mol]
nm3_to_ml = 10**21
bar_nm3_to_kJ_per_mole = 0.0602214086
Ang3tom3 = 10**-30
gmtokg = 1e-3
kb = 1.3806485e-23 #[J/K]
Jm3tobar = 1e-5
Rg = kb*N_A #[J/mol/K]

root_path='argon/'

filepaths=[]
for i in np.arange(1,8):
    filepaths.append(root_path+'his'+str(i)+'a.dat')
    
Mw_argon=39.948

VLE_RP = np.loadtxt('argon/NIST.dat',skiprows=1)
Tsat_RP = VLE_RP[:,0] #[K]
Psat_RP = VLE_RP[:,1] #[bar]
rhol_RP = VLE_RP[:,2] #[kg/m3]
rhov_RP = VLE_RP[:,3] #[kg/m3]

#VLE_Mick=np.loadtxt('argon/Mick_lit_argon.txt',skiprows=1)
#Tsat_lit = VLE_Mick[:,0]
#Psat_lit = VLE_Mick[:,1]
#rhol_lit = VLE_Mick[:,2]
#rhov_lit=VLE_Mick[:,3]

VLE_workshop=np.loadtxt('argon/workshop/argon.dat',skiprows=1)
Tsat_lit = VLE_workshop[:,0]
Psat_lit = VLE_workshop[:,1]
rhol_lit = VLE_workshop[:,2]
rhov_lit=VLE_workshop[:,3]

Psat_Potoff = Psat_lit.copy()
Tsat_Potoff = Tsat_lit.copy()

MBAR_GCMC_argon = MBAR_GCMC(root_path,filepaths,Mw_argon,compare_literature=True)
MBAR_GCMC_argon.plot_histograms()
MBAR_GCMC_argon.plot_2dhistograms()
MBAR_GCMC_argon.solve_VLE(Tsat_lit)
MBAR_GCMC_argon.plot_VLE(compare_RP=False,Tsat_RP=Tsat_RP,Psat_RP=Psat_RP,rhol_RP=rhol_RP,rhov_RP=rhov_RP,Tsat_lit=Tsat_lit,rhol_lit=rhol_lit,rhov_lit=rhov_lit,Psat_lit=Psat_lit)

### Scratchwork for debugging

#Temp_sim, u_kn_sim,f_k_sim,sumN_k = MBAR_GCMC_argon.Temp_sim, MBAR_GCMC_argon.u_kn_sim,MBAR_GCMC_argon.f_k_sim,MBAR_GCMC_argon.sumN_k
#nTsim, U_flat, Nmol_flat,Ncut = MBAR_GCMC_argon.nTsim, MBAR_GCMC_argon.U_flat, MBAR_GCMC_argon.Nmol_flat, MBAR_GCMC_argon.Ncut
#
#Temp_IG = np.min(MBAR_GCMC_argon.Temp_sim[MBAR_GCMC_argon.mu_sim == MBAR_GCMC_argon.mu_sim.min()])
##mu_IG = np.linspace(2.*MBAR_GCMC_argon.mu_sim.min(),5.*MBAR_GCMC_argon.mu_sim.min(),10)
#mu_IG = np.linspace(MBAR_GCMC_argon.mu_opt[MBAR_GCMC_argon.Temp_VLE==Temp_IG],5.*MBAR_GCMC_argon.mu_sim.min(),10)
##mu_IG = np.linspace(1.5*MBAR_GCMC_argon.mu_sim.min(),5.*MBAR_GCMC_argon.mu_sim.min(),20)
#
#N_k_all = MBAR_GCMC_argon.K_sim[:]
#N_k_all.extend([0]*len(mu_IG))
#
#u_kn_IG = np.zeros([len(mu_IG),sumN_k])
#u_kn_all = np.concatenate((u_kn_sim,u_kn_IG))
#
#f_k_guess = np.concatenate((f_k_sim,np.zeros(len(mu_IG))))
#
#for jT, mu in enumerate(mu_IG):
#    
#    u_kn_all[nTsim+jT,:] = MBAR_GCMC_argon.U_to_u(U_flat,Temp_IG,mu,Nmol_flat)
#
#mbar = MBAR(u_kn_all,N_k_all,initial_f_k=f_k_guess)
#    
#sumW_IG = np.sum(mbar.W_nk[:,nTsim:][Nmol_flat<Ncut],axis=0)
# 
#Nmol_IG = np.sum(mbar.W_nk[:,nTsim:][Nmol_flat<Ncut].T*Nmol_flat[Nmol_flat<Ncut],axis=1)/sumW_IG
##print(sumW_IG,Nmol_IG)
##print(mbar.W_nk[:,nTsim:][Nmol_flat<Ncut].T)
##print(mbar.W_nk[:,nTsim:][Nmol_flat<Ncut].T*Nmol_flat[Nmol_flat<Ncut])
#### Store previous solutions to speed-up future convergence of MBAR
#Deltaf_ij = mbar.getFreeEnergyDifferences(return_theta=False)[0]
#f_k_IG = Deltaf_ij[0,nTsim:]
#press_IG = Nmol_IG*kb*Temp_IG/MBAR_GCMC_argon.Vbox_sim[0] / Ang3tom3 * Jm3tobar
#Psat = (press_IG[1]/kb/Temp_IG*MBAR_GCMC_argon.Vbox_sim[0])
#Psat = (1. - f_k_IG[0] + f_k_IG[1])*kb*Temp_IG/MBAR_GCMC_argon.Vbox_sim[0] / Ang3tom3 * Jm3tobar
#Psat = (Nmol_IG[1] - f_k_IG[0] + f_k_IG[1])*kb*Temp_IG/MBAR_GCMC_argon.Vbox_sim[0] / Ang3tom3 * Jm3tobar
##        Psat = (- f_k_IG[0] + f_k_IG[1])*kb*Temp_IG/self.Vbox_sim[0] / Ang3tom3 * Jm3tobar
#print(f_k_sim,f_k_guess[:nTsim+1],Deltaf_ij[0,:nTsim],f_k_IG)#,Nmol_IG,press_IG,Psat)
#
#fit=stats.linregress(Nmol_IG[mu_IG<2.*MBAR_GCMC_argon.mu_sim.min()],-f_k_IG[mu_IG<2.*MBAR_GCMC_argon.mu_sim.min()])
#Nmol_plot = np.linspace(Nmol_IG.min(),Nmol_IG.max(),50)
#lnXi_plot = fit.intercept + fit.slope*Nmol_plot
#
#int_workshop=[-2.4549]
#
#plt.figure(figsize=[6,6])
#plt.plot(Nmol_IG,-f_k_IG,'bo',mfc='None',label='MBAR-GCMC')
#plt.plot(Nmol_plot,lnXi_plot,'k-',label='Linear fit')
#plt.plot(0,int_workshop,'rx',label='msoroush intercept')
#plt.xlabel('Number of Molecules')
#plt.ylabel(r'$\ln(\Xi)$')
##plt.ylim([-2.455,-2.405])
#plt.legend()
#plt.show()
#
#print('Slope for ideal gas is 1, actual slope is: '+str(fit.slope))
#print('Intercept for absolute pressure is:'+str(fit.intercept))
#
#MBAR_GCMC_argon.abs_press_int, MBAR_GCMC_argon.Temp_IG, MBAR_GCMC_argon.f_k_IG, MBAR_GCMC_argon.Nmol_IG = fit.intercept, Temp_IG, f_k_IG, Nmol_IG
#
#f_k_opt, nTsim, Temp_VLE, Vbox, abs_press_int, Temp_IG = MBAR_GCMC_argon.f_k_opt, MBAR_GCMC_argon.nTsim, MBAR_GCMC_argon.Temp_VLE, MBAR_GCMC_argon.Vbox_sim[0], MBAR_GCMC_argon.abs_press_int, MBAR_GCMC_argon.Temp_IG
##        f_k_opt, nTsim, Temp_VLE, Vbox = self.f_k_opt, self.nTsim, self.Temp_VLE, self.Vbox_sim[0]                                                                                                                       
##        print(f_k_opt)
#print(f_k_opt[:nTsim])
#print(f_k_opt[nTsim:])
#abs_press_int = np.array(int_workshop)
#
##print(abs_press_int)
##Psat = kb * Temp_VLE / Vbox * (-f_k_opt[nTsim:] - abs_press_int) / Ang3tom3 * Jm3tobar
##abs_press_int = -19.7
#Psat1 = kb * Temp_VLE / Vbox * (-f_k_opt[nTsim:]-np.log(2.)) / Ang3tom3 * Jm3tobar
#Psat2 = kb * Temp_VLE / Vbox * (-abs_press_int) / Ang3tom3 * Jm3tobar
#Psat = Psat1 + Psat2
##        print(Psat)
##        print(kb*Temp_VLE/Vbox*(-f_k_opt[nTsim:])/Ang3tom3*Jm3tobar)
##        print(kb*Temp_VLE/Vbox*(-abs_press_int)/Ang3tom3*Jm3tobar)
##        print(kb*Temp_IG/Vbox*(-abs_press_int)/Ang3tom3*Jm3tobar)
##        print((Psat-Psat_Potoff)/kb / Temp_VLE * Vbox * Ang3tom3 / Jm3tobar)
##        
#C_Potoff =  Psat_Potoff - kb*Temp_VLE/Vbox*(-f_k_opt[nTsim:])/Ang3tom3*Jm3tobar
#lnZ0_Potoff = C_Potoff / kb / Temp_VLE * Vbox * Ang3tom3 / Jm3tobar
#avglnZ0_Potoff = np.mean(lnZ0_Potoff)
##print(lnZ0_Potoff,avglnZ0_Potoff)
#plt.plot(Temp_VLE,lnZ0_Potoff,'r:')
#plt.plot(Temp_VLE,[-abs_press_int]*len(Temp_VLE),'g--')
#plt.plot(Temp_VLE,[avglnZ0_Potoff]*len(Temp_VLE),'b-')
#plt.plot(Temp_VLE,-1.*np.array([int_workshop*len(Temp_VLE)]).T,'k-')
#plt.show()
#
#P_error = (Psat_Potoff - Psat)
#f_error = P_error/ kb / Temp_VLE * Vbox * Ang3tom3 / Jm3tobar
#
#plt.plot(Temp_VLE,P_error,'bo')
#plt.show()
#
#plt.plot(Temp_VLE,f_error,'ro')
#plt.show()
#
##        Psat = Psat1 + kb*Temp_VLE/Vbox*(avglnZ0_Potoff)/Ang3tom3*Jm3tobar
#
##        Psat = (self.Nmol_IG[1] - f_k_opt[nTsim:] + self.f_k_IG[1])*kb*Temp_VLE/self.Vbox_sim[0] / Ang3tom3 * Jm3tobar
##        Psat = (self.Nmol_IG[1] - f_k_opt[nTsim:] + self.f_k_IG[1])*kb*Temp_VLE/self.Vbox_sim[0] / Ang3tom3 * Jm3tobar
##        print(f_k_opt[nTsim:],Temp_VLE.shape,self.f_k_IG,self.Nmol_IG,Psat)
##        print(self.Nmol_IG[1] - f_k_opt[nTsim:] + self.f_k_IG[1])
##        
#plt.plot(Temp_VLE,f_k_opt[nTsim:])
#plt.plot([MBAR_GCMC_argon.Temp_IG]*len(MBAR_GCMC_argon.f_k_IG),MBAR_GCMC_argon.f_k_IG,'kx')
#plt.plot(Temp_sim,f_k_opt[:nTsim],'rs',mfc='None')
#plt.show()
#
#Psat_ig = MBAR_GCMC_argon.rhovap * Rg * Temp_VLE / MBAR_GCMC_argon.Mw / gmtokg * Jm3tobar
#print(Psat_ig)
#plt.figure(figsize=[6,6])
#plt.plot(1./Temp_VLE[Psat>0],np.log10(Psat[Psat>0]),'ro',mfc='None',label='MBAR-GCMC')
#plt.plot(1./Temp_VLE,np.log10(Psat2),'bv',mfc='None',label='MBAR-GCMC, IG')
#plt.plot(1./Tsat_Potoff,np.log10(Psat_Potoff),'ks',mfc='None',label='Potoff')
#plt.plot(1./Temp_VLE,np.log10(Psat_ig),'g--',label='Ideal gas')
#plt.xlabel(r'$1/T (K)$')
#plt.ylabel(r'$\log_{\rm 10}(P_{\rm v}^{\rm sat}/\rm bar)$')
#plt.xticks([1./Temp_VLE.max(),1./Temp_VLE.min()])
#plt.legend()
#plt.show()
#
#plt.figure(figsize=[6,6])
#plt.plot(Temp_VLE,(Psat-Psat_Potoff)/Psat_Potoff*100.,'ro',mfc='None')
#plt.xlabel(r'$T (K)$')
#plt.ylabel(r'$(P_{\rm v,MBAR}^{\rm sat} - P_{\rm v,Potoff}^{\rm sat})/P_{\rm v,Potoff}^{\rm sat}) \times 100$ % (bar)')
##        plt.xticks([Temp_VLE.max(),1./Temp_VLE.min()])
#plt.legend()
#plt.show()