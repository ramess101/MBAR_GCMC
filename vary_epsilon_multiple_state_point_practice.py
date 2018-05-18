# -*- coding: utf-8 -*-
"""
MBAR GCMC
This code was supposed to scale epsilon, but I just realized that for hexane
the total energy also depends on intramolecular contributions, such that I 
cannot simply scale the energy.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pymbar import MBAR
import sys
import pdb
from Golden_search_multi import GOLDEN_multi

# Physical constants
N_A = 6.02214086e23 #[/mol]
nm3_to_ml = 10**21
bar_nm3_to_kJ_per_mole = 0.0602214086
Ang3tom3 = 10**-30
gmtokg = 1e-3
kb = 1.3806485e-23 #[J/K]
Jm3tobar = 1e-5
Rg = 8.314472 #[J/mol/K]

Mw_hexane  = 12.0109*6.+1.0079*(2.*6.+2.) #[gm/mol]

Tsat_Potoff = np.array([500,490,480,470,460,450,440,430,420,410,400,390,380,370,360,350,340,330,320])
rhol_Potoff = np.array([366.018,395.855,422.477,444.562,463.473,480.498,496.217,510.897,524.727,537.821,550.308,562.197,573.494,584.216,594.369,604.257,614.026,623.44,631.598])
rhov_Potoff = np.array([112.352,90.541,72.249,58.283,47.563,39.028,32.053,26.27,21.441,17.397,14.013,11.19,8.846,6.913,5.331,4.05,3.023,2.213,1.584])     
Psat_Potoff = np.array([27.697,23.906,20.521,17.529,14.889,12.563,10.522,8.738,7.19,5.857,4.717,3.753,2.946,2.279,1.734,1.296,0.949,0.68,0.476])
Zsat_Potoff = Psat_Potoff / rhov_Potoff / Tsat_Potoff * Mw_hexane / Rg * gmtokg / Jm3tobar

def U_to_u(Uint,Temp,mu,Nmol):
    '''
    Converts internal energy, temperature, chemical potential, and number of molecules into reduced potential energy 
    
    inputs:
        Uint: internal energy (K)
        Temp: Temperature (K)
        mu: Chemical potential (K)
        Nmol: number of molecules
    outputs:
        Ureduced: reduced potential energy
    '''
    beta = 1./Temp #[1/K]
    Ureduced = beta*(Uint) - beta*mu*Nmol  #[dimensionless]
    return Ureduced

root_path = 'H:/MBAR_GCMC/hexane_Potoff/'
Temp_range = ['510','470','430','480','450','420','390','360','330']
hist_num=['1','2','3','4','5','6','7','8','9']

#Temp_range = ['510','480']
#hist_num=['1','4']

nSnapshots=90000

U_all = np.zeros([len(Temp_range),nSnapshots])
N_all = np.zeros([len(Temp_range),nSnapshots])

for iT, Temp in enumerate(Temp_range):
    
    hist_name='/his'+hist_num[iT]+'a.dat' #Only if loading hexane_Potoff
    
    NU = np.loadtxt(root_path+Temp+hist_name,skiprows=1)
    
    U = NU[:,1]
    N = NU[:,0]
    
    U_all[iT] = U
    N_all[iT] = N
         
Temp_sim = np.array([510,470,430,480,450,420,390,360,330])  
mu_sim = np.array([-4127., -4127., -4127., -3980., -3880., -3800., -3725., -3675.,-3625.])  

#Temp_sim = np.array([510,480])  
#mu_sim = np.array([-4127., -3980.]) 

Lbox = 35. #[Angstrom]
Vbox = Lbox**3 #[Angstrom^3]

#        # N_k contains the number of snapshots from each state point, including the unsampled points
#        N_k = np.array(self.K_all.append(0))
#        # N_kn contains all of the Number of molecules in 1-d array
#        Nmol_kn = np.array(self.N_data_all)
#        Nmol_kn = Nmol_kn.reshape([Nmol_kn.size])
#        
#        # u_kn contains all the reduced potential energies in 1-d array, including the unsampled points
#        u_kn = np.zeros(len(self.Temp_all)+1,self.K_all.sum())

N_k = [nSnapshots]*len(Temp_sim)
sumN_k = nSnapshots*len(Temp_sim)
Nmol_kn = N_all.reshape([N_all.size])
u_kn_sim = np.zeros([len(Temp_sim),nSnapshots*len(Temp_sim)])

for iT, (Temp, mu) in enumerate(zip(Temp_sim, mu_sim)):
    
    for jT in range(len(Temp_sim)):
        
        jstart = nSnapshots*jT
        jend = jstart+nSnapshots
        
        u_kn_sim[iT,jstart:jend] = U_to_u(U_all[jT],Temp,mu,N_all[jT])
        
mbar = MBAR(u_kn_sim,N_k)

Deltaf_ij = mbar.getFreeEnergyDifferences(return_theta=False)[0]
#print "effective sample numbers"
#print (mbar.computeEffectiveSampleNumber())
#print('\nWhich is approximately '+str(mbar.computeEffectiveSampleNumber()/sumN_k*100.)+'%')

f_k_sim = Deltaf_ij[0,:]

#mbar2 = MBAR(u_kn_sim,N_k,initial_f_k=f_k_sim)
#
#Deltaf_ij2 = mbar2.getFreeEnergyDifferences(return_theta=False)[0]
#print "effective sample numbers"
#print (mbar2.computeEffectiveSampleNumber())
#print('\nWhich is approximately '+str(mbar2.computeEffectiveSampleNumber()/sumN_k*100.)+'%')

#Nmolk, dNmolk = mbar.computeExpectations(Nmol_kn) # Average number of molecules
#Nmolk_alt = np.zeros(len(N_k))
#for i in range(len(N_k)):
#    Nmolk_alt[i] = np.sum(mbar.W_nk[:,i]*Nmol_kn)

#print(Nmolk)
#print(Nmolk_alt)

### This was how I want to implement it, but I am not sure how to make 
Temp_VLE = np.array([500,490,480,470,460,450,440,430,420,410,400,390,380,370,360,350,340,330,320])
#Temp_VLE = np.array([450,400,370,320])

Temp_all = np.concatenate((Temp_sim,Temp_VLE))
N_k.extend([0]*len(Temp_VLE))

u_kn_VLE = np.zeros([len(Temp_VLE),nSnapshots*len(Temp_sim)])
u_kn = np.concatenate((u_kn_sim,u_kn_VLE))

f_k_guess = np.concatenate((f_k_sim,np.zeros(len(Temp_VLE))))

#np.array([-3900.]*len(Temp_VLE))

#for iT in range(len(Temp_sim)):
#    
#    istart = nSnapshots*iT
#    iend = istart+nSnapshots
#    
#    for jT, (Temp, mu) in enumerate(zip(Temp_VLE, mu_VLE)):
#        
#        u_kn[len(Temp_sim)+jT,istart:iend] = U_to_u(U_all[iT],Temp,mu,N_all[iT])
#
#mbar = MBAR(u_kn,N_k)
#Ncut = 81
#sqdeltaWi = np.zeros(len(Temp_VLE))
#
#for jT in range(len(Temp_VLE)):
#    sumWliq = np.sum(mbar.W_nk[:,jT][Nmol_kn>Ncut])
#    sumWvap = np.sum(mbar.W_nk[:,jT][Nmol_kn<=Ncut])
#    sqdeltaWi[jT] = (sumWliq - sumWvap)**2
#             
#print(sqdeltaWi)

#def sqdeltaW(mu_VLE):
#
#    for iT in range(len(Temp_sim)):
#        
#        istart = nSnapshots*iT
#        iend = istart+nSnapshots
#        
#        for jT, (Temp, mu) in enumerate(zip(Temp_VLE, mu_VLE)):
#            
#            u_kn[len(Temp_sim)+jT,istart:iend] = U_to_u(U_all[iT],Temp,mu,N_all[iT])
#    
#    mbar = MBAR(u_kn,N_k,initial_f_k=f_k_guess)
#    Ncut = 81
#    sqdeltaW = np.zeros(len(Temp_VLE))
#    
#    for jT in range(len(Temp_VLE)):
#        sumWliq = np.sum(mbar.W_nk[:,jT][Nmol_kn>Ncut])
#        sumWvap = np.sum(mbar.W_nk[:,jT][Nmol_kn<=Ncut])
#        sqdeltaW[jT] = (sumWliq - sumWvap)**2
#       
#### Could be advantageous to store this. But needs to be outside the function. Either as a global variable or within the optimizer    
#### I guess within the class I can store this as self.f_k_guess and update it each time the function is called     
##    Deltaf_ij = mbar.getFreeEnergyDifferences(return_theta=False)[0]
##    f_k_guess = Deltaf_ij[0,:]
#                 
#    return sqdeltaW

U_all_flat = U_all.flatten()
N_all_flat = N_all.flatten()

jT0 = len(Temp_sim)
Ncut = 81

eps_ratio = 1.05

def sqdeltaW(mu_VLE):
    
    for jT, (Temp, mu) in enumerate(zip(Temp_VLE, mu_VLE)):
        
        u_kn[jT0+jT,:] = U_to_u(eps_ratio*U_all_flat,Temp,mu,N_all_flat)
    
    mbar = MBAR(u_kn,N_k,initial_f_k=f_k_guess)

    sqdeltaW_VLE = np.zeros(len(Temp_VLE))
    
    for jT in range(len(Temp_VLE)):
        sumWliq = np.sum(mbar.W_nk[:,jT0+jT][Nmol_kn>Ncut])
        sumWvap = np.sum(mbar.W_nk[:,jT0+jT][Nmol_kn<=Ncut])
        sqdeltaW_VLE[jT] = (sumWliq - sumWvap)**2
       
### Could be advantageous to store this. But needs to be outside the function. Either as a global variable or within the optimizer    
### I guess within the class I can store this as self.f_k_guess and update it each time the function is called     
#    Deltaf_ij = mbar.getFreeEnergyDifferences(return_theta=False)[0]
#    f_k_guess = Deltaf_ij[0,:]
                 
    return sqdeltaW_VLE

### Optimization of mu
### Bounds for mu
mu_sim_low = np.ones(len(Temp_VLE))*mu_sim.min()
mu_sim_high = np.ones(len(Temp_VLE))*mu_sim.max()

Temp_sim_mu_low = Temp_sim[np.argmin(mu_sim)]
Temp_sim_mu_high = Temp_sim[np.argmax(mu_sim)]

### Guess for mu
mu_guess = lambda Temp: mu_sim_high + (mu_sim_low - mu_sim_high)/(Temp_sim_mu_low-Temp_sim_mu_high) * (Temp-Temp_sim_mu_high)

mu_VLE_guess = mu_guess(Temp_VLE)

print(r'$(\Delta W)^2$ for $\mu_{\rm guess}$ =')
print(sqdeltaW(mu_VLE_guess))

mu_opt = GOLDEN_multi(sqdeltaW,mu_VLE_guess,mu_sim_low,mu_sim_high,TOL=0.001,maxit=30)
sqdeltaW_opt = sqdeltaW(mu_opt)

plt.plot(Temp_VLE,mu_opt,'k-')
plt.plot(Temp_sim,mu_sim,'ro',mfc='None')
plt.xlabel(r'$T$ (K)')
plt.ylabel(r'$\mu_{\rm opt}$ (K)')
plt.xlim([300,550])
plt.ylim([-4200,-3600])
plt.show()

plt.plot(Temp_VLE,sqdeltaW_opt,'ko')
plt.xlabel(r'$T$ (K)')
plt.ylabel(r'$(\Delta W^{\rm sat})^2$')
plt.show()

###Scan of mu
#mu_scan = np.linspace(-4120,-4000,20)
#mu_scan = np.linspace(-3750,-3680,10)
#sqdeltaW_all = np.zeros([len(Temp_VLE),len(mu_scan)])
#
#for imu, mui in enumerate(mu_scan):
#  
#    mu_array = [mui]*len(Temp_VLE)
#    sqdeltaW_all[:,imu] = sqdeltaW(mu_array)
##    print(imu)
#    
#plt.plot(mu_scan,sqdeltaW_all.T)
#plt.show()

Nliq = np.zeros(len(Temp_VLE))
Nvap = np.zeros(len(Temp_VLE))
    
for jT, (Temp, mu) in enumerate(zip(Temp_VLE, mu_opt)):
    
    u_kn[jT0+jT,:] = U_to_u(U_all_flat,Temp,mu,N_all_flat)
    
    mbar = MBAR(u_kn,N_k,initial_f_k=f_k_guess)
    
    sumWliq = np.sum(mbar.W_nk[:,jT0+jT][Nmol_kn>Ncut])
    sumWvap = np.sum(mbar.W_nk[:,jT0+jT][Nmol_kn<=Ncut])

    Nliq[jT] = np.sum(mbar.W_nk[:,jT0+jT][Nmol_kn>Ncut]*Nmol_kn[Nmol_kn>Ncut])/sumWliq #Must renormalize by the liquid or vapor phase
    Nvap[jT] = np.sum(mbar.W_nk[:,jT0+jT][Nmol_kn<=Ncut]*Nmol_kn[Nmol_kn<=Ncut])/sumWvap

rholiq = Nliq/Vbox * Mw_hexane / N_A * gmtokg / Ang3tom3 #[kg/m3]
rhovap = Nvap/Vbox * Mw_hexane / N_A * gmtokg / Ang3tom3 #[kg/m3]
   
plt.plot(rhovap[Temp_VLE<425],Temp_VLE[Temp_VLE<425],'ro',label='MBAR-GCMC')
plt.plot(rholiq[Temp_VLE<425],Temp_VLE[Temp_VLE<425],'ro')
plt.plot(rhov_Potoff,Tsat_Potoff,'ks',mfc='None',label='Potoff')
plt.plot(rhol_Potoff,Tsat_Potoff,'ks',mfc='None')
plt.xlabel(r'$\rho$ (kg/m$^3$)')
plt.ylabel(r'$T$ (K)')
plt.title(r'$\epsilon = $'+str(eps_ratio)+r' $\epsilon_{\rm ref}$')
plt.xlim([0,650])
plt.ylim([320,520])
plt.legend()
plt.show()
 
    
#for imu, mui in enumerate(mu_scan):
#  
#    ui0 = U_to_u(U0,Ti,mui,N0) #Keeping epsilon and sigma constant for now
#    ui1 = U_to_u(U1,Ti,mui,N1) #Keeping epsilon and sigma constant for now
#                
#    u_kn[nStates,:len(u00)] = ui0
#    u_kn[nStates,len(u00):] = ui1
#    
#    mbar = MBAR(u_kn,N_k)
#    
#    (Deltaf_ij, dDeltaf_ij, Theta_ij) = mbar.getFreeEnergyDifferences(return_theta=True)
#    
#    sumWliq = np.sum(mbar.W_nk[:,nStates][N_kn>Ncut])
#    sumWvap = np.sum(mbar.W_nk[:,nStates][N_kn<=Ncut])
#    sqdeltaWi[imu] = (sumWliq - sumWvap)**2
#
##    Nliq[imu], dNliq = mbar.computeExpectations(N0[N0>Ncut])
##    Nvap[imu], dNvap = mbar.computeExpectations(N0[N0<Ncut])
#    
#    Nliq[imu] = np.sum(mbar.W_nk[:,nStates][N_kn>Ncut]*N_kn[N_kn>Ncut])/sumWliq #Must renormalize by the liquid or vapor phase
#    Nvap[imu] = np.sum(mbar.W_nk[:,nStates][N_kn<=Ncut]*N_kn[N_kn<=Ncut])/sumWvap



#        
#
#T0 = 510. #[K]
#mu0 = -4127. #[K]
#
#T1 = 480. #[K]
#mu1 = -3980 #[K]
#
#u00=U_to_u(U0,T0,mu0,N0)
#u01=U_to_u(U1,T0,mu0,N1)
#u10=U_to_u(U0,T1,mu1,N0)
#u11=U_to_u(U1,T1,mu1,N1)
#
#T2 = 480. #[K]
#mu2 = -4023 #[K]
#
#u20 = U_to_u(U0,T2,mu2,N0)
#u21 = U_to_u(U1,T2,mu2,N1)
#      
#N_k = np.array([len(u00),len(u11),0]) # The number of samples from each state
#N_K = np.sum(N_k)
#              
#u_kn = np.zeros([3,len(u00)+len(u11)])
#u_kn[0,:len(u00)] = u00
#u_kn[0,len(u00):] = u01     
#u_kn[1,:len(u00)] = u10
#u_kn[1,len(u00):] = u11    
#u_kn[2,:len(u00)] = u20
#u_kn[2,len(u00):] = u21   
#     
#N_kn = np.zeros(len(u00)+len(u11))
#N_kn[:len(u00)] = N0
#N_kn[len(u00):] = N1     
#              
#mbar = MBAR(u_kn,N_k)
#
#Deltaf_ij = mbar.getFreeEnergyDifferences(return_theta=False)
#print "effective sample numbers"
#print (mbar.computeEffectiveSampleNumber())
#print('\nWhich is approximately '+str(mbar.computeEffectiveSampleNumber()/N_K*100.)+'%')
#
#NAk, dNAk = mbar.computeExpectations(N_kn) # Average number of molecules
#NAk_alt = np.zeros(len(N_k))
#for i in range(len(N_k)):
#    NAk_alt[i] = np.sum(mbar.W_nk[:,i]*N_kn)
#    
#print(NAk)
#
#Nscan = np.arange(60,100)
#
#sqdeltaW0 = np.zeros(len(Nscan))
#
#for iN, Ni in enumerate(Nscan):
#
#    sumWliq = np.sum(mbar.W_nk[:,0][N_kn>Ni])
#    sumWvap = np.sum(mbar.W_nk[:,0][N_kn<=Ni])
#    sqdeltaW0[iN] = (sumWliq - sumWvap)**2
#
#plt.plot(Nscan,sqdeltaW0,'ko')
#plt.xlabel(r'$N_{\rm cut}$')
#plt.ylabel(r'$(\Delta W_0^{\rm sat})^2$')
#plt.show()
#                        
#Ncut = Nscan[np.argmin(sqdeltaW0)]
#
#mu_scan = np.linspace(-4150,-3950,200)
#sqdeltaWi = np.zeros(len(mu_scan))
#Nliq = np.zeros(len(mu_scan))
#Nvap = np.zeros(len(mu_scan))
#
#Ti = 460 #[K]
#nStates = len(N_k)-1
#                                     
#for imu, mui in enumerate(mu_scan):
#  
#    ui0 = U_to_u(U0,Ti,mui,N0) #Keeping epsilon and sigma constant for now
#    ui1 = U_to_u(U1,Ti,mui,N1) #Keeping epsilon and sigma constant for now
#                
#    u_kn[nStates,:len(u00)] = ui0
#    u_kn[nStates,len(u00):] = ui1
#    
#    mbar = MBAR(u_kn,N_k)
#    
#    (Deltaf_ij, dDeltaf_ij, Theta_ij) = mbar.getFreeEnergyDifferences(return_theta=True)
#    
#    sumWliq = np.sum(mbar.W_nk[:,nStates][N_kn>Ncut])
#    sumWvap = np.sum(mbar.W_nk[:,nStates][N_kn<=Ncut])
#    sqdeltaWi[imu] = (sumWliq - sumWvap)**2
#
##    Nliq[imu], dNliq = mbar.computeExpectations(N0[N0>Ncut])
##    Nvap[imu], dNvap = mbar.computeExpectations(N0[N0<Ncut])
#    
#    Nliq[imu] = np.sum(mbar.W_nk[:,nStates][N_kn>Ncut]*N_kn[N_kn>Ncut])/sumWliq #Must renormalize by the liquid or vapor phase
#    Nvap[imu] = np.sum(mbar.W_nk[:,nStates][N_kn<=Ncut]*N_kn[N_kn<=Ncut])/sumWvap
#          
#plt.plot(mu_scan,sqdeltaWi,'k-')
#plt.xlabel(r'$\mu$ (K)')
#plt.ylabel(r'$(\Delta W_1^{\rm sat})^2$')
#plt.show()
#
#plt.plot(mu_scan,Nliq,'r-',label='Liquid')
#plt.plot(mu_scan,Nvap,'b--',label='Vapor')
#plt.plot([mu_scan[np.argmin(sqdeltaWi)],mu_scan[np.argmin(sqdeltaWi)]],[0,np.max(Nliq)],'k--',label='Equilibrium')
#plt.xlabel(r'$\mu$ (K)')
#plt.ylabel(r'$N$')
#plt.legend()
#plt.show()
#
#rholiq = Nliq[np.argmin(sqdeltaWi)]/Vbox * Mw_hexane / N_A * gmtokg / Ang3tom3 #[kg/m3]
#rhovap = Nvap[np.argmin(sqdeltaWi)]/Vbox * Mw_hexane / N_A * gmtokg / Ang3tom3 #[kg/m3]
#           
#plt.plot(rhovap,Ti,'bo')
#plt.plot(rholiq,Ti,'ro')
#plt.plot(rhov_Potoff,Tsat_Potoff,'ks',mfc='None')
#plt.plot(rhol_Potoff,Tsat_Potoff,'ks',mfc='None')
#plt.xlabel(r'$\rho$ (kg/m$^3$)')
#plt.ylabel(r'$T$ (K)')
#plt.xlim([0,650])
#plt.ylim([320,520])
#plt.show()