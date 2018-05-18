# -*- coding: utf-8 -*-
"""
MBAR GCMC
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

root_path = 'hexane_Potoff/'
Temp_range = ['510','470','430','480','450','420','390','360','330']
hist_num=['1','2','3','4','5','6','7','8','9']

#Temp_range = ['510','480']
#hist_num=['1','4']

nSnapshots=90000

U_all = np.zeros([len(Temp_range),nSnapshots])
Nmol_all = np.zeros([len(Temp_range),nSnapshots])

for iT, Temp in enumerate(Temp_range):
    
    hist_name='/his'+hist_num[iT]+'a.dat' #Only if loading hexane_Potoff
    
    NU = np.loadtxt(root_path+Temp+hist_name,skiprows=1)
    
    U = NU[:,1]
    N = NU[:,0]
    
    U_all[iT] = U
    Nmol_all[iT] = N
         
Temp_sim = np.array([510,470,430,480,450,420,390,360,330])  
mu_sim = np.array([-4127., -4127., -4127., -3980., -3880., -3800., -3725., -3675.,-3625.])  

#Temp_sim = np.array([510,480])  
#mu_sim = np.array([-4127., -3980.]) 

Lbox = 35. #[Angstrom]
Vbox = Lbox**3 #[Angstrom^3]

## N_k contains the number of snapshots from each state point simulated
## Nmol_kn contains all of the Number of molecules in 1-d array
## u_kn_sim contains all the reduced potential energies just for the simulated points

N_k = [nSnapshots]*len(Temp_sim)
sumN_k = nSnapshots*len(Temp_sim)
Nmol_kn = Nmol_all.reshape([Nmol_all.size])
u_kn_sim = np.zeros([len(Temp_sim),nSnapshots*len(Temp_sim)])

for iT, (Temp, mu) in enumerate(zip(Temp_sim, mu_sim)):
    
    for jT in range(len(Temp_sim)):
        
        jstart = nSnapshots*jT
        jend = jstart+nSnapshots
        
        u_kn_sim[iT,jstart:jend] = U_to_u(U_all[jT],Temp,mu,Nmol_all[jT])
        
mbar = MBAR(u_kn_sim,N_k)

Deltaf_ij = mbar.getFreeEnergyDifferences(return_theta=False)[0]

f_k_sim = Deltaf_ij[0,:]
print(f_k_sim)
### This was how I want to implement it, but I am not sure how to make 
Temp_VLE = np.array([500,490,480,470,460,450,440,430,420,410,400,390,380,370,360,350,340,330,320])
#Temp_VLE = np.array([450,400,370,320])

Temp_all = np.concatenate((Temp_sim,Temp_VLE))
N_k.extend([0]*len(Temp_VLE))

u_kn_VLE = np.zeros([len(Temp_VLE),nSnapshots*len(Temp_sim)])
u_kn = np.concatenate((u_kn_sim,u_kn_VLE))

f_k_guess = np.concatenate((f_k_sim,np.zeros(len(Temp_VLE))))

U_flat = U_all.flatten()
Nmol_flat = Nmol_all.flatten()

jT0 = len(Temp_sim)
Ncut = 81

def sqdeltaW(mu_VLE):
    
    for jT, (Temp, mu) in enumerate(zip(Temp_VLE, mu_VLE)):
        
        u_kn[jT0+jT,:] = U_to_u(U_flat,Temp,mu,Nmol_flat)
#    print(u_kn.shape,np.array(N_k).shape,f_k_guess.shape)
    mbar = MBAR(u_kn,N_k,initial_f_k=f_k_guess)

    sumWliq = np.sum(mbar.W_nk[:,jT0:][Nmol_flat>Ncut],axis=0)
    sumWvap = np.sum(mbar.W_nk[:,jT0:][Nmol_flat<=Ncut],axis=0)
    sqdeltaW_VLE = (sumWliq-sumWvap)**2

### Could be advantageous to store this. But needs to be outside the function. Either as a global variable or within the optimizer    
### I guess within the class I can store this as self.f_k_guess and update it each time the function is called     
#    Deltaf_ij = mbar.getFreeEnergyDifferences(return_theta=False)[0]
#    f_k_guess = Deltaf_ij[0,:]
                 
    return sqdeltaW_VLE

### This approach is much slower, i.e. performing the MBAR analysis for each temperature individually
#N_k = [nSnapshots]*len(Temp_sim)
#N_k.extend([0])
#
#u_kn_VLE = np.zeros([1,nSnapshots*len(Temp_sim)])
#u_kn = np.concatenate((u_kn_sim,u_kn_VLE))
#
#f_k_guess = np.concatenate((f_k_sim,np.zeros(1)))
#                    
#def sqdeltaW(mu_VLE):
#    
#    sqdeltaW_VLE = np.zeros(len(Temp_VLE))
#    
#    for jT, (Temp, mu) in enumerate(zip(Temp_VLE, mu_VLE)):
#        
#        u_kn[-1,:] = U_to_u(U_all_flat,Temp,mu,N_all_flat)
#    
#        mbar = MBAR(u_kn,N_k,initial_f_k=f_k_guess)
#
#        sumWliq = np.sum(mbar.W_nk[:,-1][Nmol_kn>Ncut])
#        sumWvap = np.sum(mbar.W_nk[:,-1][Nmol_kn<=Ncut])
#        sqdeltaW_VLE[jT] = (sumWliq - sumWvap)**2
#                    
#    return sqdeltaW_VLE

### Optimization of mu
### Bounds for mu
mu_sim_low = np.ones(len(Temp_VLE))*mu_sim.min()
mu_sim_high = np.ones(len(Temp_VLE))*mu_sim.max()

Temp_sim_mu_low = Temp_sim[np.argmin(mu_sim)]
Temp_sim_mu_high = Temp_sim[np.argmax(mu_sim)]

### Guess for mu
mu_guess = lambda Temp: mu_sim_high + (mu_sim_low - mu_sim_high)/(Temp_sim_mu_low-Temp_sim_mu_high) * (Temp-Temp_sim_mu_high)

mu_VLE_guess = mu_guess(Temp_VLE)
mu_VLE_guess[mu_VLE_guess<mu_sim.min()] = mu_sim.min()
mu_VLE_guess[mu_VLE_guess>mu_sim.max()] = mu_sim.max()
print(mu_VLE_guess)
mu_lower_bound = mu_sim_low*1.005
mu_upper_bound = mu_sim_high*0.995

print(r'$(\Delta W)^2$ for $\mu_{\rm guess}$ =')
print(sqdeltaW(mu_VLE_guess))

### Optimize mu

mu_opt = GOLDEN_multi(sqdeltaW,mu_VLE_guess,mu_lower_bound,mu_upper_bound,TOL=0.0001,maxit=30)
sqdeltaW_opt = sqdeltaW(mu_opt)

plt.plot(Temp_VLE,mu_opt,'k-',label=r'$\mu_{\rm opt}$')
plt.plot(Temp_sim,mu_sim,'ro',mfc='None',label='Simulation')
plt.plot(Temp_VLE,mu_VLE_guess,'b--',label=r'$\mu_{\rm guess}$')
plt.xlabel(r'$T$ (K)')
plt.ylabel(r'$\mu_{\rm opt}$ (K)')
plt.xlim([300,550])
plt.ylim([-4200,-3600])
plt.legend()
plt.show()

plt.plot(Temp_VLE,sqdeltaW_opt,'ko')
plt.xlabel(r'$T$ (K)')
plt.ylabel(r'$(\Delta W^{\rm sat})^2$')
plt.show()

print("Effective sample numbers")
print (mbar.computeEffectiveSampleNumber())
print('\nWhich is approximately '+str(mbar.computeEffectiveSampleNumber()/sumN_k*100.)+'% of the total snapshots')

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
    
for jT, (Temp, mu) in enumerate(zip(Temp_VLE, mu_opt)):
    
    u_kn[jT0+jT,:] = U_to_u(U_flat,Temp,mu,Nmol_flat)
    
mbar = MBAR(u_kn,N_k,initial_f_k=f_k_guess)
    
sumWliq = np.sum(mbar.W_nk[:,jT0:][Nmol_kn>Ncut],axis=0)
sumWvap = np.sum(mbar.W_nk[:,jT0:][Nmol_kn<=Ncut],axis=0)

Nliq = np.sum(mbar.W_nk[:,jT0:][Nmol_kn>Ncut].T*Nmol_kn[Nmol_kn>Ncut],axis=1)/sumWliq #Must renormalize by the liquid or vapor phase
Nvap = np.sum(mbar.W_nk[:,jT0:][Nmol_kn<=Ncut].T*Nmol_kn[Nmol_kn<=Ncut],axis=1)/sumWvap

rholiq = Nliq/Vbox * Mw_hexane / N_A * gmtokg / Ang3tom3 #[kg/m3]
rhovap = Nvap/Vbox * Mw_hexane / N_A * gmtokg / Ang3tom3 #[kg/m3]
   
plt.plot(rhovap,Temp_VLE,'ro',label='MBAR-GCMC')
plt.plot(rholiq,Temp_VLE,'ro')
plt.plot(rhov_Potoff,Tsat_Potoff,'ks',mfc='None',label='Potoff')
plt.plot(rhol_Potoff,Tsat_Potoff,'ks',mfc='None')
plt.xlabel(r'$\rho$ (kg/m$^3$)')
plt.ylabel(r'$T$ (K)')
plt.xlim([0,650])
plt.ylim([320,520])
plt.legend()
plt.show()
 