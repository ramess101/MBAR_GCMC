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

T0 = 510. #[K]
mu0 = -4127. #[K]
Lbox = 35. #[Angstrom]
Vbox = Lbox**3 #[Angstrom^3]

NU0 = np.loadtxt('H:/MBAR_GCMC/hexane_Potoff/510/his1a.dat',skiprows=1)

U0 = NU0[:,1]
N0 = NU0[:,0]

u0=U_to_u(U0,T0,mu0,N0)

T1 = 500 #[K]
mu1 = -4080 #[K]

u1 = U_to_u(U0,T1,mu1,N0) #Keeping epsilon and sigma constant for now

T2 = 500 #[K]
mu2 = -4000 #[K]

u2 = U_to_u(U0,T2,mu2,N0) #Keeping epsilon and sigma constant for now   

T3 = 500 #[K]
mu3 = -4127 #[K]

u3 = U_to_u(U0,T3,mu3,N0) #Keeping epsilon and sigma constant for now    

T4 = 300 #[K]
mu4 = -3650 #[K]

u4 = U_to_u(U0,T4,mu4,N0) #Keeping epsilon and sigma constant for now  
                    
plt.hist(N0,bins=np.arange(np.min(N0),np.max(N0)))
plt.xlabel(r'$N$')
plt.ylabel('Count')
plt.show()           
           
plt.plot(N0,U0,'ko',markersize=0.05,alpha=0.5)
plt.xlabel(r'$N$')
plt.ylabel(r'$U$')           
plt.show()
      
# Using just the Model 0 mdrun samples
N_k = np.array([len(u0),0,0,0,0]) # The number of samples from the different states

u_kn = np.array([u0,u1,u2,u3,u4])
Nmol_kn = N0

mbar = MBAR(u_kn,N_k)

(Deltaf_ij, dDeltaf_ij, Theta_ij) = mbar.getFreeEnergyDifferences(return_theta=True)
print "effective sample numbers"
print (mbar.computeEffectiveSampleNumber()/N_k[0]*100.)
print('\nWhich is approximately '+str(mbar.computeEffectiveSampleNumber()/N_k[0]*100.)+'%')

NAk, dNAk = mbar.computeExpectations(N0) # Average number of molecules
NAk_alt = np.zeros(len(N_k))
for i in range(len(N_k)):
    NAk_alt[i] = np.sum(mbar.W_nk[:,i]*N0)
    
print(NAk)

Nscan = np.arange(60,100)

sqdeltaW0 = np.zeros(len(Nscan))

for iN, Ni in enumerate(Nscan):

    sumWliq = np.sum(mbar.W_nk[:,0][N0>Ni])
    sumWvap = np.sum(mbar.W_nk[:,0][N0<=Ni])
    sqdeltaW0[iN] = (sumWliq - sumWvap)**2

plt.plot(Nscan,sqdeltaW0,'ko')
plt.xlabel(r'$N_{\rm cut}$')
plt.ylabel(r'$(\Delta W_0^{\rm sat})^2$')
plt.show()
                        
Ncut = Nscan[np.argmin(sqdeltaW0)]

mu_scan = np.linspace(-4150,-3950,200)
sqdeltaWi = np.zeros(len(mu_scan))
Nliq = np.zeros(len(mu_scan))
Nvap = np.zeros(len(mu_scan))

Ti = 500 #[K]

for imu, mui in enumerate(mu_scan):
  
    ui = U_to_u(U0,Ti,mui,N0) #Keeping epsilon and sigma constant for now
                
    N_k = np.array([len(u0),0]) # The number of samples from the two states           
               
    u_kn = np.array([u0,ui])
    Nmol_kn = N0
    
    mbar = MBAR(u_kn,N_k)
    
    (Deltaf_ij, dDeltaf_ij, Theta_ij) = mbar.getFreeEnergyDifferences(return_theta=True)
    
    sumWliq = np.sum(mbar.W_nk[:,1][N0>Ncut])
    sumWvap = np.sum(mbar.W_nk[:,1][N0<=Ncut])
    sqdeltaWi[imu] = (sumWliq - sumWvap)**2

#    Nliq[imu], dNliq = mbar.computeExpectations(N0[N0>Ncut])
#    Nvap[imu], dNvap = mbar.computeExpectations(N0[N0<Ncut])
    
    Nliq[imu] = np.sum(mbar.W_nk[:,1][N0>Ncut]*N0[N0>Ncut])/sumWliq #Must renormalize by the liquid or vapor phase
    Nvap[imu] = np.sum(mbar.W_nk[:,1][N0<=Ncut]*N0[N0<=Ncut])/sumWvap
          
plt.plot(mu_scan,sqdeltaWi,'k-')
plt.xlabel(r'$\mu$ (K)')
plt.ylabel(r'$(\Delta W_1^{\rm sat})^2$')
plt.show()

plt.plot(mu_scan,Nliq,'r-',label='Liquid')
plt.plot(mu_scan,Nvap,'b--',label='Vapor')
plt.plot([mu_scan[np.argmin(sqdeltaWi)],mu_scan[np.argmin(sqdeltaWi)]],[0,np.max(Nliq)],'k--',label='Equilibrium')
plt.xlabel(r'$\mu$ (K)')
plt.ylabel(r'$N$')
plt.legend()
plt.show()

rholiq = Nliq[np.argmin(sqdeltaWi)]/Vbox * Mw_hexane / N_A * gmtokg / Ang3tom3 #[kg/m3]
rhovap = Nvap[np.argmin(sqdeltaWi)]/Vbox * Mw_hexane / N_A * gmtokg / Ang3tom3 #[kg/m3]
           
plt.plot(rhovap,Ti,'bo')
plt.plot(rholiq,Ti,'ro')
plt.plot(rhov_Potoff,Tsat_Potoff,'ks',mfc='None')
plt.plot(rhol_Potoff,Tsat_Potoff,'ks',mfc='None')
plt.xlabel(r'$\rho$ (kg/m$^3$)')
plt.ylabel(r'$T$ (K)')
plt.xlim([0,650])
plt.ylim([320,520])
plt.show()