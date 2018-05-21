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

T1 = 480. #[K]
mu1 = -3980 #[K]

NU0 = np.loadtxt('H:/MBAR_GCMC/hexane_Potoff/510/his1a.dat',skiprows=1)
NU1 = np.loadtxt('H:/MBAR_GCMC/hexane_Potoff/480/his4a.dat',skiprows=1)

U0 = NU0[:,1]
N0 = NU0[:,0]

U1 = NU1[:,1]
N1 = NU1[:,0]

u00=U_to_u(U0,T0,mu0,N0)
u01=U_to_u(U1,T0,mu0,N1)
u10=U_to_u(U0,T1,mu1,N0)
u11=U_to_u(U1,T1,mu1,N1)

T2 = 480. #[K]
mu2 = -4023 #[K]

u20 = U_to_u(U0,T2,mu2,N0)
u21 = U_to_u(U1,T2,mu2,N1)
      
N_k = np.array([len(u00),len(u11),0]) # The number of samples from each state
N_K = np.sum(N_k)
              
u_kn = np.zeros([3,len(u00)+len(u11)])
u_kn[0,:len(u00)] = u00
u_kn[0,len(u00):] = u01     
u_kn[1,:len(u00)] = u10
u_kn[1,len(u00):] = u11    
u_kn[2,:len(u00)] = u20
u_kn[2,len(u00):] = u21   
     
N_kn = np.zeros(len(u00)+len(u11))
N_kn[:len(u00)] = N0
N_kn[len(u00):] = N1     
              
mbar = MBAR(u_kn,N_k)

Deltaf_ij = mbar.getFreeEnergyDifferences(return_theta=False)
print "effective sample numbers"
print (mbar.computeEffectiveSampleNumber())
print('\nWhich is approximately '+str(mbar.computeEffectiveSampleNumber()/N_K*100.)+'%')

NAk, dNAk = mbar.computeExpectations(N_kn) # Average number of molecules
NAk_alt = np.zeros(len(N_k))
for i in range(len(N_k)):
    NAk_alt[i] = np.sum(mbar.W_nk[:,i]*N_kn)
    
print(NAk)

Nscan = np.arange(60,100)

sqdeltaW0 = np.zeros(len(Nscan))

for iN, Ni in enumerate(Nscan):

    sumWliq = np.sum(mbar.W_nk[:,0][N_kn>Ni])
    sumWvap = np.sum(mbar.W_nk[:,0][N_kn<=Ni])
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
nStates = len(N_k)-1
                                     
for imu, mui in enumerate(mu_scan):
  
    ui0 = U_to_u(U0,Ti,mui,N0) #Keeping epsilon and sigma constant for now
    ui1 = U_to_u(U1,Ti,mui,N1) #Keeping epsilon and sigma constant for now
                
    u_kn[nStates,:len(u00)] = ui0
    u_kn[nStates,len(u00):] = ui1
    
    mbar = MBAR(u_kn,N_k)
    
    (Deltaf_ij, dDeltaf_ij, Theta_ij) = mbar.getFreeEnergyDifferences(return_theta=True)
    
    sumWliq = np.sum(mbar.W_nk[:,nStates][N_kn>Ncut])
    sumWvap = np.sum(mbar.W_nk[:,nStates][N_kn<=Ncut])
    sqdeltaWi[imu] = (sumWliq - sumWvap)**2

#    Nliq[imu], dNliq = mbar.computeExpectations(N0[N0>Ncut])
#    Nvap[imu], dNvap = mbar.computeExpectations(N0[N0<Ncut])
    
    Nliq[imu] = np.sum(mbar.W_nk[:,nStates][N_kn>Ncut]*N_kn[N_kn>Ncut])/sumWliq #Must renormalize by the liquid or vapor phase
    Nvap[imu] = np.sum(mbar.W_nk[:,nStates][N_kn<=Ncut]*N_kn[N_kn<=Ncut])/sumWvap
          
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