# -*- coding: utf-8 -*-
"""
GCMC histogram reweighting
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

Ang3tom3 = 10**-30
gmtokg = 1e-3
kb = 1.3806485e-23 #[J/K]
Jm3tobar = 1e-5
Rg = 8.314472 #[J/mol/K]
N_A = Rg/kb #[1/mol]

Mw_hexane  = 12.0109*6.+1.0079*(2.*6.+2.) #[gm/mol]

Tsat_Potoff = np.array([500,490,480,470,460,450,440,430,420,410,400,390,380,370,360,350,340,330,320])
rhol_Potoff = np.array([366.018,395.855,422.477,444.562,463.473,480.498,496.217,510.897,524.727,537.821,550.308,562.197,573.494,584.216,594.369,604.257,614.026,623.44,631.598])
rhov_Potoff = np.array([112.352,90.541,72.249,58.283,47.563,39.028,32.053,26.27,21.441,17.397,14.013,11.19,8.846,6.913,5.331,4.05,3.023,2.213,1.584])     
Psat_Potoff = np.array([27.697,23.906,20.521,17.529,14.889,12.563,10.522,8.738,7.19,5.857,4.717,3.753,2.946,2.279,1.734,1.296,0.949,0.68,0.476])
Zsat_Potoff = Psat_Potoff / rhov_Potoff / Tsat_Potoff * Mw_hexane / Rg * gmtokg / Jm3tobar

#plt.plot(Tsat_Potoff,Zsat_Potoff,'b-',label='Potoff et al.')
#plt.plot(Tsat_Potoff,np.ones(len(Tsat_Potoff)),'k--',label='Ideal Gas')
#plt.xlabel('Temperature [K]')
#plt.ylabel(r'$Z_v^{sat}$')
#plt.legend()
#plt.show()

class histogram_reweighting():
    def __init__(self,filepaths,use_stored_C=False):
        self.filepaths = filepaths
        self.extract_all_data()
        self.min_max_all_data()
        self.calc_fi_NU()
        if use_stored_C:
            self.C_all = C_stored
        else:    
            self.solve_C()
        self.N_cut = 70
        self.Mw = Mw_hexane 
#        
        #self.solve_mu(480)
        #self.calc_P_NU_muT(-4127,500)
        
    def calc_rho(self,Temp_VLE_all):
        
        rho_vapor = np.zeros(len(Temp_VLE_all))
        rho_liquid = np.zeros(len(Temp_VLE_all))
        
        for iT, Temp_VLE in enumerate(Temp_VLE_all):
            
            mu_VLE = self.solve_mu(Temp_VLE,refined=True)
            
            P_NU_muT, N_muT = self.calc_P_NU_muT(mu_VLE,Temp_VLE)
            N_unique = self.N_unique
            N_cut = self.N_cut
            Vbox = self.Vbox_all[0]
            #print(P_NU_muT)
            #print(N_muT)
            Nave_vapor = np.sum(N_muT[N_unique<=N_cut]*N_unique[N_unique<=N_cut])/np.sum(N_muT[N_unique<=N_cut])
            Nave_liquid = np.sum(N_muT[N_unique>N_cut]*N_unique[N_unique>N_cut])/np.sum(N_muT[N_unique>N_cut])
            #print(Nave_vapor,Nave_liquid)
            rho_vapor[iT] = Nave_vapor / Vbox
            rho_liquid[iT] = Nave_liquid / Vbox
        
        rho_vapor *= self.Mw / N_A * gmtokg / Ang3tom3 #[kg/m3]
        rho_liquid *= self.Mw / N_A * gmtokg / Ang3tom3 #[kg/m3]
        
        return rho_vapor, rho_liquid
        
    def solve_mu(self,Temp,refined=False):
        
        mu_min = self.mu_all.min() - 20. #Start with the most negative (with a buffer)
        
        maxit = 1000
        converged = False
        
        mu_old = mu_min
        it = 0
        diff_area_peaks_old = self.diff_area_peaks(mu_old,Temp)
        
        while it < maxit or not converged:
            
            mu_new = mu_old + 10.
            
            diff_area_peaks_new = self.diff_area_peaks(mu_new,Temp)
            
            if diff_area_peaks_new * diff_area_peaks_old < 0: # If there is a sign change
            
                converged = True
                mu_opt = (mu_new + mu_old)/2.
                         
            else:
                
                mu_old = mu_new.copy()
                diff_area_peaks_old = diff_area_peaks_new.copy()
                
            it += 1  
            
        if refined:
                
            SSE = lambda mu: self.diff_area_peaks(mu,Temp)**2*1e15
            
            guess = mu_opt     
    
            bnds = ((mu_old,mu_new),)                        
                                            
            opt = minimize(SSE,guess,bounds=bnds) 

            mu_opt = opt.x[0]                              
        
        return mu_opt
            
        #return mu_opt
# This approach had problems because it would just converged to 
#        
#        SSE = lambda mu: self.diff_area_peaks(mu,Temp)**2*1e15
#        
#        guess = self.mu_all[0]     
#
#        bnds = ((None,self.mu_all.min()),)                        
#                                        
#        opt = minimize(SSE,guess,bounds=bnds)                               
#        #print(opt)
#        return opt.x[0]
    
    def diff_area_peaks(self,mu,Temp,verbose=False,plot=False):
        P_NU_muT, N_muT = self.calc_P_NU_muT(mu,Temp)
        N_unique = self.N_unique

        N_cut = self.N_cut
        
        area_vapor = np.sum(N_muT[N_unique<=N_cut])
        area_liquid = np.sum(N_muT[N_unique>N_cut])
        diff_area = area_liquid - area_vapor
        
        if verbose:
            print(area_vapor)
            print(area_liquid)
            print(diff_area)
            
        if plot:
            
            rhov_Temp_Potoff = rhov_Potoff[Tsat_Potoff==Temp]
            rhol_Temp_Potoff = rhol_Potoff[Tsat_Potoff==Temp]
        
            rhov_Temp_Potoff /= self.Mw / N_A * gmtokg / Ang3tom3 #[kg/m3]
            rhol_Temp_Potoff /= self.Mw / N_A * gmtokg / Ang3tom3 #[kg/m3]

            Nv_Temp_Potoff = rhov_Temp_Potoff * self.Vbox_all[0]
            Nl_Temp_Potoff = rhol_Temp_Potoff * self.Vbox_all[0]
            
#            for iT, Temp_i in enumerate(self.Temp_all):
#                    
#                if Temp_i == Temp:
#                        
#                    plt.hist(self.N_data_all[iT],bins=self.N_range,normed=True,label='Simulation')
                    
            plt.plot(N_unique,N_muT,'k-')
            plt.plot(N_unique[N_unique<=N_cut],N_muT[N_unique<=N_cut],'r--',label='Vapor')
            plt.plot(N_unique[N_unique>N_cut],N_muT[N_unique>N_cut],'b:',label='Liquid')
            plt.plot([Nv_Temp_Potoff,Nv_Temp_Potoff],[0,np.max(N_muT)],'k-',label='Potoff Vapor')
            plt.plot([Nl_Temp_Potoff,Nl_Temp_Potoff],[0,np.max(N_muT)],'k-',label='Potoff Liquid')
            plt.xlabel('Number of Molecules')
            plt.ylabel('PDF')
            plt.legend()
            plt.title(r'$\mu = $'+str(mu)+' K, T = '+str(Temp)+' K')
            plt.show()
        
        return diff_area
    
    def solve_C(self):
        
        C_old, mu_all, Temp_all = self.C_all.copy(), self.mu_all, self.Temp_all
#        C_new = np.zeros(len(C_old)) # This approach appears to converge slightly faster
        TOL = 1e-5
        
        maxit = 1000
#        print(self.C_all)
        for iteration in range(maxit):

            i = 0
            
            for mu_i, Temp_i in zip(mu_all, Temp_all):
            
                p_NU_i, N_NU_i = self.calc_P_NU_muT(mu_i,Temp_i)
                #print(p_NU_i.shape)
                sum_p_NU_i = np.sum(p_NU_i)
#                print(sum_p_NU_i)
#                print(np.sum(N_NU_i))
                self.C_all[i] = np.log(sum_p_NU_i)
#                print(self.C_all)
#                C_new[i] = np.log(sum_p_NU_i)                
                i += 1 
#            print(iteration)    
#            self.C_all = C_new.copy()
#            print(self.C_all)
            if np.mean(np.abs(self.C_all-C_old)) < TOL:
                print('Weights converged after '+str(iteration)+' iterations')
                break
            
            if iteration == maxit-1:
                print('Weights did not converge within '+str(iteration)+' iterations')
            
            C_old = self.C_all.copy()
        #self.C_all = np.array([1.825916099,-4.550827154])
            #print(self.C_all)
        
    def calc_P_NU_muT(self,mu,Temp):
        fi_NU, K_all, mu_all, Temp_all, C_all, U_flat, N_flat = self.fi_NU, self.K_all, self.mu_all, self.Temp_all, self.C_all, self.U_flat, self.N_flat
        sum_fi_flat = np.sum(fi_NU,axis=2).reshape(fi_NU.shape[0]*fi_NU.shape[1])
        
        beta = 1./Temp
        
        P_NU_muT = np.zeros(len(sum_fi_flat))
        
        for i_NU, sum_fi in enumerate(sum_fi_flat):
            
            if sum_fi > 0:
                               
                numerator = sum_fi * np.exp(-beta * U_flat[i_NU] + beta * mu * N_flat[i_NU])
                denominator = 0
                
                for K_i, mu_i, Temp_i, C_i in zip(K_all,mu_all,Temp_all,C_all):
                    beta_i = 1./Temp_i
                    denominator += K_i * np.exp(-beta_i * U_flat[i_NU] + beta_i * mu_i * N_flat[i_NU] - C_i)
                    
                P_NU_muT[i_NU] = numerator / denominator

        N_unique = self.N_unique
        N_muT = np.zeros(len(N_unique))
        
        for i_N, N_i in enumerate(N_unique):
            N_muT[i_N] = np.sum(P_NU_muT[N_flat==N_i])
                        
#        plt.scatter(N_flat,U_flat,c=P_NU_muT)
#        plt.xlabel('Number of Molecules')
#        plt.ylabel('Energy (K)')
#        plt.show()
#
#        plt.plot(N_unique,N_muT)
#        plt.xlabel('Number of Molecules')
#        plt.ylabel('Count')
#        plt.show()
        
#        print(P_NU_muT.shape)
        return P_NU_muT, N_muT
        
    def calc_fi_NU(self):
        N_min, N_max, U_min, U_max = self.N_min, self.N_max, self.U_min, self.U_max
        N_data_all, U_data_all, K_all = self.N_data_all, self.U_data_all, self.K_all
        
        N_range = np.linspace(N_min,N_max,N_max-N_min+1)
        U_range = np.linspace(U_min,U_max,200)
        
        NU_range = np.array([N_range,U_range])
        
        fi_NU = np.zeros([len(N_range)-1,len(U_range)-1,len(K_all)])
        N_flat = np.zeros((len(N_range)-1)*(len(U_range)-1))
        U_flat = N_flat.copy()
        #print(len(N_range))
        #print(len(U_range))
        #print(len(K_all))
        #print(fi_NU.shape)
        for i_R, K_i in enumerate(K_all):
            N_data_i = N_data_all[i_R]
            U_data_i = U_data_all[i_R]
            #print(N_data_i.shape)
            #print(U_data_i.shape)
            #print(NU_range.shape)
            hist_NU = np.histogram2d(N_data_i,U_data_i,NU_range)
            #print(hist_NU[0].shape)
            #print(hist_NU[1].shape)
            #print(hist_NU[2].shape)
            fi_NU[:,:,i_R] = hist_NU[0]
            self.K_all[i_R] = np.sum(hist_NU[0])
        i_NU = 0    
            
        for i_N, N_i in enumerate(hist_NU[1][:-1]):
            for i_U, U_i in enumerate(hist_NU[2][:-1]):
                N_flat[i_NU] = N_i
                U_flat[i_NU] = U_i
                i_NU += 1
        #print(N_flat.shape)
        #print(U_flat.shape)
        #print(fi_NU[:,:,0].shape)
        
#        plt.scatter(N_flat,U_flat,c=fi_NU[:,:,0].reshape(len(N_flat)))
#        plt.xlabel('Number of Molecules')
#        plt.ylabel('Energy (K)')
#        plt.show()
        
        N_unique = np.unique(N_flat)

        self.C_all = np.zeros(len(K_all))
        self.N_range = N_range
        self.fi_NU, self.N_flat, self.U_flat, self.N_unique = fi_NU, N_flat, U_flat, N_unique
        
    def min_max_all_data(self):
        
        N_data_all, U_data_all, K_all = self.N_data_all, self.U_data_all, self.K_all
        
        N_min = []
        N_max = []
        
        U_min = []
        U_max = []
        
        for i_R, K_i in enumerate(K_all):
            
            N_min.append(np.min(N_data_all[i_R]))
            N_max.append(np.max(N_data_all[i_R]))
            
            U_min.append(np.min(U_data_all[i_R]))
            U_max.append(np.max(U_data_all[i_R]))
        
        N_min = np.min(N_min)
        N_max = np.max(N_max)
        
        U_min = np.min(U_min)
        U_max = np.max(U_max)
        
        self.N_min, self.N_max, self.U_min, self.U_max = N_min, N_max, U_min, U_max
                
    def extract_all_data(self):
        filepaths = self.filepaths
        
        N_data_all = []
        U_data_all = []
        K_all = []
        
        N_min_all = []
        N_max_all = []
        
        U_min_all = []
        U_max_all = []
        
        Temp_all = np.zeros(len(filepaths))
        mu_all = np.zeros(len(filepaths))
        Vbox_all = np.zeros(len(filepaths))
        
        for ipath, filepath in enumerate(filepaths):
            N_data, U_data, Temp, mu, Vbox = self.extract_data(filepath)
            
            Temp_all[ipath] = Temp
            mu_all[ipath] = mu
            Vbox_all[ipath] = Vbox
                    
            N_data_all.append(N_data)
            U_data_all.append(U_data)
            K_all.append(len(N_data))
                    
            N_min_all.append(np.min(N_data))
            N_max_all.append(np.max(N_data))
             
            U_min_all.append(np.min(U_data))
            U_max_all.append(np.max(U_data))

            
        self.Temp_all, self.mu_all, self.Vbox_all, self.N_data_all, self.U_data_all, self.K_all = Temp_all, mu_all, Vbox_all, N_data_all, U_data_all, K_all
            
    def extract_data(self,filepath):
        NU_data = np.loadtxt(filepath,skiprows=1)
        N_data = NU_data[:,0]
        U_data = NU_data[:,1] #[K]
        mu_V_T = np.genfromtxt(filepath,skip_footer=len(NU_data))
        Temp = mu_V_T[0] #[K]
        mu = mu_V_T[2] #[K]
        Lbox = mu_V_T[3] #[Angstrom]
        Vbox = Lbox**3 #[Angstrom^3]
        return N_data, U_data, Temp, mu, Vbox
    
    def plot_histograms(self):
        N_data_all, N_range = self.N_data_all, self.N_range
        mu_all, Temp_all = self.mu_all, self.Temp_all
        
        plt.figure(figsize=(8,8))
        
        for mu, Temp, N_data in zip(mu_all, Temp_all,N_data_all):
            
            plt.hist(N_data,bins=N_range,alpha=0.5,normed=True,label=r'$\mu = $'+str(int(mu))+' K, T = '+str(int(Temp))+' K')
        
        plt.xlabel('Number of Molecules')
        plt.ylabel('Probability Density Function')
        plt.legend()
        plt.show()
        
    def plot_VLE(self,Temp_VLE_all):
        
        rhov, rhol = self.calc_rho(Temp_VLE_all)
        Psat = self.calc_Psat(Temp_VLE_all)
        
        plt.plot(rhov,Temp_VLE_all,'bo',mfc='None',markersize=8)
        plt.plot(rhol,Temp_VLE_all,'bo',mfc='None',markersize=8,label='This Work')
        plt.plot(rhol_Potoff,Tsat_Potoff,'ks',mfc='None',markersize=8)
        plt.plot(rhov_Potoff,Tsat_Potoff,'ks',mfc='None',markersize=8,label='Potoff et al.')
        plt.xlabel('Density (kg/m$^3$)')
        plt.ylabel('Temperature (K)')
        plt.legend()
        plt.show()
        
        plt.plot(1000./Temp_VLE_all,np.log10(Psat),'bo',mfc='None',markersize=8,label='This Work')
        plt.plot(1000./Tsat_Potoff,np.log10(Psat_Potoff),'ks',mfc='None',markersize=8,label='Potoff et al.')
        plt.xlabel('1000/T (K)')
        plt.ylabel(r'$\log_{10}(P_v^{sat}/bar)$')
        plt.legend()
        plt.show()
#        
    def calc_Psat(self,Temp_VLE_all):
        
        N_unique, N_cut = self.N_unique, self.N_cut
        
        Psat = np.zeros(len(Temp_VLE_all))
        
#        C_ideal = self.C_all[-1]
#        
#        N_data_ideal = self.N_data_all[-1]
#        Temp_ideal = self.Temp_all[-1]       
#        N_ideal = np.mean(N_data_ideal)
        Vbox = self.Vbox_all[0]
                
        Temp_ideal = 1000.
        beta_ideal = 1./Temp_ideal
#        mu_ideal = self.solve_mu(Temp_ideal,refined=True)
        mu_ideal = -20000.
        P_NU_muT, N_muT = self.calc_P_NU_muT(mu_ideal,Temp_ideal)
        C_ideal = np.log(np.sum(P_NU_muT))
        N_ideal = np.sum(N_muT[N_unique<=N_cut]*N_unique[N_unique<=N_cut])/np.sum(N_muT[N_unique<=N_cut])
        print(N_muT[:10])
        plt.plot(N_unique,N_muT)
        plt.plot([N_ideal,N_ideal],[0,np.max(N_muT)])
        plt.xlim([0,10])
        plt.xlabel('Number of Molecules')
        plt.ylabel('PDF')
        plt.show()
     
        rho_ideal = N_ideal / Vbox * self.Mw / N_A * gmtokg / Ang3tom3 #[kg/m3]
        print(C_ideal,N_ideal,rho_ideal)
        print(self.C_all)
        P_ideal = N_ideal / beta_ideal / Vbox * kb / Ang3tom3 * Jm3tobar
        print('Ideal gas pressure at reference is '+str(P_ideal)+' bar')
        
        for iT, Temp_VLE in enumerate(Temp_VLE_all):
            
            beta_VLE = 1./Temp_VLE
            
            mu_VLE = self.solve_mu(Temp_VLE,refined=True)
            
            P_NU_muT, N_muT = self.calc_P_NU_muT(mu_VLE,Temp_VLE)
            
            C_VLE = np.log(np.sum(P_NU_muT))
            
            print(C_VLE)
            
            deltaC = C_VLE - C_ideal
            
            print(deltaC)
            
            Psat[iT] = (deltaC + N_ideal) / beta_VLE / Vbox
                
        Psat *= kb / Ang3tom3 * Jm3tobar
        
        return Psat

filepaths = []
        
#root_path = 'H:/GOMC_practice/GCMC/hexane_replicates/'
#Temp_range = ['510','480','450','420','390','360','330','460','410']
#hist_num=['2']*len(Temp_range)

root_path = 'H:/MBAR_GCMC/hexane_Potoff/'
Temp_range = ['510','470','430','480','450','420','390','360','330']
hist_num=['1','2','3','4','5','6','7','8','9']

for iT, Temp in enumerate(Temp_range):
    
    hist_name='/his'+hist_num[iT]+'a.dat' #Only if loading hexane_Potoff
    
    filepaths.append(root_path+Temp+hist_name)

hist_trial = histogram_reweighting(filepaths,use_stored_C=False)
print(hist_trial.K_all)
#print(hist_trial.N_min)
#print(hist_trial.fi_NU)
hist_trial.plot_histograms()
C_stored = hist_trial.C_all
#Temp_VLE_all = np.array([420,390,360])
#Temp_VLE_all = np.array([460,410])
Temp_VLE_all = np.array([480,450,420,390,360,330])
#Temp_VLE_all = np.array([480.,330.])
#Temp_VLE_all = Tsat_Potoff
#hist_trial.calc_rho(Temp_VLE_all)
#hist_trial.diff_area_peaks(-4023.,480.,plot=True)
hist_trial.plot_VLE(Temp_VLE_all)
#Temp_VLE_all = np.array([320])
#for Temp_VLE in Temp_VLE_all:
#    
#    mu_VLE = hist_trial.solve_mu(Temp_VLE,refined=True)
#    print(mu_VLE)
#    hist_trial.diff_area_peaks(mu_VLE,Temp_VLE,plot=True)
#mu_range = np.linspace(-4100,-3900,200)
#diff_range = np.zeros(len(mu_range))
#
#for i_mu, mu_i in enumerate(mu_range):
#    
#    diff_range[i_mu] = hist_trial.diff_area_peaks(mu_i,Temp_VLE)
#    
#plt.plot(mu_range,diff_range)
#plt.xlabel('mu (K)')
#plt.ylabel('Difference areas')
#plt.title('Temperature = '+str(Temp_VLE)+' K')
#plt.ylim([diff_range.min(),-diff_range.min()])
#plt.show()