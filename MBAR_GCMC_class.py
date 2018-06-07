"""
MBAR GCMC
"""

import numpy as np
import matplotlib.pyplot as plt
from pymbar import MBAR
from Golden_search_multi import GOLDEN_multi
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

class MBAR_GCMC():
    def __init__(self,root_path,filepaths,Mw,trim_data=False,compare_literature=False):
        self.root_path = root_path
        self.filepaths = filepaths
        self.trim_data = trim_data            
        self.extract_all_sim_data()
        self.min_max_sim_data()
        self.build_MBAR_sim()
        self.Ncut = self.solve_Ncut()
        self.Mw = Mw
        self.compare_literature = compare_literature
                
    def min_max_sim_data(self):
        '''
        For the purpose of plotting histograms, this finds the minimum and
        maximum values of number of molecules, N, and internal energy, U
        '''
        N_data_sim, U_data_sim, K_sim = self.N_data_sim, self.U_data_sim, self.K_sim
        
        N_min = []
        N_max = []
        
        U_min = []
        U_max = []
        
        for i_R, K_i in enumerate(K_sim):
            
            N_min.append(np.min(N_data_sim[i_R]))
            N_max.append(np.max(N_data_sim[i_R]))
            
            U_min.append(np.min(U_data_sim[i_R]))
            U_max.append(np.max(U_data_sim[i_R]))
        
        N_min = np.min(N_min)
        N_max = np.max(N_max)
        
        U_min = np.min(U_min)
        U_max = np.max(U_max)
        
        N_range = np.linspace(N_min,N_max,N_max-N_min+1)
        U_range = np.linspace(U_min,U_max,200)
        
        self.N_min, self.N_max, self.U_min, self.U_max, self.N_range,self.U_range = N_min, N_max, U_min, U_max, N_range, U_range
                
    def extract_all_sim_data(self):
        '''
        Parses the number of molecules, N, internal energy, U, snapshots, K, for the
        simulated temperatures, Temp, volume, Vbox, and chemical potentials, mu
        '''
        filepaths = self.filepaths
        
        N_data_sim = []
        U_data_sim = []
        K_sim = []
        
        Nmol_flat = np.array([])
        U_flat = np.array([])
        
        N_min_sim = []
        N_max_sim = []
        
        U_min_sim = []
        U_max_sim = []
        
        Temp_sim = np.zeros(len(filepaths))
        mu_sim = np.zeros(len(filepaths))
        Vbox_sim = np.zeros(len(filepaths))
        
        for ipath, filepath in enumerate(filepaths):
            N_data, U_data, Temp, mu, Vbox = self.extract_data(filepath)
            
            Temp_sim[ipath] = Temp
            mu_sim[ipath] = mu
            Vbox_sim[ipath] = Vbox
                    
            N_data_sim.append(N_data)
            U_data_sim.append(U_data)
            K_sim.append(len(N_data))
            
            Nmol_flat = np.append(Nmol_flat,N_data)
            U_flat = np.append(U_flat,U_data)
            
#            print(N_data_sim)        
            N_min_sim.append(np.min(N_data))
            N_max_sim.append(np.max(N_data))
             
            U_min_sim.append(np.min(U_data))
            U_max_sim.append(np.max(U_data))
   
        self.Temp_sim, self.mu_sim, self.Vbox_sim, self.N_data_sim, self.U_data_sim, self.K_sim, self.Nmol_flat, self.U_flat = Temp_sim, mu_sim, Vbox_sim, N_data_sim, U_data_sim, K_sim, Nmol_flat, U_flat
            
    def extract_data(self,filepath):
        '''
        For a single filepath, returns the number of molecules, N, internal
        energy, U, temperature, Temp, chemical potential, mu, and volumbe, Vbox
        '''
        NU_data = np.loadtxt(filepath,skiprows=1)
        if self.trim_data:
            subset_size = 5000
        else:
            subset_size = len(NU_data)
        subset_data = np.random.choice(np.arange(0,len(NU_data)),size=subset_size,replace=False)
        N_data = NU_data[subset_data,0]
        U_data = NU_data[subset_data,1] #[K]
        mu_V_T = np.genfromtxt(filepath,skip_footer=len(NU_data))
        Temp = mu_V_T[0] #[K]
        mu = mu_V_T[2] #[K]
        Lbox = mu_V_T[3] #[Angstrom]
        Vbox = Lbox**3 #[Angstrom^3]
        return N_data, U_data, Temp, mu, Vbox
    
    def plot_histograms(self):
        '''
        Plots the histograms for the number of molecules
        '''
        N_data_sim, N_range = self.N_data_sim, self.N_range
        mu_sim, Temp_sim = self.mu_sim, self.Temp_sim
        
        plt.figure(figsize=(8,8))
        
        for mu, Temp, N_data in zip(mu_sim, Temp_sim,N_data_sim):
            
            plt.hist(N_data,bins=N_range,alpha=0.5,normed=True,label=r'$\mu = $'+str(int(mu))+' K, T = '+str(int(Temp))+' K')

        plt.plot([self.Ncut,self.Ncut],[0,0.25*plt.gca().get_ylim()[1]],'k-',label=r'$N_{\rm cut} = $'+str(self.Ncut))
        
        plt.xlabel('Number of Molecules')
        plt.ylabel('Probability Density Function')
        plt.legend()
        plt.show()
        
    def plot_2dhistograms(self):
        '''
        Plots the two-dimensional histograms for the number of molecules and internal energy
        '''
        N_data_sim, N_range, U_data_sim, U_range = self.N_data_sim, self.N_range, self.U_data_sim, self.U_range
        mu_sim, Temp_sim = self.mu_sim, self.Temp_sim
        
#        plt.figure(figsize=(8,8))
        
#        for mu, Temp, N_data, U_data in zip(mu_sim, Temp_sim,N_data_sim,U_data_sim):
#            plt.hist2d(N_data,U_data,bins=[N_range,U_range],normed=True,label=r'$\mu = $'+str(int(mu))+' K, T = '+str(int(Temp))+' K')
#
#        #plt.plot([self.Ncut,self.Ncut],[plt.gca().get_ylim()[0],plt.gca().get_ylim()[1]],'k-',label=r'$N_{\rm cut} = $'+str(self.Ncut))
#        plt.colorbar()
#        plt.xlabel('Number of Molecules')
#        plt.ylabel('Internal Energy (K)')
#        plt.legend()
#        plt.show()
        
        color_list = ['r','g','b','y','m','c','brown','orange','pink','grey']

        plt.figure(figsize=(8,8))

        for isim, (mu, Temp, N_data, U_data) in enumerate(zip(mu_sim, Temp_sim,N_data_sim,U_data_sim)):
            plt.plot(N_data,U_data,'o',color=color_list[isim],markersize=0.5,alpha=0.05)
            plt.plot([],[],'o',color=color_list[isim],label=r'$\mu = $'+str(int(mu))+' K, T = '+str(int(Temp))+' K')

        plt.plot([self.Ncut,self.Ncut],[plt.gca().get_ylim()[0],plt.gca().get_ylim()[1]],'k-',label=r'$N_{\rm cut} = $'+str(self.Ncut))
        
        plt.xlabel('Number of Molecules')
        plt.ylabel('Internal Energy (K)')
        plt.legend()
        plt.show()
        
    def U_to_u(self,Uint,Temp,mu,Nmol):
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
        
    def build_MBAR_sim(self):
        '''
        Creates an instance of the MBAR object for just the simulated state points
        N_k: contains the number of snapshots from each state point simulated
        Nmol_kn: contains all of the Number of molecules in 1-d array
        u_kn_sim: contains all the reduced potential energies just for the simulated points
        f_k_sim: the converged reduced free energies for each simulated state point (used as initial guess for non-simulated state points)
        '''         
        
        Temp_sim, mu_sim, nSnapshots, Nmol_flat, U_flat = self.Temp_sim, self.mu_sim, self.K_sim, self.Nmol_flat, self.U_flat
        
        N_k_sim = np.array(nSnapshots)
        sumN_k = np.sum(N_k_sim)
#        Nmol_flat = np.array(N_data_sim).flatten()
#        U_flat = np.array(U_data_sim).flatten()
        u_kn_sim = np.zeros([len(Temp_sim),sumN_k])
        
        for iT, (Temp, mu) in enumerate(zip(Temp_sim, mu_sim)):
    
            u_kn_sim[iT] = self.U_to_u(U_flat,Temp,mu,Nmol_flat)
        
        mbar_sim = MBAR(u_kn_sim,N_k_sim)
        
        Deltaf_ij = mbar_sim.getFreeEnergyDifferences(return_theta=False)[0]
        f_k_sim = Deltaf_ij[:,0]        
#        print(f_k_sim)
        
        self.u_kn_sim, self.f_k_sim, self.sumN_k, self.N_k_sim, self.mbar_sim = u_kn_sim, f_k_sim, sumN_k, N_k_sim, mbar_sim
    
    def solve_Ncut(self,method=2,show_plot=False):
        '''
        The MBAR_GCMC class uses a cutoff in the number of molecules, Ncut, to 
        distinguish between liquid and vapor phases. The value of Ncut is determined
        by equating the pressures at the bridge, i.e. the sum of the weights for
        the highest temperature simulated should be equal in the vapor and
        liquid phases.
        '''
        mbar_sim, Nmol_flat = self.mbar_sim,self.Nmol_flat
        
        bridge_index = np.argmax(self.Temp_sim)
        
        if method == 1:
        
            Nscan = np.arange(60,100)
            
            sqdeltaW_bridge = np.zeros(len(Nscan))
            
            for iN, Ni in enumerate(Nscan):
            
                sumWliq_bridge = np.sum(mbar_sim.W_nk[:,bridge_index][Nmol_flat>Ni])
                sumWvap_bridge = np.sum(mbar_sim.W_nk[:,bridge_index][Nmol_flat<=Ni])
                sqdeltaW_bridge[iN] = (sumWliq_bridge - sumWvap_bridge)**2
            
            if show_plot:
            
                plt.plot(Nscan,sqdeltaW_bridge,'k-')
                plt.xlabel(r'$N_{\rm cut}$')
                plt.ylabel(r'$(\Delta W_{\rm bridge}^{\rm sat})^2$')
                plt.show()
                                
            Ncut = Nscan[np.argmin(sqdeltaW_bridge)]
        
        elif method == 2:
        
            ### Alternative method that finds the minimum between the two peaks
            ### Note that this might not be best since noisy data could move Ncut quite a bit
            Nmol_bridge = self.N_data_sim[bridge_index]
            
            Nmol_mid = (Nmol_bridge.min()+Nmol_bridge.max())/2.
            
            Nmol_low = Nmol_bridge[Nmol_bridge<=Nmol_mid]
            Nmol_high = Nmol_bridge[Nmol_bridge>=Nmol_mid]
            
            Nmol_low_count,Nmol_low_bins = np.histogram(Nmol_low,bins=int(Nmol_low.max()-Nmol_low.min()))
            Nmol_high_count,Nmol_high_bins = np.histogram(Nmol_high,bins=int(Nmol_high.max()-Nmol_high.min()))
            
            Nmol_low_peak = Nmol_low_bins[:-2][np.argmax(Nmol_low_count[:-1])] #Need to remove final bin
            Nmol_high_peak = Nmol_high_bins[:-2][np.argmax(Nmol_high_count[:-1])]
            
            Nmol_valley = Nmol_bridge[Nmol_bridge>=Nmol_low_peak]
            Nmol_valley = Nmol_valley[Nmol_valley<=Nmol_high_peak]
            
            if show_plot:
            
                plt.hist(Nmol_bridge,bins=int(Nmol_bridge.max()-Nmol_bridge.min()+1),color='w')
                plt.hist(Nmol_low,bins=int(Nmol_low.max()-Nmol_low.min()+1),color='b',alpha=0.3)
                plt.hist(Nmol_high,bins=int(Nmol_high.max()-Nmol_high.min()+1),color='r',alpha=0.3)
                plt.hist(Nmol_valley,bins=int(Nmol_valley.max()-Nmol_valley.min()+1),color='k',alpha=0.3)
                plt.xlabel('N')
                plt.ylabel('Count')
                plt.show()
            
            Nmol_valley_count, Nmol_valley_bins = np.histogram(Nmol_valley,bins=int(Nmol_valley.max()-Nmol_valley.min()))
            Ncut = int(Nmol_valley_bins[:-2][np.argmin(Nmol_valley_count[:-1])])
        
        print('Liquid and vapor phase is divided by Nmol = '+str(Ncut))

        return Ncut
    
    def build_MBAR_VLE_matrices(self):
        '''
        Build u_kn, N_k, and f_k_guess by appending the u_kn_sim, N_k_sim, and
        f_k_sim with empty matrices of the appropriate dimensions, determined by
        the number of VLE points desired.
        '''
        Temp_VLE, Temp_sim, u_kn_sim,f_k_sim,sumN_k = self.Temp_VLE,self.Temp_sim, self.u_kn_sim,self.f_k_sim,self.sumN_k
        
        # nTsim is used to keep track of the number of simulation temperatures
        nTsim = len(Temp_sim)
        
#        Temp_VLE_all = np.concatenate((Temp_sim,Temp_VLE))
        N_k_all = self.K_sim[:]
        N_k_all.extend([0]*len(Temp_VLE))

        u_kn_VLE = np.zeros([len(Temp_VLE),sumN_k])
        u_kn_all = np.concatenate((u_kn_sim,u_kn_VLE))
        
        f_k_guess = np.concatenate((f_k_sim,np.zeros(len(Temp_VLE))))
        
        self.u_kn_all, self.f_k_guess, self.N_k_all, self.nTsim = u_kn_all, f_k_guess, N_k_all,nTsim
        
    def mu_guess_bounds(self):
        '''
        Start with reasonable guess for mu and provide bounds for optimizer
        '''
        
        mu_sim, Temp_sim, Temp_VLE = self.mu_sim, self.Temp_sim, self.Temp_VLE
        
        ### Bounds for mu
        mu_sim_low = np.ones(len(Temp_VLE))*mu_sim.min()
        mu_sim_high = np.ones(len(Temp_VLE))*mu_sim.max()
        
        Temp_sim_mu_low = Temp_sim[np.argmin(mu_sim)]
        Temp_sim_mu_high = Temp_sim[np.argmax(mu_sim)]

        ### Guess for mu determined by the mu at the lowest and highest temperatures
        mu_guess = lambda Temp: mu_sim_high + (mu_sim_low - mu_sim_high)/(Temp_sim_mu_low-Temp_sim_mu_high) * (Temp-Temp_sim_mu_high)
        
        mu_VLE_guess = mu_guess(Temp_VLE)
        mu_VLE_guess[mu_VLE_guess<mu_sim.min()] = mu_sim.min()
        mu_VLE_guess[mu_VLE_guess>mu_sim.max()] = mu_sim.max()
        
        ### Buffer for the lower and upper bounds. Necessary for Golden search.
        mu_lower_bound = mu_sim_low*1.02
        mu_upper_bound = mu_sim_high*0.995
        
        return mu_VLE_guess, mu_lower_bound, mu_upper_bound
        
    def solve_VLE(self,Temp_VLE,show_plot=True):
        '''
        Determine optimal values of mu that result in equal pressures by 
        minimizing the square difference of the weights in the liquid and vapor
        phases. Subsequentally, calls the function to compute the saturation
        properties.
        '''
        
        self.Temp_VLE = Temp_VLE

        self.build_MBAR_VLE_matrices()
        mu_VLE_guess, mu_lower_bound, mu_upper_bound = self.mu_guess_bounds()
        
        ### Optimization of mu
        mu_VLE_guess, mu_lower_bound, mu_upper_bound = self.mu_scan(Temp_VLE)
        mu_opt = GOLDEN_multi(self.sqdeltaW,mu_VLE_guess,mu_lower_bound,mu_upper_bound,TOL=0.0001,maxit=30)
        sqdeltaW_opt = self.sqdeltaW(mu_opt)
        
        self.f_k_opt = self.f_k_guess.copy()
        self.mu_opt = mu_opt
        
        self.calc_rhosat()
        self.calc_Psat()
        
        self.print_VLE()
        
        if show_plot:
        
            plt.plot(Temp_VLE,mu_opt,'k-',label=r'$\mu_{\rm opt}$')
            plt.plot(self.Temp_sim,self.mu_sim,'ro',mfc='None',label='Simulation')
            plt.plot(Temp_VLE,mu_VLE_guess,'b--',label=r'$\mu_{\rm guess}$')
            plt.xlabel(r'$T$ (K)')
            plt.ylabel(r'$\mu_{\rm opt}$ (K)')
#            plt.xlim([300,550])
#            plt.ylim([-4200,-3600])
            plt.legend()
            plt.show()
            
            plt.plot(Temp_VLE,sqdeltaW_opt,'ko')
            plt.xlabel(r'$T$ (K)')
            plt.ylabel(r'$(\Delta W^{\rm sat})^2$')
            plt.show()
            
            plt.plot(Temp_VLE,self.f_k_opt[self.nTsim:])
            plt.plot([self.Temp_IG]*len(self.f_k_IG),self.f_k_IG,'ko')
            plt.show()
            
            print("Effective number of samples")
            print (self.mbar.computeEffectiveSampleNumber())
            print('\nWhich is approximately '+str(self.mbar.computeEffectiveSampleNumber()/self.sumN_k*100.)+'% of the total snapshots')
     
    def sqdeltaW(self,mu_VLE):
        '''
        Computes the square difference between the sum of the weights in the
        vapor and liquid phases.
        Stores the optimal reduced free energy as f_k_guess for future iterations
        Stores mbar, sumWliq, and sumWvap for computing VLE properties if converged
        '''
        
        nTsim, U_flat, Nmol_flat,Ncut, f_k_guess, Temp_VLE, u_kn_all, N_k_all = self.nTsim, self.U_flat, self.Nmol_flat, self.Ncut,self.f_k_guess, self.Temp_VLE, self.u_kn_all, self.N_k_all

        for jT, (Temp, mu) in enumerate(zip(Temp_VLE, mu_VLE)):
            
            u_kn_all[nTsim+jT,:] = self.U_to_u(U_flat,Temp,mu,Nmol_flat)

        mbar = MBAR(u_kn_all,N_k_all,initial_f_k=f_k_guess)
    
        sumWliq = np.sum(mbar.W_nk[:,nTsim:][Nmol_flat>Ncut],axis=0)
        sumWvap = np.sum(mbar.W_nk[:,nTsim:][Nmol_flat<=Ncut],axis=0)
        sqdeltaW_VLE = (sumWliq-sumWvap)**2
        ### It could be better to use just the absolute difference between the two
        
        ### Store previous solutions to speed-up future convergence of MBAR
        Deltaf_ij = mbar.getFreeEnergyDifferences(return_theta=False)[0]
        self.f_k_guess = Deltaf_ij[:,0]
        self.mbar, self.sumWliq, self.sumWvap  = mbar, sumWliq, sumWvap
              
        return sqdeltaW_VLE
    
    def calc_rhosat(self):
        '''
        Computes the saturated liquid and vapor densities
        '''
        Nmol_flat, Ncut, mbar, sumWliq, sumWvap, Mw, Vbox, nTsim = self.Nmol_flat, self.Ncut, self.mbar, self.sumWliq, self.sumWvap, self.Mw, self.Vbox_sim[0], self.nTsim
        
        Nliq = np.sum(mbar.W_nk[:,nTsim:][Nmol_flat>Ncut].T*Nmol_flat[Nmol_flat>Ncut],axis=1)/sumWliq #Must renormalize by the liquid or vapor phase
        Nvap = np.sum(mbar.W_nk[:,nTsim:][Nmol_flat<=Ncut].T*Nmol_flat[Nmol_flat<=Ncut],axis=1)/sumWvap
        
        rholiq = Nliq/Vbox * Mw / N_A * gmtokg / Ang3tom3 #[kg/m3]
        rhovap = Nvap/Vbox * Mw / N_A * gmtokg / Ang3tom3 #[kg/m3]
        
        self.rholiq, self.rhovap = rholiq, rhovap
    
    def plot_VLE(self,compare_RP=False,Tsat_RP=[],rhol_RP=[],rhov_RP=[],Psat_RP=[],Tsat_lit=[],rhol_lit=[],rhov_lit=[],Psat_lit=[]):
        '''
        Plots the saturation densities and compares with literature values if available
        This could be compressed significantly with some for loops.
        '''
        Temp_VLE, rholiq, rhovap, Psat = self.Temp_VLE, self.rholiq, self.rhovap, self.Psat
        
        Psat_ig = rhovap * Rg * Temp_VLE / self.Mw / gmtokg * Jm3tobar
        
        if compare_RP:
        
            plt.plot(rhov_RP,Tsat_RP,'k-',label='REFPROP')
            plt.plot(rhol_RP,Tsat_RP,'k-')
            
        plt.plot(rhovap,Temp_VLE,'ro',label='MBAR-GCMC')
        plt.plot(rholiq,Temp_VLE,'ro')
        
        if self.compare_literature:
                    
            plt.plot(rhov_lit,Tsat_lit,'ks',mfc='None',label='Potoff')
            plt.plot(rhol_lit,Tsat_lit,'ks',mfc='None')
        
        plt.xlabel(r'$\rho$ (kg/m$^3$)')
        plt.ylabel(r'$T$ (K)')
        plt.xlim([-10,1.04*rholiq.max()])
        plt.ylim([0.98*Temp_VLE.min(),1.02*Temp_VLE.max()])
        plt.legend()
        plt.show()
                
        plt.figure(figsize=[6,6])
        
        if compare_RP:
            plt.plot(1./Tsat_RP,np.log10(Psat_RP),'k-',label='REFPROP') 
        
        plt.plot(1./Temp_VLE,np.log10(Psat),'ro',label='MBAR-GCMC')
        plt.plot(1./Temp_VLE,np.log10(Psat_ig),'r--',label='Ideal gas')
        
        if self.compare_literature:
                    
            plt.plot(1./Tsat_lit,np.log10(Psat_lit),'ks',mfc='None',label='Potoff')
        
        plt.xlabel(r'$1/T^{\rm sat} (K)$')
        plt.ylabel(r'$\log_{\rm 10}(P_{\rm v}^{\rm sat}/\rm bar)$')
        plt.xticks([1./Temp_VLE.max(),1./Temp_VLE.min()])
        plt.legend()
        plt.show()
        
        plt.figure(figsize=[6,6])
        
        if compare_RP:
            plt.plot(Tsat_RP,np.log10(Psat_RP),'k-',label='REFPROP') 
        
        plt.plot(Temp_VLE,np.log10(Psat),'ro',label='MBAR-GCMC')
        plt.plot(Temp_VLE,np.log10(Psat_ig),'r--',label='Ideal gas')
        
        if self.compare_literature:
                    
            plt.plot(Tsat_lit,np.log10(Psat_lit),'ks',mfc='None',label='Potoff')
        
        plt.xlabel(r'$T^{\rm sat} (K)$')
        plt.ylabel(r'$\log_{\rm 10}(P_{\rm v}^{\rm sat}/\rm bar)$')
        plt.legend()
        plt.show()
        
        if compare_RP:
            
            Temp_compare = Temp_VLE[Temp_VLE >= max([Tsat_RP.min(), Temp_VLE.min()])]
            rhovap_compare = rhovap[Temp_VLE >= max([Tsat_RP.min(), Temp_VLE.min()])]
            rholiq_compare = rholiq[Temp_VLE >= max([Tsat_RP.min(), Temp_VLE.min()])]
            Psat_compare = Psat[Temp_VLE >= max([Tsat_RP.min(), Temp_VLE.min()])]
            
            rhovap_compare = rhovap_compare[Temp_compare <= min([Tsat_RP.max(), Temp_VLE.max()])]
            rholiq_compare = rholiq_compare[Temp_compare <= min([Tsat_RP.max(), Temp_VLE.max()])]
            Psat_compare = Psat_compare[Temp_compare <= min([Tsat_RP.max(), Temp_VLE.max()])]
            
            Temp_compare = Temp_compare[Temp_compare <= min([Tsat_RP.max(), Temp_VLE.max()])]
            
            rhov_RP_compare = np.interp(Temp_compare,Tsat_RP,rhov_RP)
            rhol_RP_compare = np.interp(Temp_compare,Tsat_RP,rhol_RP)
            Psat_RP_compare = np.interp(Temp_compare,Tsat_RP,Psat_RP)
                        
            plt.figure(figsize=[6,6])
            plt.plot(Temp_compare,rhovap_compare-rhov_RP_compare,'ro',mfc='None')
            plt.xlabel(r'$T^{\rm sat} (K)$')
            plt.ylabel(r'$\rho_{\rm v,MBAR}^{\rm sat} - \rho_{\rm v,REFPROP}^{\rm sat}$ (kg/m$^3$)')
            plt.legend()
            plt.show()
            
            plt.figure(figsize=[6,6])
            plt.plot(Temp_compare,(rhovap_compare-rhov_RP_compare)/rhov_RP_compare*100.,'ro',mfc='None')
            plt.xlabel(r'$T^{\rm sat} (K)$')
            plt.ylabel(r'$(\rho_{\rm v,MBAR}^{\rm sat} - \rho_{\rm v,REFPROP}^{\rm sat})/\rho_{\rm v,REFPROP}^{\rm sat}) \times 100$ %')
            plt.legend()
            plt.show()
        
            plt.figure(figsize=[6,6])
            plt.plot(Temp_compare,rholiq_compare-rhol_RP_compare,'ro',mfc='None')
            plt.xlabel(r'$T^{\rm sat} (K)$')
            plt.ylabel(r'$\rho_{\rm l,MBAR}^{\rm sat} - \rho_{\rm l,REFPROP}^{\rm sat}$ (kg/m$^3$)')
            plt.legend()
            plt.show()
            
            plt.figure(figsize=[6,6])
            plt.plot(Temp_compare,(rholiq_compare-rhol_RP_compare)/rhol_RP_compare*100.,'ro',mfc='None')
            plt.xlabel(r'$T^{\rm sat} (K)$')
            plt.ylabel(r'$(\rho_{\rm l,MBAR}^{\rm sat} - \rho_{\rm l,REFPROP}^{\rm sat})/\rho_{\rm l,REFPROP}^{\rm sat}) \times 100$ %')
            plt.legend()
            plt.show()
            
            plt.figure(figsize=[6,6])
            plt.plot(Temp_compare,Psat_compare-Psat_RP_compare,'ro',mfc='None')
            plt.xlabel(r'$T^{\rm sat} (K)$')
            plt.ylabel(r'$P_{\rm v,MBAR}^{\rm sat} - P_{\rm v,REFPROP}^{\rm sat}$ (bar)')
            plt.legend()
            plt.show()
            
            plt.figure(figsize=[6,6])
            plt.plot(Temp_compare,(Psat_compare-Psat_RP_compare)/Psat_RP_compare*100.,'ro',mfc='None')
            plt.xlabel(r'$T^{\rm sat} (K)$')
            plt.ylabel(r'$(P_{\rm v,MBAR}^{\rm sat} - P_{\rm v,REFPROP}^{\rm sat})/P_{\rm v,REFPROP}^{\rm sat}) \times 100$ %')
            plt.legend()
            plt.show()
        
        if self.compare_literature:
            
#            if not (Temp_VLE==Tsat_lit).all():
#            rholiq = rholiq[Temp_VLE.argsort()]            
#            Temp_compare = Temp_VLE[Temp_VLE >= max([Tsat_lit.min(), Temp_VLE.min()])]
#            rhovap_compare = rhovap[Temp_VLE >= max([Tsat_lit.min(), Temp_VLE.min()])]
#            rholiq_compare = rholiq[Temp_VLE >= max([Tsat_lit.min(), Temp_VLE.min()])]
#            Psat_compare = Psat[Temp_VLE >= max([Tsat_lit.min(), Temp_VLE.min()])]
#            
#            rhovap_compare = rhovap_compare[Temp_compare <= min([Tsat_lit.max(), Temp_VLE.max()])]
#            rholiq_compare = rholiq_compare[Temp_compare <= min([Tsat_lit.max(), Temp_VLE.max()])]
#            Psat_compare = Psat_compare[Temp_compare <= min([Tsat_lit.max(), Temp_VLE.max()])]
#            
#            Temp_compare = Temp_compare[Temp_compare <= min([Tsat_lit.max(), Temp_VLE.max()])]
#            
#            #This requires that Tsat_lit is sorted from smallest to greatest
#            
#            rhov_lit_compare = np.interp(Temp_compare,Tsat_lit,rhov_lit)
#            rhol_lit_compare = np.interp(Temp_compare,Tsat_lit,rhol_lit)
#            Psat_lit_compare = np.interp(Temp_compare,Tsat_lit,Psat_lit)
            
            ### For now, just assume Temp_VLE = Tsat_lit

            Temp_compare = Temp_VLE[Tsat_lit==Temp_VLE]
            rhovap_compare = rhovap[Tsat_lit==Temp_VLE]
            rholiq_compare = rholiq[Tsat_lit==Temp_VLE]
            Psat_compare = Psat[Tsat_lit==Temp_VLE]
            
            rhov_lit_compare = rhov_lit[Tsat_lit==Temp_VLE]
            rhol_lit_compare = rhol_lit[Tsat_lit==Temp_VLE]
            Psat_lit_compare = Psat_lit[Tsat_lit==Temp_VLE]
            
#            print(Temp_compare,Tsat_lit,Temp_VLE)
#            print(rhol_lit_compare,rholiq_compare)
#            print(Psat_lit_compare,Psat_compare)
            
            plt.figure(figsize=[6,6])
            plt.plot(Temp_compare,rhovap_compare-rhov_lit_compare,'ro',mfc='None')
            plt.xlabel(r'$T^{\rm sat} (K)$')
            plt.ylabel(r'$\rho_{\rm v,MBAR}^{\rm sat} - \rho_{\rm v,lit}^{\rm sat}$ (kg/m$^3$)')
            plt.legend()
            plt.show()
            
            plt.figure(figsize=[6,6])
            plt.plot(Temp_compare,(rhovap_compare-rhov_lit_compare)/rhov_lit_compare*100.,'ro',mfc='None')
            plt.xlabel(r'$T^{\rm sat} (K)$')
            plt.ylabel(r'$(\rho_{\rm v,MBAR}^{\rm sat} - \rho_{\rm v,lit}^{\rm sat})/\rho_{\rm v,lit}^{\rm sat}) \times 100$ %')
            plt.legend()
            plt.show()
        
            plt.figure(figsize=[6,6])
            plt.plot(Temp_compare,rholiq_compare-rhol_lit_compare,'ro',mfc='None')
            plt.xlabel(r'$T^{\rm sat} (K)$')
            plt.ylabel(r'$\rho_{\rm l,MBAR}^{\rm sat} - \rho_{\rm l,lit}^{\rm sat}$ (kg/m$^3$)')
            plt.legend()
            plt.show()
            
            plt.figure(figsize=[6,6])
            plt.plot(Temp_compare,(rholiq_compare-rhol_lit_compare)/rhol_lit_compare*100.,'ro',mfc='None')
            plt.xlabel(r'$T^{\rm sat} (K)$')
            plt.ylabel(r'$(\rho_{\rm l,MBAR}^{\rm sat} - \rho_{\rm l,lit}^{\rm sat})/\rho_{\rm l,lit}^{\rm sat}) \times 100$ %')
            plt.legend()
            plt.show()
        
            plt.figure(figsize=[6,6])
            plt.plot(Temp_compare,Psat_compare-Psat_lit_compare,'ro',mfc='None')
            plt.xlabel(r'$T^{\rm sat} (K)$')
            plt.ylabel(r'$P_{\rm v,MBAR}^{\rm sat} - P_{\rm v,lit}^{\rm sat}$ (bar)')
            plt.legend()
            plt.show()
            
            plt.figure(figsize=[6,6])
            plt.plot(Temp_compare,(Psat_compare-Psat_lit_compare)/Psat_lit_compare*100.,'ro',mfc='None')
            plt.xlabel(r'$T^{\rm sat} (K)$')
            plt.ylabel(r'$(P_{\rm v,MBAR}^{\rm sat} - P_{\rm v,lit}^{\rm sat})/P_{\rm v,lit}^{\rm sat}) \times 100$ %')
            plt.legend()
            plt.show()
        
    def print_VLE(self):
        '''
        Prints the saturation densities and compares with literature values if available
        '''
        Temp_VLE, rholiq, rhovap,Psat = self.Temp_VLE, self.rholiq, self.rhovap,self.Psat
        
        fT = open(self.root_path+'Tsat','w')
        fv = open(self.root_path+'rhovsat','w')
        fl = open(self.root_path+'rholsat','w')
        fp = open(self.root_path+'Psat','w')
        for Temp, rhov, rhol,Pv in zip(Temp_VLE,rhovap,rholiq,Psat):
            fT.write(str(Temp)+'\n')
            fv.write(str(rhov)+'\n')
            fl.write(str(rhol)+'\n')
            fp.write(str(Pv)+'\n')
        fT.close()
        fv.close()
        fl.close()
        fp.close()
        
    def mu_scan(self,Temp_VLE):
        '''
        Plots a scan of mu to help visualize the optimization.
        '''
        
        self.Temp_VLE = Temp_VLE

        self.build_MBAR_VLE_matrices()
        mu_VLE_guess, mu_lower_bound, mu_upper_bound = self.mu_guess_bounds()
        mu_range = np.linspace(mu_lower_bound[0],mu_upper_bound[0],10)
        
        sqdeltaW_plot = np.zeros([len(mu_range),len(Temp_VLE)])
        
        for i, mu in enumerate(mu_range):
            mu_array = mu*np.ones(len(Temp_VLE))
            sqdeltaW_plot[i] = self.sqdeltaW(mu_array)
        plt.plot(mu_range,sqdeltaW_plot)
        plt.xlabel(r'$\mu$ (K)')
        plt.ylabel(r'$(\Delta W^{\rm sat})^2$')
        plt.show()
        
        mu_opt = mu_range[sqdeltaW_plot.argmin(axis=0)]
        mu_lower = mu_range[sqdeltaW_plot.argmin(axis=0)-1]
        mu_upper = mu_range[sqdeltaW_plot.argmin(axis=0)+1]
        
        return mu_opt, mu_lower, mu_upper
    
    def calc_abs_press_int(self,show_plot=True):
        '''
        Fits ln(Xi) with respect to N for low-density vapor
        '''
        Temp_sim, u_kn_sim,f_k_sim,sumN_k = self.Temp_sim, self.u_kn_sim,self.f_k_sim,self.sumN_k
        nTsim, U_flat, Nmol_flat,Ncut = self.nTsim, self.U_flat, self.Nmol_flat, self.Ncut
        
        Temp_IG = np.min(Temp_sim[self.mu_sim == self.mu_sim.min()]) 
#        print(Temp_IG)

        mu_IG = np.linspace(2.*self.mu_opt[self.Temp_VLE==Temp_IG],5.*self.mu_opt[self.Temp_VLE==Temp_IG],10)

        N_k_all = self.K_sim[:]
        N_k_all.extend([0]*len(mu_IG))

        u_kn_IG = np.zeros([len(mu_IG),sumN_k])
        u_kn_all = np.concatenate((u_kn_sim,u_kn_IG))
        
        f_k_guess = np.concatenate((f_k_sim,np.zeros(len(mu_IG))))

        for jT, mu in enumerate(mu_IG):
            
            u_kn_all[nTsim+jT,:] = self.U_to_u(U_flat,Temp_IG,mu,Nmol_flat)

        mbar = MBAR(u_kn_all,N_k_all,initial_f_k=f_k_guess)
                
        sumW_IG = np.sum(mbar.W_nk[:,nTsim:][Nmol_flat<Ncut],axis=0)
         
        Nmol_IG = np.sum(mbar.W_nk[:,nTsim:][Nmol_flat<Ncut].T*Nmol_flat[Nmol_flat<Ncut],axis=1)/sumW_IG
#        print(sumW_IG,Nmol_IG)
#        print(mbar.W_nk[:,nTsim:][Nmol_flat<Ncut].T)
#        print(mbar.W_nk[:,nTsim:][Nmol_flat<Ncut].T*Nmol_flat[Nmol_flat<Ncut])
        ### Store previous solutions to speed-up future convergence of MBAR
        Deltaf_ij = mbar.getFreeEnergyDifferences(return_theta=False)[0]
        f_k_IG = Deltaf_ij[nTsim:,0]
#        print(f_k_sim,f_k_guess[:nTsim+1],Deltaf_ij[0,:nTsim],f_k_IG)#,Nmol_IG,press_IG,Psat)

        fit=stats.linregress(Nmol_IG[mu_IG<2.*self.mu_sim.min()],f_k_IG[mu_IG<2.*self.mu_sim.min()])
        
        if show_plot:
            
            Nmol_plot = np.linspace(Nmol_IG.min(),Nmol_IG.max(),50)
            lnXi_plot = fit.intercept + fit.slope*Nmol_plot

            plt.figure(figsize=[6,6])
            plt.plot(Nmol_IG,f_k_IG,'bo',mfc='None',label='MBAR-GCMC')
            plt.plot(Nmol_plot,lnXi_plot,'k-',label='Linear fit')
            plt.xlabel('Number of Molecules')
            plt.ylabel(r'$\ln(\Xi)$')
            plt.legend()
            plt.show()
            
            print('Slope for ideal gas is 1, actual slope is: '+str(fit.slope))
            print('Intercept for absolute pressure is:'+str(fit.intercept))
        
        self.abs_press_int, self.Temp_IG, self.f_k_IG, self.Nmol_IG = fit.intercept, Temp_IG, f_k_IG, Nmol_IG
        
    def calc_Psat(self):
        '''
        Computes the saturated vapor pressure
        '''
        self.calc_abs_press_int()
        f_k_opt, nTsim, Temp_VLE, Vbox, abs_press_int, Temp_IG = self.f_k_opt, self.nTsim, self.Temp_VLE, self.Vbox_sim[0], self.abs_press_int, self.Temp_IG
        
        Psat = kb * Temp_VLE * (f_k_opt[nTsim:]-np.log(2.) - abs_press_int) / Vbox / Ang3tom3 * Jm3tobar

        self.Psat = Psat
        
def main():
       
    filepaths = []
            
#    root_path = 'hexane_Potoff/'
#    hist_num=['1','2','3','4','5','6','7','8','9']
    
    root_path = 'hexane_Potoff_replicates/'
#    root_path = 'hexane_Potoff_replicates_2/'
#    root_path = 'hexane_eps_scaled/'
    Temp_range = ['510','470','430','480','450','420','390','360','330']
    hist_num=['2','2','2','2','2','2','2','2','2']
    
    for iT, Temp in enumerate(Temp_range):
        
        hist_name='/his'+hist_num[iT]+'a.dat' #Only if loading hexane_Potoff
        
        filepaths.append(root_path+Temp+hist_name)

    Mw_hexane  = 12.0109*6.+1.0079*(2.*6.+2.) #[gm/mol]
    
#    Temp_VLE_plot = Tsat_Potoff
#    Temp_VLE_plot = np.array([360., 350.])
    MBAR_GCMC_trial = MBAR_GCMC(root_path,filepaths,Mw_hexane,compare_literature=True)
#    MBAR_GCMC_trial.plot_histograms()
#    MBAR_GCMC_trial.plot_2dhistograms()
    MBAR_GCMC_trial.solve_VLE(Temp_VLE_plot)
    MBAR_GCMC_trial.plot_VLE()
#    MBAR_GCMC_trial.mu_scan(Temp_VLE_plot)
    
if __name__ == '__main__':
    '''
    python MBAR_GCMC_class.py  
    '''

    main()   