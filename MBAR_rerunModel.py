"""
MBAR GCMC
"""

import numpy as np
import matplotlib.pyplot as plt
from pymbar import MBAR
from Golden_search_multi import GOLDEN_multi, GOLDEN
from scipy import stats

### Figure font size
font = {'size' : '18'}
plt.rc('font',**font)

# Physical constants
N_A = 6.02214086e23 #[/mol]
nm3_to_ml = 10**21
bar_nm3_to_kJ_per_mole = 0.0602214086
Ang3tom3 = 10**-30
gmtokg = 1e-3
kb = 1.3806485e-23 #[J/K]
Jm3tobar = 1e-5
Rg = kb*N_A #[J/mol/K]
JtokJ = 1e-3

compute_RMS = lambda yhat,yset: np.sqrt(np.mean((yhat - yset)**2.))
compute_MAPD = lambda yhat,yset: np.mean(np.abs((yhat - yset)/yset*100.))
compute_AD = lambda yhat,yset: np.mean((yhat - yset)/yset*100.)
compute_APD = lambda yhat,ydata: np.abs((yhat - ydata)/ydata*100.)

### Branched alkanes
#Score_w8 = np.array([0.6135,0.0123,0.2455,0.0245,0.0613,0.0061,0.0245,0.0123])
### Alkynes
Score_w8 = np.array([0.757,0.0,0.152,0.0,0.076,0.0,0.015,0.0])

convert_AD2Score = lambda AD_rhol, AD_rhov, AD_Psat: Score_w8[0]*np.abs(AD_rhol) + Score_w8[1]*np.abs(AD_rhov) + Score_w8[2]*np.abs(AD_Psat)
convert_MAPD2Score = lambda MAPD_rhol, MAPD_rhov, MAPD_Psat: Score_w8[0]*MAPD_rhol + Score_w8[1]*MAPD_rhov + Score_w8[2]*MAPD_Psat
                                                                 
#RMS_rhol = lambda rhol,rhol_RP: np.sqrt(np.mean((rhol - rhol_RP)**2))
#MAPD_rhol = lambda rhol,rhol_RP: np.mean(np.abs((rhol - rhol_RP)/rhol_RP*100.))
#AD_rhol = lambda rhol,rhol_RP: np.mean((rhol - rhol_RP)/rhol_RP*100.)
#
#RMS_rhov = lambda rhov,rhov_RP: np.sqrt(np.mean((rhov - rhov_RP)**2))
#MAPD_rhov = lambda rhov,rhov_RP: np.mean(np.abs((rhov - rhov_RP)/rhov_RP*100.))
#AD_rhov = lambda rhov,rhov_RP: np.mean((rhov - rhov_RP)/rhov_RP*100.)

class MBAR_GCMC():
    def __init__(self,root_path,filepaths,Mw,trim_data=False,compare_literature=False,trim_size=5000):
        self.root_path = root_path
        self.filepaths = filepaths
        self.trim_data = trim_data 
        self.trim_size = trim_size
        self.Mw = Mw
        self.compare_literature = compare_literature
        self.extract_all_sim_data()
        self.min_max_sim_data()
        self.build_MBAR_sim()
        self.Ncut = self.solve_Ncut()
                
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
        Urr_data_sim = []
        K_sim = []
        
        Nmol_flat = np.array([])
        U_flat = np.array([])
        Urr_flat = np.array([])
        
        N_min_sim = []
        N_max_sim = []
        
        U_min_sim = []
        U_max_sim = []
        
        Temp_sim = np.zeros(len(filepaths))
        mu_sim = np.zeros(len(filepaths))
        Vbox_sim = np.zeros(len(filepaths))
        
        for ipath, filepath in enumerate(filepaths):
            N_data, U_data, Urr_data, Temp, mu, Vbox = self.extract_data(filepath)
            
            Temp_sim[ipath] = Temp
            mu_sim[ipath] = mu
            Vbox_sim[ipath] = Vbox
                    
            N_data_sim.append(N_data)
            U_data_sim.append(U_data)
            Urr_data_sim.append(Urr_data)
            K_sim.append(len(N_data))
            
            Nmol_flat = np.append(Nmol_flat,N_data)
            U_flat = np.append(U_flat,U_data)
            Urr_flat = np.append(Urr_flat,Urr_data)
#            print(N_data_sim)        
            N_min_sim.append(np.min(N_data))
            N_max_sim.append(np.max(N_data))
             
            U_min_sim.append(np.min(U_data))
            U_max_sim.append(np.max(U_data))
   
        self.Temp_sim, self.mu_sim, self.Vbox_sim, self.N_data_sim, self.U_data_sim, self.K_sim, self.Nmol_flat, self.U_flat, self.Urr_data_sim, self.Urr_flat = Temp_sim, mu_sim, Vbox_sim, N_data_sim, U_data_sim, K_sim, Nmol_flat, U_flat, Urr_data_sim, Urr_flat
            
    def extract_data(self,filepath):
        '''
        For a single filepath, returns the number of molecules, N, internal
        energy, U, temperature, Temp, chemical potential, mu, and volumbe, Vbox
        '''
        NU_data = np.loadtxt(filepath,skiprows=1)
        if self.trim_data:
            subset_size = self.trim_size
        else:
            subset_size = len(NU_data)
        subset_data = np.random.choice(np.arange(0,len(NU_data)),size=subset_size,replace=False)
        N_data = NU_data[subset_data,0]
        U_data = NU_data[subset_data,1] #[K]
        Urr_data = NU_data[subset_data,2] #[K]
        mu_V_T = np.genfromtxt(filepath,skip_footer=len(NU_data))
        Temp = mu_V_T[0] #[K]
        mu = mu_V_T[2] #[K]
        Lbox = mu_V_T[3] #[Angstrom]
        Vbox = Lbox**3 #[Angstrom^3]
        return N_data, U_data, Urr_data, Temp, mu, Vbox
    
    def bootstrap_data(self):
        '''
        Bootstrapping re-sampling of data
        '''
        Nmol_flat, U_flat, nSnapshots = self.Nmol_flat, self.U_flat, self.K_sim
        
        Nmol_flat_stored = Nmol_flat.copy()
        U_flat_stored = U_flat.copy()
        
        irand = np.random.randint(0,len(Nmol_flat),len(Nmol_flat))
        
        Nmol_flat = Nmol_flat_stored[irand] #np.random.choice(np.arange(0,len(Nmol_flat)),size=len(Nmol_flat),replace=True)
        U_flat = U_flat_stored[irand]
        
        self.Nmol_flat, self.U_flat, self.Nmol_flat_stored, self.U_flat_stored = Nmol_flat, U_flat, Nmol_flat_stored, U_flat_stored
        
    def VLE_uncertainty(self,Temp_VLE,eps_scaled=1.,nBoots=20):
        '''
        Bootstrap the VLE uncertainties rather than using MBAR uncertainties
        '''
        
        rholiqBoots = np.zeros([len(Temp_VLE),nBoots])
        rhovapBoots = np.zeros([len(Temp_VLE),nBoots])
        PsatBoots = np.zeros([len(Temp_VLE),nBoots])
        DeltaHvBoots = np.zeros([len(Temp_VLE),nBoots])
        
        for iBoot in range(nBoots):
            
            self.bootstrap_data()
            self.build_MBAR_sim()
            self.Ncut = self.solve_Ncut()
            self.solve_VLE(Temp_VLE,eps_scaled)
            
            rholiqBoots[:,iBoot] = self.rholiq
            rhovapBoots[:,iBoot] = self.rhovap
            PsatBoots[:,iBoot] = self.Psat
            DeltaHvBoots[:,iBoot] = self.DeltaHv
                        
        tstat = stats.t.interval(0.95,nBoots,loc=0,scale=1)[1]
                       
        ### Should I use the standard error, i.e., divide by the square root of nBoots?
        
        urholiq = tstat * np.std(rholiqBoots,axis=1)
        urhovap = tstat * np.std(rhovapBoots,axis=1)
        uPsat = tstat * np.std(PsatBoots,axis=1)
        uDeltaHv = tstat * np.std(DeltaHvBoots,axis=1)
        
        self.urholiq, self.urhovap, self.uPsat, self.uDeltaHv = urholiq, urhovap, uPsat, uDeltaHv
        self.rholiqBoots, self.rhovapBoots, self.PsatBoots, self.DeltaHvBoots = rholiqBoots, rhovapBoots, PsatBoots, DeltaHvBoots

        return urholiq, urhovap, uPsat, uDeltaHv
    
    def Score_uncertainty(self,Temp_VLE,rhol_RP,rhov_RP,Psat_RP,DeltaHv_RP,remove_low_high_Tsat=False,eps_scaled=1.,nBoots=20):
        '''
        Bootstrap the uncertainty in the scoring function
        '''
                
        self.Temp_VLE, self.rhol_RP, self.rhov_RP, self.Psat_RP, self.DeltaHv_RP = Temp_VLE, rhol_RP, rhov_RP, Psat_RP, DeltaHv_RP
        
        if remove_low_high_Tsat:  #Remove the low T and high T ends for more stable results
            self.Temp_VLE = self.Temp_VLE[2:-2]
            self.rhol_RP = rhol_RP[2:-2]
            self.rhov_RP = rhov_RP[2:-2]
            self.Psat_RP = Psat_RP[2:-2]
            self.DeltaHv_RP = DeltaHv_RP[2:-2]
            
        self.eps_computed = []
        self.Score_computed = []
        
        rholiqBoots = np.zeros([len(Temp_VLE),nBoots])
        rhovapBoots = np.zeros([len(Temp_VLE),nBoots])
        PsatBoots = np.zeros([len(Temp_VLE),nBoots])
        DeltaHvBoots = np.zeros([len(Temp_VLE),nBoots])
        ScoreBoots = np.zeros(nBoots)
        
        self.Ncut_stored = self.Ncut
        
        for iBoot in range(nBoots):
            
            self.bootstrap_data()
            self.build_MBAR_sim()
#            self.Ncut = self.solve_Ncut()
            self.Ncut = int(self.Ncut_stored*np.random.uniform(0.95,1.05))
            print('Liquid and vapor phase is divided by Nmol = '+str(self.Ncut))
            Score_iBoot = self.Scoring_function(eps_scaled)
            print('Scoring function value for '+str(iBoot)+'th bootstrap = '+str(Score_iBoot))
            
            rholiqBoots[:,iBoot] = self.rholiq
            rhovapBoots[:,iBoot] = self.rhovap
            PsatBoots[:,iBoot] = self.Psat
            DeltaHvBoots[:,iBoot] = self.DeltaHv
            ScoreBoots[iBoot] = Score_iBoot                        
                        
        tstat = stats.t.interval(0.95,nBoots,loc=0,scale=1)[1]
                       
        ### Should I use the standard error, i.e., divide by the square root of nBoots?
        
        urholiq = tstat * np.std(rholiqBoots,axis=1)
        urhovap = tstat * np.std(rhovapBoots,axis=1)
        uPsat = tstat * np.std(PsatBoots,axis=1)
        uDeltaHv = tstat * np.std(DeltaHvBoots,axis=1)
        
        ### The scoring function values are certainly not normal
        i95 = int(0.025*nBoots)
        Score_low95 = np.sort(ScoreBoots)[i95]
        Score_high95 = np.sort(ScoreBoots)[-(i95+1)]
        
        self.urholiq, self.urhovap, self.uPsat, self.uDeltaHv, self.Score_low95, self.Score_high95 = urholiq, urhovap, uPsat, uDeltaHv, Score_low95, Score_high95
        
        return urholiq, urhovap, uPsat, uDeltaHv, Score_low95, Score_high95
        
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
        mu_upper_bound = mu_sim_high*0.98
        
        return mu_VLE_guess, mu_lower_bound, mu_upper_bound
        
    def solve_VLE(self,Temp_VLE,eps_scaled=1.,show_plot=False):
        '''
        Determine optimal values of mu that result in equal pressures by 
        minimizing the square difference of the weights in the liquid and vapor
        phases. Subsequentally, calls the function to compute the saturation
        properties.
        '''
        
        self.Temp_VLE = Temp_VLE
        
        self.build_MBAR_VLE_matrices()
        mu_VLE_guess, mu_lower_bound, mu_upper_bound = self.mu_guess_bounds()
        sqdeltaW_scaled = lambda mu: self.sqdeltaW(mu,eps_scaled)
        
        ### Optimization of mu
#        try:
#            mu_VLE_guess = self.mu_opt
#        except:
#            mu_VLE_guess, mu_lower_bound, mu_upper_bound = self.mu_scan(Temp_VLE,eps_scaled)
        mu_VLE_guess, mu_lower_bound, mu_upper_bound = self.mu_scan(Temp_VLE,eps_scaled) 
        mu_opt = GOLDEN_multi(sqdeltaW_scaled,mu_VLE_guess,mu_lower_bound,mu_upper_bound,TOL=0.0001,maxit=30)

        self.f_k_opt = self.f_k_guess.copy()
        self.mu_opt = mu_opt
        
        self.calc_rhosat()
        self.calc_Psat(eps_scaled)
        self.calc_Usat(eps_scaled)
        self.calc_DeltaHv(eps_scaled)
        
#        self.print_VLE()
        
        if show_plot:
        
            sqdeltaW_opt = self.sqdeltaW(mu_opt)
            
            plt.plot(Temp_VLE,mu_opt,'k-',label=r'$\mu_{\rm opt}$')
            plt.plot(self.Temp_sim,self.mu_sim,'ro',mfc='None',label='Simulation')
            plt.plot(Temp_VLE,mu_VLE_guess,'b--',label=r'$\mu_{\rm guess}$')
            plt.xlabel(r'$T$ (K)')
            plt.ylabel(r'$\mu_{\rm opt}$ (K)')
            plt.xlim([300,550])
#            plt.ylim([-4200,-3600])
            plt.legend()
            plt.show()
            
            plt.plot(Temp_VLE,sqdeltaW_opt,'ko')
            plt.xlabel(r'$T$ (K)')
            plt.ylabel(r'$(\Delta W^{\rm sat})^2$')
            plt.show()
            
            print("Effective number of samples")
            print (self.mbar.computeEffectiveSampleNumber())
            print('\nWhich is approximately '+str(self.mbar.computeEffectiveSampleNumber()/self.sumN_k*100.)+'% of the total snapshots')
     
    def sqdeltaW(self,mu_VLE,eps_scaled):
        '''
        Computes the square difference between the sum of the weights in the
        vapor and liquid phases.
        Stores the optimal reduced free energy as f_k_guess for future iterations
        Stores mbar, sumWliq, and sumWvap for computing VLE properties if converged
        '''
        
        nTsim, Nmol_flat,Ncut, f_k_guess, Temp_VLE, u_kn_all, N_k_all = self.nTsim, self.Nmol_flat, self.Ncut,self.f_k_guess, self.Temp_VLE, self.u_kn_all, self.N_k_all

        if eps_scaled == 1.:
        
            U_flat = self.U_flat
            
        else:
            
            U_flat = self.Urr_flat

        for jT, (Temp, mu) in enumerate(zip(Temp_VLE, mu_VLE)):
            
            u_kn_all[nTsim+jT,:] = self.U_to_u(U_flat,Temp,mu,Nmol_flat)

        mbar = MBAR(u_kn_all,N_k_all,initial_f_k=f_k_guess)
    
        sumWliq = np.sum(mbar.W_nk[:,nTsim:][Nmol_flat>Ncut],axis=0)
        sumWvap = np.sum(mbar.W_nk[:,nTsim:][Nmol_flat<=Ncut],axis=0)
        sqdeltaW_VLE = (sumWliq-sumWvap)**2
        
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
#        Nliq = mbar.computeExpectations(Nmol_flat)[0]
#        print(len(Nliq))
#        print(Nliq[nTsim:])
#        print(len(Nmol_flat))
#        print(len(Nmol_flat[Nmol_flat>Ncut]))
##        print(mbar.u_kn)
#        #print(len(mbar.u_kn[Nmol_flat>Ncut]))
##        Nliq = mbar.computeExpectations(Nmol_flat[Nmol_flat>Ncut],ukn=mbar.u_kn[Nmol_flat>Ncut])
##        print(Nliq)
# 
#        dNliq = mbar.computeExpectations(Nmol_flat)[1][nTsim:]
#        dNvap = mbar.computeExpectations(Nmol_flat)[1][nTsim:]
#
#        print(dNliq)
        
        Nmol_liq = Nmol_flat[Nmol_flat>Ncut]
        Nmol_vap = Nmol_flat[Nmol_flat<=Ncut]

        Nmol_compute_liq = Nmol_flat.copy()
        Nmol_compute_liq[Nmol_flat<=Ncut] = np.random.choice(Nmol_liq,size=len(Nmol_vap),replace=True)

        Nmol_compute_vap = Nmol_flat.copy()
        Nmol_compute_vap[Nmol_flat>Ncut] = np.random.choice(Nmol_vap,size=len(Nmol_liq),replace=True)
        
#        Nliq, dNliq = mbar.computeExpectations(Nmol_compute_liq)
#        Nvap, dNvap = mbar.computeExpectations(Nmol_compute_vap)
#        
#        print(Nliq[nTsim:])
#        print(Nvap[nTsim:])
#        print(dNliq[nTsim:])
#        print(dNvap[nTsim:])
        
        dNliq = mbar.computeExpectations(Nmol_compute_liq)[1][nTsim:]
        dNvap = mbar.computeExpectations(Nmol_compute_vap)[1][nTsim:]

        Nliq = np.sum(mbar.W_nk[:,nTsim:][Nmol_flat>Ncut].T*Nmol_flat[Nmol_flat>Ncut],axis=1)/sumWliq #Must renormalize by the liquid or vapor phase
        Nvap = np.sum(mbar.W_nk[:,nTsim:][Nmol_flat<=Ncut].T*Nmol_flat[Nmol_flat<=Ncut],axis=1)/sumWvap
#        print(Nliq,Nvap)

        rholiq = Nliq/Vbox * Mw / N_A * gmtokg / Ang3tom3 #[kg/m3]
        rhovap = Nvap/Vbox * Mw / N_A * gmtokg / Ang3tom3 #[kg/m3]
        
        drholiq = rholiq * dNliq / Nliq
        drhovap = rhovap * dNvap / Nvap
        
        self.rholiq, self.rhovap, self.Nliq, self.Nvap, self.drholiq, self.drhovap, self.dNliq, self.dNvap = rholiq, rhovap, Nliq, Nvap, drholiq, drhovap, dNliq, dNvap
        assert 1==2
        
    def calc_Usat(self,eps_scaled=1.):
        '''
        Computes the saturated liquid and vapor internal energies
        '''
        
        Nmol_flat, Ncut, mbar, sumWliq, sumWvap, nTsim, Nliq, Nvap = self.Nmol_flat, self.Ncut, self.mbar, self.sumWliq, self.sumWvap, self.nTsim, self.Nliq, self.Nvap
        
        if eps_scaled == 1.:
        
            U_flat = self.U_flat
            
        else:
            
            U_flat = self.Urr_flat
        
        #Convert energy to per molecule basis
#        Umol_flat = U_flat.copy()
#        Umol_flat[Nmol_flat > 0] /= Nmol_flat[Nmol_flat > 0] #Avoid dividing by zero
#      
#        Uliq = np.sum(mbar.W_nk[:,nTsim:][Nmol_flat>Ncut].T*Umol_flat[Nmol_flat>Ncut],axis=1)/sumWliq #Must renormalize by the liquid or vapor phase
#        Uvap = np.sum(mbar.W_nk[:,nTsim:][Nmol_flat<=Ncut].T*Umol_flat[Nmol_flat<=Ncut],axis=1)/sumWvap

        Uliq = np.sum(mbar.W_nk[:,nTsim:][Nmol_flat>Ncut].T*U_flat[Nmol_flat>Ncut],axis=1)/sumWliq #Must renormalize by the liquid or vapor phase
        Uvap = np.sum(mbar.W_nk[:,nTsim:][Nmol_flat<=Ncut].T*U_flat[Nmol_flat<=Ncut],axis=1)/sumWvap           
                           
        Uliq /= Nliq
        Uvap /= Nvap
                                      
        Uliq *= Rg #[J/mol]
        Uvap *= Rg #[J/mol]       
          
        self.Uliq, self.Uvap = Uliq, Uvap
    
    def plot_VLE(self,Tsat_RP,rhol_RP,rhov_RP,Tsat_Potoff,rhol_Potoff,rhov_Potoff):
        '''
        Plots the saturation densities and compares with literature values if available
        '''
        Temp_VLE, rholiq, rhovap = self.Temp_VLE, self.rholiq, self.rhovap
           
        plt.plot(rhov_RP,Tsat_RP,'k-',label='REFPROP')
        plt.plot(rhol_RP,Tsat_RP,'k-')
        
        plt.plot(rhovap,Temp_VLE,'ro',label='MBAR-GCMC')
        plt.plot(rholiq,Temp_VLE,'ro')
        
        if self.compare_literature:
            plt.plot(rhov_Potoff,Tsat_Potoff,'ks',mfc='None',label='Potoff')
            plt.plot(rhol_Potoff,Tsat_Potoff,'ks',mfc='None')
        
        plt.xlabel(r'$\rho$ (kg/m$^3$)')
        plt.ylabel(r'$T$ (K)')
        plt.xlim([-10,1.04*rholiq.max()])
        plt.ylim([0.98*Temp_VLE.min(),1.02*Temp_VLE.max()])
        plt.legend()
        plt.show()
        
    def print_VLE(self):
        '''
        Prints the saturation densities and compares with literature values if available
        '''
        Temp_VLE, rholiq, rhovap = self.Temp_VLE, self.rholiq, self.rhovap
        
        fT = open(self.root_path+'Tsat','w')
        fv = open(self.root_path+'rhovsat','w')
        fl = open(self.root_path+'rholsat','w')
        for Temp, rhov, rhol in zip(Temp_VLE,rhovap,rholiq):
            fT.write(str(Temp)+'\n')
            fv.write(str(rhov)+'\n')
            fl.write(str(rhol)+'\n')   
        fT.close()
        fv.close()
        fl.close()
        
    def mu_scan(self,Temp_VLE,eps_scaled,show_plot=False):
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
            sqdeltaW_plot[i] = self.sqdeltaW(mu_array,eps_scaled)
        
        if show_plot:
            plt.plot(mu_range,sqdeltaW_plot)
            plt.xlabel(r'$\mu$ (K)')
            plt.ylabel(r'$(\Delta W^{\rm sat})^2$')
            plt.show()
        
        mu_opt = mu_range[sqdeltaW_plot.argmin(axis=0)]
        mu_lower = mu_opt - (mu_range[1]-mu_range[0]) #mu_range[sqdeltaW_plot.argmin(axis=0)-1]
        mu_upper = mu_opt + (mu_range[1]-mu_range[0]) #mu_range[sqdeltaW_plot.argmin(axis=0)+1]
        
        return mu_opt, mu_lower, mu_upper
    
    def calc_abs_press_int(self,eps_scaled,show_plot=False):
        '''
        Fits ln(Xi) with respect to N for low-density vapor
        '''
        Temp_sim, u_kn_sim,f_k_sim,sumN_k = self.Temp_sim, self.u_kn_sim,self.f_k_sim,self.sumN_k
        nTsim, Nmol_flat,Ncut = self.nTsim, self.Nmol_flat, self.Ncut

        if eps_scaled == 1.:
        
            U_flat = self.U_flat
            
        else:
            
            U_flat = self.Urr_flat
        
        
        Temp_IG = np.min(Temp_sim[self.mu_sim == self.mu_sim.min()]) 
#        print(Temp_IG)

#        mu_IG = np.linspace(2.*self.mu_opt[self.Temp_VLE==Temp_IG],5.*self.mu_opt[self.Temp_VLE==Temp_IG],10)
        mu_IG = np.linspace(self.mu_sim.min(),3.*self.mu_sim.min(),10)

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
        
    def calc_Psat(self,eps_scaled=1.):
        '''
        Computes the saturated vapor pressure
        '''
        try:
            abs_press_int = self.abs_press_int
        except:
            self.calc_abs_press_int(eps_scaled)
        f_k_opt, nTsim, Temp_VLE, Vbox, abs_press_int = self.f_k_opt, self.nTsim, self.Temp_VLE, self.Vbox_sim[0], self.abs_press_int
        
        Psat = kb * Temp_VLE * (f_k_opt[nTsim:]-np.log(2.) - abs_press_int) / Vbox / Ang3tom3 * Jm3tobar #-log(2) accounts for two phases
        self.Psat = Psat
        
    def calc_DeltaHv(self,eps_scaled=1.):
        '''
        Fits Psat to line to compute deltaHvap
        '''
        Psat, Temp_VLE, rholiq, rhovap, Uliq, Uvap, Mw = self.Psat, self.Temp_VLE, self.rholiq, self.rhovap, self.Uliq, self.Uvap, self.Mw

        Vliq = Mw / rholiq * gmtokg #[m3/mol]
        Vvap = Mw / rhovap * gmtokg #[m3/mol]

        DeltaUv = Uvap - Uliq #[J/mol]

        PV = Psat * (Vvap - Vliq) #[bar m3/mol]
        PV /= Jm3tobar #[J/mol]

        DeltaHv = DeltaUv + PV #[J/mol]

        DeltaHv *= JtokJ #[kJ/mol]

### Wrong method
#        logPsat = np.log(Psat)
#        invTemp = 1./Temp_VLE
#        
#        slope = np.polyfit(invTemp,logPsat,1)[1]
#        
#        DeltaHv = -slope * Rg #[J/mol]
#        
        self.DeltaHv = DeltaHv      
    
    def eps_optimize(self,Temp_VLE,rhol_RP,rhov_RP,Psat_RP,DeltaHv_RP,eps_low,eps_high,compound,remove_low_high_Tsat=False):
        
        self.Temp_VLE, self.rhol_RP, self.rhov_RP, self.Psat_RP, self.DeltaHv_RP = Temp_VLE, rhol_RP, rhov_RP, Psat_RP, DeltaHv_RP
        
        if remove_low_high_Tsat:  #Remove the low T and high T ends for more stable results
            self.Temp_VLE = self.Temp_VLE[2:-2]
            self.rhol_RP = rhol_RP[2:-2]
            self.rhov_RP = rhov_RP[2:-2]
            self.Psat_RP = Psat_RP[2:-2]
            self.DeltaHv_RP = DeltaHv_RP[2:-2]
            
        self.eps_computed = []
        self.Score_computed = []
            
        eps_opt = GOLDEN(self.Scoring_function,eps_low,1.,eps_high,TOL=0.0001)
            
        Score_opt = self.Scoring_function(eps_opt)
        
        eps_computed = np.array(self.eps_computed)
        Score_computed = np.array(self.Score_computed)
        
        plt.figure(figsize=(8,8))  
        plt.plot(np.sort(eps_computed),Score_computed[np.argsort(eps_computed)],'ko-',mfc='None')
#        plt.plot(eps_computed,Score_computed,'b--',mfc='None')
        plt.plot(eps_opt,Score_opt,'r*',markersize=10,label='Optimal')
        plt.xlabel(r'$\epsilon / \epsilon_{\rm Potoff}$')
        plt.ylabel(r'Score')
        plt.title(compound)        
#        plt.legend()
#        plt.savefig('figures/'+compound+'_Score_eps_scan.pdf')
        plt.show()
        
        self.eps_opt, self.Score_opt = eps_opt, Score_opt
        
    def Scoring_function(self,eps_scaled):
            
        self.solve_VLE(self.Temp_VLE, eps_scaled)
        MAPD_rhol = compute_MAPD(self.rholiq,self.rhol_RP)
        MAPD_rhov = compute_MAPD(self.rhovap,self.rhov_RP)
        MAPD_Psat = compute_MAPD(self.Psat,self.Psat_RP)
        MAPD_DeltaHv = compute_MAPD(self.DeltaHv,self.DeltaHv_RP)
        
        APD_rhol = compute_APD(self.rholiq,self.rhol_RP)
        APD_rhov = compute_APD(self.rhovap,self.rhov_RP)
        APD_Psat = compute_APD(self.Psat,self.Psat_RP)
        APD_DeltaHv = compute_APD(self.DeltaHv,self.DeltaHv_RP)
                
        nTemp_VLE = len(self.Temp_VLE)

        dAPD_rhol = np.zeros(nTemp_VLE-1)
        dAPD_rhov = np.zeros(nTemp_VLE-1)
        dAPD_Psat = np.zeros(nTemp_VLE-1)
        dAPD_DeltaHv = np.zeros(nTemp_VLE-1)
        
        for iT in range(nTemp_VLE-1):
            
            dAPD_rhol[iT] = (APD_rhol[iT+1]-APD_rhol[iT])/(self.Temp_VLE[iT+1]-self.Temp_VLE[iT])
            dAPD_rhov[iT] = (APD_rhov[iT+1]-APD_rhov[iT])/(self.Temp_VLE[iT+1]-self.Temp_VLE[iT])
            dAPD_Psat[iT] = (APD_Psat[iT+1]-APD_Psat[iT])/(self.Temp_VLE[iT+1]-self.Temp_VLE[iT])
            dAPD_DeltaHv[iT] = (APD_DeltaHv[iT+1]-APD_DeltaHv[iT])/(self.Temp_VLE[iT+1]-self.Temp_VLE[iT])
        
        MdAPD_rhol = np.mean(dAPD_rhol) 
        MdAPD_rhov = np.mean(dAPD_rhov)   
        MdAPD_Psat = np.mean(dAPD_Psat)               
        MdAPD_DeltaHv = np.mean(dAPD_DeltaHv)
    
        Score = Score_w8[0]*MAPD_rhol + Score_w8[1]*MAPD_rhov + Score_w8[2]*MAPD_Psat + Score_w8[3]*MAPD_DeltaHv
        Score += Score_w8[4]*MdAPD_rhol + Score_w8[5]*MdAPD_rhov + Score_w8[6]*MdAPD_Psat + Score_w8[7]*MdAPD_DeltaHv

        self.eps_computed.append(eps_scaled)
        self.Score_computed.append(Score)
        
        return Score
    
    def eps_scan(self,Temp_VLE,rhol_RP,rhov_RP,Psat_RP,rhol_Potoff,rhov_Potoff,Psat_Potoff,eps_low,eps_high,neps,compound,remove_low_high_Tsat=False):
        
        self.Temp_VLE = Temp_VLE
        
        if remove_low_high_Tsat:  #Remove the low T and high T ends for more stable results
            self.Temp_VLE = self.Temp_VLE[2:-2]
            rhol_RP = rhol_RP[2:-2]
            rhov_RP = rhov_RP[2:-2]
            Psat_RP = Psat_RP[2:-2]
            rhol_Potoff = rhol_Potoff[2:-2]
            rhov_Potoff = rhov_Potoff[2:-2]
            Psat_Potoff = Psat_Potoff[2:-2]
        
        eps_range = np.linspace(eps_low,eps_high,neps)
        
        RMS_rhol_plot = np.zeros(len(eps_range))
        AD_rhol_plot = np.zeros(len(eps_range))
        MAPD_rhol_plot = np.zeros(len(eps_range))

        RMS_rhov_plot = np.zeros(len(eps_range))
        AD_rhov_plot = np.zeros(len(eps_range))
        MAPD_rhov_plot = np.zeros(len(eps_range))
        
        RMS_logPsat_plot = np.zeros(len(eps_range))
        AD_Psat_plot = np.zeros(len(eps_range))
        MAPD_Psat_plot = np.zeros(len(eps_range))
                
        for ieps, eps_scaled in enumerate(eps_range):
            
            self.solve_VLE(self.Temp_VLE, eps_scaled)
            RMS_rhol_plot[ieps] = compute_RMS(self.rholiq,rhol_RP) 
            AD_rhol_plot[ieps] = compute_AD(self.rholiq,rhol_RP)
            MAPD_rhol_plot[ieps] = compute_MAPD(self.rholiq,rhol_RP)
            
            RMS_rhov_plot[ieps] = compute_RMS(self.rhovap,rhov_RP)        
            AD_rhov_plot[ieps] = compute_AD(self.rhovap,rhov_RP)
            MAPD_rhov_plot[ieps] = compute_MAPD(self.rhovap,rhov_RP)
            
            RMS_logPsat_plot[ieps] = compute_RMS(np.log10(self.Psat),np.log10(Psat_RP))        
            AD_Psat_plot[ieps] = compute_AD(self.Psat,Psat_RP)
            MAPD_Psat_plot[ieps] = compute_MAPD(self.Psat,Psat_RP)
    
### Wrong, used AD instead of MAPD        
#        Score_plot = convert_AD2Score(AD_rhol_plot,AD_rhov_plot,AD_Psat_plot)
#        
#        AD_rhol_Potoff = compute_AD(rhol_Potoff,rhol_RP)
#        AD_rhov_Potoff = compute_AD(rhov_Potoff,rhov_RP)
#        AD_Psat_Potoff = compute_AD(Psat_Potoff,Psat_RP)
#        
#        Score_Potoff = convert_AD2Score(AD_rhol_Potoff,AD_rhov_Potoff,AD_Psat_Potoff)
#        
#        AD_rhol_shifted = AD_rhol_plot+AD_rhol_Potoff-AD_rhol_plot[eps_range==1]
#        AD_rhov_shifted = AD_rhov_plot+AD_rhov_Potoff-AD_rhov_plot[eps_range==1]
#        AD_Psat_shifted = AD_Psat_plot+AD_Psat_Potoff-AD_Psat_plot[eps_range==1]
#        
#        Score_shifted = Score_plot+Score_Potoff-Score_plot[eps_range==1]
#        
#        linfit_AD_rhol = np.polyfit(eps_range,AD_rhol_shifted,1)
#        linfit_AD_rhov = np.polyfit(eps_range,AD_rhov_shifted,1)
#        linfit_AD_Psat = np.polyfit(eps_range,AD_Psat_shifted,1)
#        
#        AD_rhol_hat = lambda epsilon: np.polyval(linfit_AD_rhol,epsilon)
#        AD_rhov_hat = lambda epsilon: np.polyval(linfit_AD_rhov,epsilon)
#        AD_Psat_hat = lambda epsilon: np.polyval(linfit_AD_Psat,epsilon)
#        
#        Score_hat = lambda epsilon: convert_AD2Score(AD_rhol_hat(epsilon),AD_rhov_hat(epsilon),AD_Psat_hat(epsilon))
#        
#        eps_refined = np.linspace(0.98,1.02,num=4000)
#        
#        Score_refined = Score_hat(eps_refined)
#        AD_rhol_refined = AD_rhol_hat(eps_refined)
#        AD_rhov_refined = AD_rhov_hat(eps_refined)
#        AD_Psat_refined = AD_Psat_hat(eps_refined)
        
        Score_plot = convert_MAPD2Score(MAPD_rhol_plot,MAPD_rhov_plot,MAPD_Psat_plot)
        
        MAPD_rhol_Potoff = compute_MAPD(rhol_Potoff,rhol_RP)
        MAPD_rhov_Potoff = compute_MAPD(rhov_Potoff,rhov_RP)
        MAPD_Psat_Potoff = compute_MAPD(Psat_Potoff,Psat_RP)
        
        Score_Potoff = convert_MAPD2Score(MAPD_rhol_Potoff,MAPD_rhov_Potoff,MAPD_Psat_Potoff)
        
        MAPD_rhol_shifted = MAPD_rhol_plot+MAPD_rhol_Potoff-MAPD_rhol_plot[eps_range==1]
        MAPD_rhov_shifted = MAPD_rhov_plot+MAPD_rhov_Potoff-MAPD_rhov_plot[eps_range==1]
        MAPD_Psat_shifted = MAPD_Psat_plot+MAPD_Psat_Potoff-MAPD_Psat_plot[eps_range==1]
        
        Score_shifted = Score_plot+Score_Potoff-Score_plot[eps_range==1]
        
        ### This approach can result in negative MAPD, which is problematic
#        linfit_MAPD_rhol = np.polyfit(eps_range,MAPD_rhol_shifted,2)
#        linfit_MAPD_rhov = np.polyfit(eps_range,MAPD_rhov_shifted,2)
#        linfit_MAPD_Psat = np.polyfit(eps_range,MAPD_Psat_shifted,2)
#
#        MAPD_rhol_hat = lambda epsilon: np.polyval(linfit_MAPD_rhol,epsilon)
#        MAPD_rhov_hat = lambda epsilon: np.polyval(linfit_MAPD_rhov,epsilon)
#        MAPD_Psat_hat = lambda epsilon: np.polyval(linfit_MAPD_Psat,epsilon)

        ### This approach does not provide a great fit
#        linfit_MAPD_rhol = np.polyfit(eps_range,MAPD_rhol_plot,2)
#        linfit_MAPD_rhov = np.polyfit(eps_range,MAPD_rhov_plot,2)
#        linfit_MAPD_Psat = np.polyfit(eps_range,MAPD_Psat_plot,2)
#        
#        MAPD_rhol_hat = lambda epsilon: np.polyval(linfit_MAPD_rhol,epsilon)
#        MAPD_rhov_hat = lambda epsilon: np.polyval(linfit_MAPD_rhov,epsilon)
#        MAPD_Psat_hat = lambda epsilon: np.polyval(linfit_MAPD_Psat,epsilon)

        ### By squaring MAPD, we get a better fit. By shifting by 10 we ensure that the fit is always positive
        linfit_MAPD_rhol = np.polyfit(eps_range,MAPD_rhol_plot**2.+10.,2)
        linfit_MAPD_rhov = np.polyfit(eps_range,MAPD_rhov_plot**2.+10.,2)
        linfit_MAPD_Psat = np.polyfit(eps_range,MAPD_Psat_plot**2.+10.,2)
        
        MAPD_rhol_unshifted = lambda epsilon: np.sqrt(np.polyval(linfit_MAPD_rhol,epsilon))
        MAPD_rhov_unshifted = lambda epsilon: np.sqrt(np.polyval(linfit_MAPD_rhov,epsilon))
        MAPD_Psat_unshifted = lambda epsilon: np.sqrt(np.polyval(linfit_MAPD_Psat,epsilon))
        
        MAPD_rhol_hat = lambda epsilon: MAPD_rhol_unshifted(epsilon)+MAPD_rhol_Potoff-MAPD_rhol_unshifted(1.)
        MAPD_rhov_hat = lambda epsilon: MAPD_rhov_unshifted(epsilon)+MAPD_rhov_Potoff-MAPD_rhov_unshifted(1.)
        MAPD_Psat_hat = lambda epsilon: MAPD_Psat_unshifted(epsilon)+MAPD_Psat_Potoff-MAPD_Psat_unshifted(1.)
        
        Score_hat_unshifted = lambda epsilon: convert_MAPD2Score(MAPD_rhol_unshifted(epsilon),MAPD_rhov_unshifted(epsilon),MAPD_Psat_unshifted(epsilon))
        
        Score_hat = lambda epsilon: Score_hat_unshifted(epsilon) + Score_Potoff - Score_hat_unshifted(1.)
        
        eps_refined = np.linspace(eps_range[0],eps_range[-1],num=4000)
        
        Score_refined = Score_hat(eps_refined)
        MAPD_rhol_refined = MAPD_rhol_hat(eps_refined)+MAPD_rhol_Potoff-MAPD_rhol_hat(1.)
        MAPD_rhov_refined = MAPD_rhov_hat(eps_refined)+MAPD_rhov_Potoff-MAPD_rhov_hat(1.)
        MAPD_Psat_refined = MAPD_Psat_hat(eps_refined)+MAPD_Psat_Potoff-MAPD_Psat_hat(1.)
        
        eps_opt = eps_refined[np.argmin(Score_refined)]
        Score_opt = np.min(Score_refined)
        
        self.Score_Potoff,self.Score_plot, self.AD_rhol_plot, self.AD_rhov_plot, self.AD_Psat_plot = Score_Potoff,Score_plot, AD_rhol_plot, AD_rhov_plot, AD_Psat_plot
        self.eps_opt, self.Score_opt = eps_opt,Score_opt
#        self.Score_refined, self.eps_refined, self.AD_rhol_refined, self.AD_rhov_refined, self.AD_Psat_refined = Score_refined,eps_refined, AD_rhol_refined, AD_rhov_refined, AD_Psat_refined
        self.Score_refined, self.eps_refined, self.MAPD_rhol_refined, self.MAPD_rhov_refined, self.MAPD_Psat_refined = Score_refined,eps_refined, MAPD_rhol_refined, MAPD_rhov_refined, MAPD_Psat_refined

                    
#            print(AD_rhol_plot[ieps],AD_rhov_plot[ieps])
            
        plt.figure(figsize=(8,8))
        
        plt.plot(eps_range,RMS_rhol_plot,'r-',label=r'$\rho_{\rm liq}$')
        plt.plot(eps_range,RMS_rhov_plot,'b--',label=r'$\rho_{\rm vap}$')
        plt.plot(eps_range,RMS_logPsat_plot,'g:',label=r'$\log_{\rm 10}(P^{\rm sat}_{\rm vap}/(\rm bar))$')
        plt.plot(1,compute_RMS(rhol_Potoff,rhol_RP),'rs',label=r'$\rho_{\rm liq}$, Potoff')
        plt.plot(1,compute_RMS(rhov_Potoff,rhov_RP),'bo',label=r'$\rho_{\rm vap}$, Potoff')
        plt.plot(1,compute_RMS(Psat_Potoff,Psat_RP),'g^',label=r'$\log_{\rm 10}(P^{\rm sat}_{\rm vap}/(\rm bar))$, Potoff')
        plt.xlabel(r'$\epsilon / \epsilon_{\rm Potoff}$')
        plt.ylabel(r'Root-mean-square')
        plt.title(compound)
                    
        plt.legend()
#        plt.savefig('figures/'+compound+'_RMS_eps_scan.pdf')
        plt.show()

#        plt.figure(figsize=(8,8))
##        
##        plt.plot(eps_range,AD_rhol_plot,'r-',label=r'$\rho_{\rm liq}$')
##        plt.plot(eps_range,AD_rhov_plot,'b--',label=r'$\rho_{\rm vap}$')
##        plt.plot(eps_range,AD_Psat_plot,'g:',label=r'$P^{\rm sat}_{\rm vap}$')
##        plt.plot(eps_range,AD_rhol_shifted,'ro',label=r'$\rho_{\rm liq}$')
##        plt.plot(eps_range,AD_rhov_shifted,'bs',label=r'$\rho_{\rm vap}$')
##        plt.plot(eps_range,AD_Psat_shifted,'gd',label=r'$P^{\rm sat}_{\rm vap}$')
#        plt.plot(eps_refined,AD_rhol_refined,'r-',label=r'$\rho^{\rm sat}_{\rm liq}$')
#        plt.plot(eps_refined,AD_rhov_refined,'b--',label=r'$\rho^{\rm sat}_{\rm vap}$')
#        plt.plot(eps_refined,AD_Psat_refined,'g:',label=r'$P^{\rm sat}_{\rm vap}$')
##        plt.plot(1,AD_rhol_Potoff,'rs',label=r'$\rho_{\rm liq}$, Potoff')
##        plt.plot(1,AD_rhov_Potoff,'bo',label=r'$\rho_{\rm vap}$, Potoff')
##        plt.plot(1,AD_Psat_Potoff,'g^',label=r'$P^{\rm sat}_{\rm vap}$, Potoff')
#        plt.plot([eps_opt,eps_opt],[np.min([np.min(AD_rhov_shifted),np.min(AD_Psat_shifted),np.min(AD_rhol_shifted)]),np.max([np.max(AD_rhov_shifted),np.max(AD_Psat_shifted),np.max(AD_rhol_shifted)])],'k-.',label=r'$\min(S)$')
#        plt.plot([eps_range[0],eps_range[-1]],[0,0],'k-')
#        plt.xlabel(r'$\epsilon / \epsilon_{\rm Potoff}$')
#        plt.ylabel(r'Average percent deviation')
#        plt.title(compound)
#            
#        plt.legend()
##        plt.savefig('figures/'+compound+'_AD_eps_scan.pdf')
#        plt.show()
                
        plt.figure(figsize=(8,8))
       
#        plt.plot(eps_range,MAPD_rhol_plot,'r-',label=r'$\rho_{\rm liq}$')
#        plt.plot(eps_range,MAPD_rhov_plot,'b--',label=r'$\rho_{\rm vap}$')
#        plt.plot(eps_range,MAPD_Psat_plot,'g:',label=r'$P^{\rm sat}_{\rm vap}$')
        plt.plot(eps_range,MAPD_rhol_shifted,'ro',label=r'$\rho_{\rm liq}$')
        plt.plot(eps_range,MAPD_rhov_shifted,'bs',label=r'$\rho_{\rm vap}$')
        plt.plot(eps_range,MAPD_Psat_shifted,'gd',label=r'$P^{\rm sat}_{\rm vap}$')
        plt.plot(eps_refined,MAPD_rhol_refined,'r-',label=r'$\rho^{\rm sat}_{\rm liq}$')
        plt.plot(eps_refined,MAPD_rhov_refined,'b--',label=r'$\rho^{\rm sat}_{\rm vap}$')
        plt.plot(eps_refined,MAPD_Psat_refined,'g:',label=r'$P^{\rm sat}_{\rm vap}$')
        plt.plot(1,MAPD_rhol_Potoff,'rs',label=r'$\rho_{\rm liq}$, Potoff')
        plt.plot(1,MAPD_rhov_Potoff,'bo',label=r'$\rho_{\rm vap}$, Potoff')
        plt.plot(1,MAPD_Psat_Potoff,'g^',label=r'$P^{\rm sat}_{\rm vap}$, Potoff')
        plt.plot([eps_opt,eps_opt],[np.min([np.min(MAPD_rhov_shifted),np.min(MAPD_Psat_shifted),np.min(MAPD_rhol_shifted)]),np.max([np.max(MAPD_rhov_shifted),np.max(MAPD_Psat_shifted),np.max(MAPD_rhol_shifted)])],'k-.',label=r'$\min(S)$')
        plt.plot([eps_range[0],eps_range[-1]],[0,0],'k-')
        plt.xlabel(r'$\epsilon / \epsilon_{\rm Potoff}$')
        plt.ylabel(r'Mean absolute percent deviation')
        plt.title(compound)
            
        plt.legend()
#        plt.savefig('figures/'+compound+'_MAPD_eps_scan.pdf')
        plt.show()
        
        plt.figure(figsize=(8,8))  
        plt.plot(eps_range,Score_plot,'k-',label='Unshifted') 
        plt.plot(eps_range,Score_shifted,'r-',label='Shifted')
        plt.plot(eps_refined,Score_refined,'b-',label='Smoothed')
#        plt.plot(1,Score_Potoff,'rs',label=r'Potoff')
        plt.plot(eps_opt,Score_opt,'b*',markersize=10,label='Optimal')
        plt.xlabel(r'$\epsilon / \epsilon_{\rm Potoff}$')
        plt.ylabel(r'Score')
        plt.title(compound)
            
#        plt.legend()
#        plt.savefig('figures/'+compound+'_Score_eps_scan.pdf')
        plt.show()
        
def main():
       
    filepaths = []
            
#    root_path = 'hexane_Potoff/'
#    hist_num=['1','2','3','4','5','6','7','8','9']
    
#    root_path = 'hexane_Potoff_replicates/'
    root_path = 'hexane_Potoff_replicates_2/'
#    root_path = 'hexane_eps_scaled/'
    Temp_range = ['510','470','430','480','450','420','390','360','330']
    hist_num=['2','2','2','2','2','2','2','2','2']
    
    for iT, Temp in enumerate(Temp_range):
        
        hist_name='/his'+hist_num[iT]+'a.dat' #Only if loading hexane_Potoff
        
        filepaths.append(root_path+Temp+hist_name)

    Mw_hexane  = 12.0109*6.+1.0079*(2.*6.+2.) #[gm/mol]
    
    Temp_VLE_plot = Tsat_Potoff
#    Temp_VLE_plot = np.array([360., 350.])
    MBAR_GCMC_trial = MBAR_GCMC(root_path,filepaths,Mw_hexane,compare_literature=True)
#    MBAR_GCMC_trial.plot_histograms()
#    MBAR_GCMC_trial.plot_2dhistograms()
#    MBAR_GCMC_trial.solve_VLE(Temp_VLE_plot)
#    MBAR_GCMC_trial.plot_VLE()
#    MBAR_GCMC_trial.mu_scan(Temp_VLE_plot)
    MBAR_GCMC_trial.eps_scan(Temp_VLE_plot)
    
if __name__ == '__main__':
    '''
    python MBAR_GCMC_class.py  
    '''

    main()   