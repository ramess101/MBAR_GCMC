# -*- coding: utf-8 -*-
"""
Class for basis function generation

@author: ram9
"""

from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt

# Physical constants
N_A = 6.02214086e23 #[/mol]
nm3_to_ml = 10**21
nm3_to_m3 = 10**27
bar_nm3_to_kJ_per_mole = 0.0602214086
R_g = 8.3144598 / 1000. #[kJ/mol/K]
kJm3tobar = 1./100.

### Normally I will import this function. But I don't have it locally
def convert_eps_sig_C6_Clam(eps,sig,lam,n=6.,print_Cit=True):
    Ncoef = lam/(lam-n)*(lam/n)**(n/(lam-n))
    C6 = Ncoef * eps * sig ** n
    Clam = Ncoef * eps * sig ** lam
    
    if print_Cit:
    
        f = open('C6_it','w')
        f.write(str(C6))
        f.close()
        
        f = open('Clam_it','w')
        f.write(str(Clam))
        f.close()
        
    else:
        
        return C6, Clam


class basis_function():
    def __init__(self,iRef,iRefs,N_basis,N_frames,debug_mode=True):#Temp_sim,mu_sim,iRef,iRefs,eps_low,eps_high,sig_low,sig_high,lam_low,lam_high,rerun_flag=False,debug_mode=True):

        self.debug_mode = debug_mode
    #        self.Temp_sim = Temp_sim
#        self.rho_sim = rho_sim
        self.iRef = iRef
        self.iRefs = iRefs
        try:
            self.N_Refs = len(iRefs)
        except:
            self.iRefs = np.array([iRefs])
            self.N_Refs = len(self.iRefs)
#        
#        self.site_types = ['CH3','CH2','CH3CH2']
#        
#        # The number of elements must be equal to the number of site types
#        self.eps_high = eps_high
#        self.eps_low = eps_low
#        self.sig_high = sig_high
#        self.sig_low = sig_low
#        self.lam_high = lam_high
#        self.lam_low = lam_low
#        self.rerun_flag = rerun_flag
#        
#        self.eps_sig_lam_refs = self.compile_refs()
#        self.eps_sig_lam_ref = self.eps_sig_lam_refs[0]
#        #self.eps_ref = self.eps_sig_lam_ref[0]
#        #self.sig_ref = self.eps_sig_lam_ref[1]
#        self.lam_ref = self.eps_sig_lam_ref[2]
#        
#        self.eps_sig_lam_basis, self.eps_basis, self.sig_basis, self.lam_basis = self.create_eps_sig_lam_basis()
#        self.Cmatrix, self.lam_index = self.create_Cmatrix()
#        self.rerun_basis_functions()
#        self.generate_basis_functions_site_types()        
#        self.calc_refs_basis()
#        self.validate_refs()
#        print('Basis functions were validated for iRef= '+str(iRef))                 
#        
        self.N_basis = N_basis
        self.N_sims = self.N_basis + self.N_Refs
        self.N_frames = N_frames
        self.site_types = ['CH3','CH2','CH3CH2']
        self.build_eps_sig_lam_basis()
        self.compile_UN_values()
        self.build_Cmatrix_lam_index()
        self.build_Cmatrix_ref()
        self.generate_basis_functions()
        self.load_basis_functions()
                            
    def build_eps_sig_lam_basis(self):

        '''
        This function needs to be more general
        Also need a function for building eps_sig_lam for any set of epsilon, sigma (lambda should always be fixed)
        '''
        
        debug_mode = self.debug_mode
        lam_constant = [16]*(self.N_sims)
        
### Need to automate this
### Tried this previously

#for site in site_types:
#    
#    try:
#    
#        Cmatrix_basis[site] = eps_basis[site]*sig_basis[site]

                       
        eps_basis = {'CH3':np.array([121.25,121.25,121.25,0.,0.,121.25,121.25]),'CH2':np.array([61,0.,0.,61.,61.,61.,61.]),'CH3CH2':[]}
        eps_basis['CH3CH2']=np.sqrt(eps_basis['CH3']*eps_basis['CH2'])
        sig_basis = {'CH3':np.array([0.3783,0.377,0.379,0.3783,0.3783,0.377,0.379]),'CH2':np.array([0.399,0.399,0.399,0.398,0.400,0.398,0.400]),'CH3CH2':[]}
        sig_basis['CH3CH2'] = (sig_basis['CH3']+sig_basis['CH2'])/2.
        lam_basis = {'CH3':np.array(lam_constant),'CH2':np.array(lam_constant),'CH3CH2':[]}
        lam_basis['CH3CH2'] = (lam_basis['CH3']+lam_basis['CH2'])/2.

        if debug_mode:
            print(eps_basis,sig_basis,lam_basis)
                 
        self.eps_basis, self.sig_basis, self.lam_basis = eps_basis, sig_basis, lam_basis

    def compile_UN_values(self):
        
        '''
        Compiles the number of molecules (N) and energies (U) from N_frames
        Output:
            U_basis: N_frames x N_basis (rows x columns) matrix of energy rerun values
            Nmol: N_frames array of number of molecules in each frame
        '''
         
        N_basis, N_frames, debug_mode = self.N_basis, self.N_frames, self.debug_mode
        
        U_basis = {}
        Nmol = np.array([])
        
        for istage in np.arange(1,N_frames):
            
            if debug_mode: print(istage)
            
            for ibasis in range(N_basis+1):
                if debug_mode: print('ibasis='+str(ibasis))
                
                NU_ibasis = np.loadtxt('his_rr'+str(istage)+'_basis_function_'+str(ibasis),skiprows=1)
                try:
                    U_basis[ibasis] = np.append(U_basis[ibasis],NU_ibasis[:,1])
                except:
                    U_basis[ibasis] = NU_ibasis[:,1]
            
            Nmol = np.append(Nmol,NU_ibasis[:,0])
         
        if debug_mode: print(Nmol)
        
        self.U_basis, self.Nmol = U_basis, Nmol
           
    def build_Cmatrix_lam_index(self):
            
        '''
        This function creates the matrix of C6, C12, C13... where C6 is always the
        zeroth column and the rest is the range of lambda from lam_low to lam_high.
        These values were predefined in rerun_basis_functions
        '''
        
        site_types, lam_basis, eps_basis, sig_basis, site_types, N_basis, debug_mode = self.site_types, self.lam_basis, self.eps_basis, self.sig_basis, self.site_types, self.N_basis, self.debug_mode
    
        Cmatrix = {'CH3':[],'CH2':[],'CH3CH2':[]}
        lam_index = {'CH3':[],'CH2':[],'CH3CH2':[]}
        
        for isite, site_type in enumerate(site_types):
        
            Cmatrix[site_type] = np.zeros([len(lam_basis[site_type]),len(lam_basis[site_type])])
            
            C6_basis, Clam_basis = convert_eps_sig_C6_Clam(eps_basis[site_type],sig_basis[site_type],lam_basis[site_type],print_Cit=False)
            
            lam_index[site_type] = lam_basis[site_type].copy()
            lam_index[site_type][0] = 6
        
            Cmatrix[site_type][:,0] = C6_basis
                  
            for ilam, lam in enumerate(lam_index[site_type]):
                for iBasis, lam_rerun in enumerate(lam_basis[site_type]):
                    if lam == lam_rerun:
                        Cmatrix[site_type][iBasis,ilam] = Clam_basis[iBasis]
    
        Cmatrix_basis = np.zeros([N_basis,N_basis])
        
        for isite, site_type in enumerate(site_types):
            for ilam, lam in enumerate(np.array([6,16])):
                for ibasis in range(N_basis):
                    Cmatrix_basis[ibasis,2*isite+ilam] = Cmatrix[site_type][ibasis+1][ilam]

        if debug_mode: print(Cmatrix_basis.shape)

        self.Cmatrix, self.Cmatrix_basis = Cmatrix, Cmatrix_basis

    def build_Cmatrix_ref(self):
        '''
        Builds the Cmatrix for the reference(s) (not multiple yet)
        '''
            
        site_types, Cmatrix, N_basis, iRef = self.site_types, self.Cmatrix, self.N_basis, self.iRef
        
        Cmatrix_ref = np.zeros([N_basis])
        
        for isite, site_type in enumerate(site_types):
            for ilam, lam in enumerate(np.array([6,16])):
                Cmatrix_ref[2*isite+ilam] = Cmatrix[site_type][iRef][ilam]
        
        self.Cmatrix_ref = Cmatrix_ref
        
    def generate_basis_functions(self):

        '''
        Input:
            U_basis: energies from rerun files
        Output:
            sumr6lam_all: printed to 'basis functions' file
        '''
        
        U_basis, Cmatrix_basis,N_basis, Cmatrix_ref = self.U_basis, self.Cmatrix_basis, self.N_basis, self.Cmatrix_ref
        
        f = open('basis_functions','w')
        f.write('r6_CH3'+'\t'+'rlam_CH3'+'\t'+'r6_CH2'+'\t'+'rlam_CH2'+'\t'+'r6_CH3CH2'+'\t'+'rlam_CH3CH2'+'\n')
        
        for iframe in range(len(U_basis[0])):
            
            U_basis_frame = []
            
            for ibasis in range(N_basis):
                
                U_basis_frame.append(U_basis[ibasis+1][iframe])
                
            U_basis_frame = np.array(U_basis_frame)
            
            sumr6lam = np.linalg.solve(Cmatrix_basis,U_basis_frame)
            
            U_basis_ref_frame = np.linalg.multi_dot([Cmatrix_ref,sumr6lam])
            
            for isum in sumr6lam:
                f.write(repr(isum)+'\t') #Using str(isum) which rounds the value leads to an error that is a few orders of magnitude greater, but still acceptable
            f.write('\n')       
        
            if np.abs(U_basis_ref_frame - U_basis[0][iframe]) > 1:
        
                print('Basis function estimate: '+str(U_basis_ref_frame))
                print('Actual energy: '+str(U_basis[0][iframe]))
                
        f.close()
        
    def load_basis_functions(self):
        '''
        Load the sumr6lam_all basis functions from file
        '''
        
        debug_mode = self.debug_mode
        
        sumr6lam_all = np.loadtxt('basis_functions',skiprows=1)
        
        if debug_mode: print(sumr6lam_all.shape)
        
        self.sumr6lam_all = sumr6lam_all
        
    def compute_U_theta(self,Cmatrix_theta=[]):
        '''
        Computes the energy for a given parameter set. Still need to allow
        for arbitrary Cmatrix_theta
        
        Input:
            Cmatrix_theta: Cmatrix for an arbitrary theta parameter set
        Output:
            U_theta: Computed energies for parameter set theta
        '''
        
        if Cmatrix_theta == []:
            print('No theta provided')
            return
        
        sumr6lam_all = self.sumr6lam_all
        
        U_theta = np.linalg.multi_dot([Cmatrix_theta,sumr6lam_all.T])

        return U_theta

    def parity_plot(self):
        '''
        Plots the basis function values compared with the direct reference simulation values
        '''
        
        U_basis, Nmol = self.U_basis, self.Nmol
        
        U_basis_ref = self.compute_U_theta(self.Cmatrix_ref)
        
        plt.plot(U_basis[0],U_basis_ref,'k+',mfc='None',markersize=5)
        plt.plot([np.min(U_basis_ref),np.max(U_basis_ref)],[np.min(U_basis_ref),np.max(U_basis_ref)],'r-')
        plt.plot([np.min(U_basis_ref),np.max(U_basis_ref)],[1.05*np.min(U_basis_ref),1.05*np.max(U_basis_ref)],'r--')
        plt.plot([np.min(U_basis_ref),np.max(U_basis_ref)],[0.95*np.min(U_basis_ref),0.95*np.max(U_basis_ref)],'r--')
        plt.xlabel('Direct simulation energy (K)')
        plt.ylabel('Basis function energy (K)')
        plt.show()
            
        plt.plot(U_basis[0],U_basis_ref,'k+',mfc='None',markersize=5)
        plt.plot([np.min(U_basis_ref),np.max(U_basis_ref)],[np.min(U_basis_ref),np.max(U_basis_ref)],'r-')
        plt.plot([np.min(U_basis_ref),np.max(U_basis_ref)],[1.05*np.min(U_basis_ref),1.05*np.max(U_basis_ref)],'r--')
        plt.plot([np.min(U_basis_ref),np.max(U_basis_ref)],[0.95*np.min(U_basis_ref),0.95*np.max(U_basis_ref)],'r--')
        plt.xlabel('Direct simulation energy (K)')
        plt.ylabel('Basis function energy (K)')
        plt.xlim([None,-50000])
        plt.ylim([None,-50000])
        plt.show()
        
        plt.plot(U_basis[0][Nmol<20],U_basis_ref[Nmol<20],'k+',mfc='None',markersize=5)
        plt.plot([np.min(U_basis_ref),np.max(U_basis_ref)],[np.min(U_basis_ref),np.max(U_basis_ref)],'r-')
        plt.plot([np.min(U_basis_ref),np.max(U_basis_ref)],[1.05*np.min(U_basis_ref),1.05*np.max(U_basis_ref)],'r--')
        plt.plot([np.min(U_basis_ref),np.max(U_basis_ref)],[0.95*np.min(U_basis_ref),0.95*np.max(U_basis_ref)],'r--')
        plt.xlabel('Direct simulation energy (K)')
        plt.ylabel('Basis function energy (K)')
        plt.xlim([-10000,-1000])
        plt.ylim([-10000,-1000])
        plt.show()
        
        plt.plot(U_basis[0]/Nmol,U_basis_ref/Nmol,'k+',mfc='None',markersize=5)
        plt.plot([np.min(U_basis_ref/Nmol),np.max(U_basis_ref/Nmol)],[np.min(U_basis_ref/Nmol),np.max(U_basis_ref/Nmol)],'r-')
        plt.plot([np.min(U_basis_ref/Nmol),np.max(U_basis_ref/Nmol)],[1.05*np.min(U_basis_ref/Nmol),1.05*np.max(U_basis_ref/Nmol)],'r--')
        plt.plot([np.min(U_basis_ref/Nmol),np.max(U_basis_ref/Nmol)],[0.95*np.min(U_basis_ref/Nmol),0.95*np.max(U_basis_ref/Nmol)],'r--')
        plt.xlabel('Direct simulation energy (K/molecule)')
        plt.ylabel('Basis function energy (K/molecule)')
        #plt.xlim([None,-50000])
        #plt.ylim([None,-50000])
        plt.show()
        
        print(np.mean((U_basis_ref-U_basis[0])/U_basis[0]*100.))
        print(np.mean((U_basis_ref[Nmol<30]-U_basis[0][Nmol<30])/U_basis[0][Nmol<30]*100.))



#    def compile_refs(self):
#        
#        iRef,nRefs,iRefs = self.iRef,self.nRefs,self.iRefs
#     
#        eps_sig_lam_refs = np.zeros([nRefs,9]) # Changed this from 3 to 9, need better way to track for other compounds
#        
#        for iiiRef, iiRef in enumerate(iRefs):
#            
#            eps_sig_lam_refs[iiiRef,:] = np.loadtxt('../ref'+str(iiRef)+'/eps_sig_lam_ref')
#            
#        return eps_sig_lam_refs
#        
#    def create_eps_sig_lam_basis(self):
#        # Removed eps_ref and sig_ref to avoid ambiguity with more than 1 site type
#        site_types,lam_ref,eps_low,eps_high,sig_low,sig_high,lam_low,lam_high = self.site_types,self.lam_ref,self.eps_low,self.eps_high,self.sig_low,self.sig_high,self.lam_low,self.lam_high            
#    
#    # I am going to keep it the way I had it where I just submit with real parameters 
#    # And solve linear system of equations. 
#    #    print(lam_low)
#    #    print(lam_high)
#    
#        eps_basis = {'CH3':[],'CH2':[],'CH3CH2':[]}
#        sig_basis = {'CH3':[],'CH2':[],'CH3CH2':[]}
#        lam_basis = {'CH3':[],'CH2':[],'CH3CH2':[]}
#        eps_sig_lam_basis = {'CH3':[],'CH2':[],'CH3CH2':[]}
#    
#        for isite, site_type in enumerate(site_types):
#            
#            if int(lam_low[isite]) == 12 and int(lam_high[isite]) == 12:
#            
#                nBasis = len(range(int(lam_low[isite]),int(lam_high[isite])+1))+2 #The 2 is necessary if only using LJ 12-6
#                #The LJ 12-6 case still needs to be worked out
#            else:
#                
#                nBasis = len(range(int(lam_low[isite]),int(lam_high[isite])+1))+1 #The 2 is necessary if only using LJ 12-6
#            
#            eps_basis[site_type] = np.ones(nBasis)*eps_low[isite]
#            sig_basis[site_type] = np.ones(nBasis)*sig_low[isite] 
#            lam_basis[site_type] = np.ones(nBasis)*lam_ref # Should be 12 to start
#            
#            eps_basis[site_type][0] = eps_high[isite]
#            sig_basis[site_type][0] = sig_high[isite]
#    
#            lam_basis[site_type][1:] = range(int(lam_low[isite]),int(lam_high[isite])+1) 
#            
#            eps_sig_lam_basis[site_type] = np.array([eps_basis,sig_basis,lam_basis]).transpose()
#        
#        print(eps_basis)
#        print(sig_basis)
#        print(lam_basis)
#    
#        eps_sig_lam_basis = np.array([eps_basis,sig_basis,lam_basis]).transpose()
#        
#        self.nBasis = nBasis
#        
#        return eps_sig_lam_basis, eps_basis, sig_basis, lam_basis
#
#    def create_Cmatrix(self):
#        '''
#        This function creates the matrix of C6, C12, C13... where C6 is always the
#        zeroth column and the rest is the range of lambda from lam_low to lam_high.
#        These values were predefined in rerun_basis_functions
#        '''
#        
#        site_types,iRef,eps_basis,sig_basis,lam_basis = self.site_types,self.iRef,self.eps_basis.copy(),self.sig_basis.copy(),self.lam_basis.copy()
#        
#        Cmatrix = {'CH3':[],'CH2':[],'CH3CH2':[]}
#        lam_index = {'CH3':[],'CH2':[],'CH3CH2':[]}
#        
#        for isite, site_type in enumerate(site_types):
#        
#            Cmatrix[site_type] = np.zeros([len(lam_basis[site_type]),len(lam_basis[site_type])])
#            
#            C6_basis, Clam_basis = convert_eps_sig_C6_Clam(eps_basis[site_type],sig_basis[site_type],lam_basis[site_type],print_Cit=False)
#            
#            lam_index[site_type] = lam_basis[site_type].copy()
#            lam_index[site_type][0] = 6
#        
#            Cmatrix[site_type][:,0] = C6_basis
#                  
#            for ilam, lam in enumerate(lam_index[site_type]):
#                for iBasis, lam_rerun in enumerate(lam_basis[site_type]):
#                    if lam == lam_rerun:
#                        Cmatrix[site_type][iBasis,ilam] = Clam_basis[iBasis]
#                        
##        fpath = "../ref"+str(iRef)+"/"
##        
##        f = open(fpath+'Cmatrix','w')
##        g = open(fpath+'lam_index','w')
##        
##        for ilam, lam in enumerate(lam_index):
##            for jlam in range(len(lam_basis)):
##                f.write(str(Cmatrix[ilam,jlam])+'\t')
##            f.write('\n')
##            g.write(str(lam)+'\t')              
##        
##        f.close()
##        g.close()       
#             
##        print(Cmatrix)
##        print(lam_index)
#
#        return Cmatrix, lam_index 
#    
#    def check_basis_functions_U(self,LJsr,sumr6lam,Cmatrix):
#        LJhat = np.linalg.multi_dot([Cmatrix,sumr6lam])
#        assert (np.abs(LJsr - LJhat) < 1e-3).all(), 'Basis functions for internal energy deviate by at most:'+str(np.max(np.abs(LJsr-LJhat)))
#        
#    def check_basis_functions_Vir(self,Vir_0,Vir_1,Vir_2,sumrdr6lam_vdw,sumrdr6lam_LINCS,Cmatrix):
#        Vir_vdw_hat = np.linalg.multi_dot([Cmatrix,sumrdr6lam_vdw])
#        Vir_LINCS_hat = np.linalg.multi_dot([Cmatrix,sumrdr6lam_LINCS])
#        Vir_total_hat = Vir_vdw_hat + Vir_LINCS_hat + Vir_2
#        
#        assert (np.abs(Vir_1 - Vir_vdw_hat) < 1e-3).all(), 'Basis function for vdw virial deviates at most by:'+str(np.max(np.abs(Vir_1-Vir_vdw_hat)))
# 
#        assert (np.abs(Vir_0 - Vir_1 - Vir_2 - Vir_LINCS_hat) < 1e-3).all(), 'Basis function for LINCS virial deviates at most by:'+str(np.np.max(np.abs(Vir_0 - Vir_1 - Vir_2 -Vir_LINCS_hat)))
#    
#        assert (np.abs(Vir_0 - Vir_total_hat) < 1e-3).all(), 'Basis function for virial deviates at most by:'+str(np.np.max(np.abs(Vir_0 - Vir_total_hat)))
#        
#        return Vir_vdw_hat, Vir_LINCS_hat, Vir_total_hat
#        
#    def check_basis_functions_press(self,press,p_0,p_1,p_2,Vir_vdw_hat, Vir_LINCS_hat, Vir_total_hat,KE,iState):
#        
#        press_vdw_hat = self.convert_VirialtoP(KE,Vir_vdw_hat,iState)
#        press_LINCS_hat = self.convert_VirialtoP(KE,Vir_LINCS_hat,iState)
#        press_total_hat = self.convert_VirialtoP(KE,Vir_total_hat,iState)
#        
#        press_dev = np.abs(press-press_total_hat)
#                
#        assert (press_dev < 1e-3).all(), 'Basis functions for the total pressure deviate by at most:'+str(np.max(press_dev))
#    
#    def rerun_basis_functions(self):
#        ''' 
#    This function submits the rerun simulations that are necessary to generate the
#    basis functions. It starts by performing a rerun simulation with the LJ model 
#    at the highest epsilon and sigma. It then submits a rerun using LJ with the 
#    lowest epsilon and sigma. Then, it submits a single rerun for all the different
#    Mie values for lambda. 
#        '''
#    
#        site_types,iRef,eps_basis,sig_basis,lam_basis,rerun_flag = self.site_types,self.iRef,self.eps_basis,self.sig_basis,self.lam_basis,self.rerun_flag
#        
#        iRerun = 0
#        
#        self.iBasis = range(iRerun,iRerun+self.nBasis)
#        
#        #iRerun_basis = iRerun #This approach is no longer used
#        
#        for iRerun_basis in self.iBasis:
#            
#            for site_type in site_types:
#                
#                eps_rerun = eps_basis[site_type][iRerun_basis]
#                sig_rerun = sig_basis[site_type][iRerun_basis]
#                lam_rerun = lam_basis[site_type][iRerun_basis]
#                   
#                fpathRef = "../ref"+str(iRef)+"/"
#                #print(fpathRef)
#            
#                f = open(fpathRef+'eps'+site_type+'_it','w')
#                f.write(str(eps_rerun))
#                f.close()
#            
#                f = open(fpathRef+'sig'+site_type+'_it','w')
#                f.write(str(sig_rerun))
#                f.close()
#            
#                f = open(fpathRef+'lam'+site_type+'_it','w')
#                f.write(str(lam_rerun))
#                f.close()
#                
#            f = open(fpathRef+'iRerun','w')
#            f.write(str(iRerun_basis))
#            f.close()
#            
#            if rerun_flag:
#            
#                subprocess.call(fpathRef+"C3H8RerunITIC_basis")
#        
#                #iRerun_basis += 1 #This approach is no longer used
#            
#            #print('iRerun is = '+str(iRerun)+', while iRerun_basis = '+str(iRerun_basis))
#            
#        f = open(fpathRef+'iBasis','w')
#        f.write(str(iRerun_basis-1))
#        f.close()
#        
#    def generate_basis_functions_site_types(self):
#        
#        site_types = self.site_types
#        
#        self.sumr6lam_state = {'CH3':[],'CH2':[],'CH3CH2':[]}
#        self.sumrdr6lam_vdw_state = {'CH3':[],'CH2':[],'CH3CH2':[]}
#        self.sumrdr6lam_LINCS_state = {'CH3':[],'CH2':[],'CH3CH2':[]}
#        
#        for site_type in site_types:
#            
#            self.KE_state, self.Vir_novdw_state, self.sumr6lam_state[site_type], self.sumrdr6lam_vdw_state[site_type], self.sumrdr6lam_LINCS_state[site_type], self.U_novdw_state = self.generate_basis_functions(site_type)
#    
#    def generate_basis_functions(self,site_type):
#        
#        iRef, iBasis, Cmatrix = self.iRef, self.iBasis,self.Cmatrix[site_type]
#        
#        nSets = len(iBasis)
#        
#        g_start = 28 #Row where data starts in g_energy output
#        g_t = 0 #Column for the snapshot time
#        g_LJsr = 1 #Column where the 'Lennard-Jones' short-range interactions are located
#        g_LJdc = 2 #Column where the 'Lennard-Jones' dispersion corrections are located
#        g_en = 3 #Column where the potential energy is located
#        g_KE = 4 #Column where KE is located
#        g_p = 5 #Column where p is located
#        
#        iState = 0
#        
#        for run_type in ITIC: 
#        
#            for irho in np.arange(0,nrhos[run_type]):
#        
#                for iTemp in np.arange(0,nTemps[run_type]):
#        
#                    if run_type == 'Isochore':
#        
#                        fpath = run_type+'/rho'+str(irho)+'/T'+str(iTemp)+'/NVT_eq/NVT_prod/'
#        
#                    else:
#        
#                        fpath = run_type+'/rho_'+str(irho)+'/NVT_eq/NVT_prod/'
#                    
#                    #print(fpath)  
#                    en_p = open('../ref'+str(iRef)+'/'+fpath+'energy_press_ref%srr%s.xvg' %(iRef,iRef),'r').readlines()[g_start:] #Read all lines starting at g_start for "state" k
#                        
#                    nSnaps = len(en_p)
#                    #print(N_k)
#                    #print(nSnaps)
#                    
#                    if iState == 0:
#                        KE_state = np.zeros([nStates,nSnaps])
#                        Vir_novdw_state = np.zeros([nStates,nSnaps])
#                        sumr6lam_state = np.zeros([nStates,nSets,nSnaps])
#                        sumrdr6lam_vdw_state = np.zeros([nStates,nSets,nSnaps])
#                        sumrdr6lam_LINCS_state = np.zeros([nStates,nSets,nSnaps])
#                        U_novdw_state = np.zeros([nStates,nSnaps])
#                        
#                    t = np.zeros([nSets,nSnaps])
#                    LJsr = np.zeros([nSets,nSnaps])
#                    LJdc = np.zeros([nSets,nSnaps])
#                    en = np.zeros([nSets,nSnaps])
#                    p = np.zeros([nSets,nSnaps])
#                    U_total = np.zeros([nSets,nSnaps])
#                    LJ_total = np.zeros([nSets,nSnaps])
#                    p_0 = np.zeros([nSets,nSnaps])
#                    p_1 = np.zeros([nSets,nSnaps])
#                    p_2 = np.zeros([nSets,nSnaps])
#                    Vir_0 = np.zeros([nSets,nSnaps])
#                    Vir_1 = np.zeros([nSets,nSnaps])
#                    Vir_2 = np.zeros([nSets,nSnaps])
#                    KE = np.zeros([nSets,nSnaps])
#        
#                    for iSet, enum in enumerate(iBasis): 
#
#                        en_p = open('../ref'+str(iRef)+'/'+fpath+'energy_press_ref%srr%sbasis0'%(iRef,enum)+site_type+'.xvg','r').readlines()[g_start:] #Read all lines starting at g_start for "state" k
#                      
#                        for frame in xrange(nSnaps):
#                            t[iSet][frame] = float(en_p[frame].split()[g_t])
#                            LJsr[iSet][frame] = float(en_p[frame].split()[g_LJsr])
#                            LJdc[iSet][frame] = float(en_p[frame].split()[g_LJdc])
#                            en[iSet][frame] = float(en_p[frame].split()[g_en])
#                            p[iSet][frame] = float(en_p[frame].split()[g_p])
#                            KE[iSet][frame] = float(en_p[frame].split()[g_KE])
#                            #f.write(str(p[iSet][frame])+'\n')
#    
#                        U_total[iSet] = en[iSet] # For TraPPEfs we just used potential because dispersion was erroneous. I believe we still want potential even if there are intramolecular contributions. 
#                        LJ_total[iSet] = LJsr[iSet] + LJdc[iSet] #In case we want just the LJ total (since that would be U_res as long as no LJ intra). We would still use U_total for MBAR reweighting but LJ_total would be the observable
#                    
#                    U_novdw_state[iState,:] = U_total[0,:] - LJ_total[0,:] #There are several ways to track U_novdw
#                    
#                    for iSet, enum in enumerate(iBasis):
#                            
#                        en_p_0 = open('../ref'+str(iRef)+'/'+fpath+'energy_press_ref%srr%sbasis0'%(iRef,enum)+site_type+'.xvg','r').readlines()[g_start:] #Read all lines starting at g_start for "state" k
#                        en_p_1 = open('../ref'+str(iRef)+'/'+fpath+'energy_press_ref%srr%sbasis1'%(iRef,enum)+site_type+'.xvg','r').readlines()[g_start:] #Read all lines starting at g_start for "state" k
#                        en_p_2 = open('../ref'+str(iRef)+'/'+fpath+'energy_press_ref%srr0basis2'%(iRef)+self.site_types[0]+'.xvg','r').readlines()[g_start:] #Read all lines starting at g_start for "state" k #Only create this basis2 file for the first site type
#                        
#                        for frame in xrange(nSnaps):
#                            p_0[iSet][frame] = float(en_p_0[frame].split()[g_p])
#                            p_1[iSet][frame] = float(en_p_1[frame].split()[g_p])
#                            p_2[iSet][frame] = float(en_p_2[frame].split()[g_p])      
#        
#                        Vir_0[iSet] = self.convert_PtoVirial(KE[iSet],p_0[iSet],iState)
#                        Vir_1[iSet] = self.convert_PtoVirial(KE[iSet],p_1[iSet],iState)
#                        Vir_2[iSet] = self.convert_PtoVirial(KE[iSet],p_2[iSet],iState)
#                                 
#                    sumr6lam = np.zeros([nSets,nSnaps])
#                    sumrdr6lam_vdw = np.zeros([nSets,nSnaps])
#                    sumrdr6lam_LINCS = np.zeros([nSets,nSnaps])
#                    
#                    f0 = open('../ref'+str(iRef)+'/'+fpath+'basis_functions_Usr_'+site_type,'w')
#                    f1 = open('../ref'+str(iRef)+'/'+fpath+'basis_functions_vir_vdw_'+site_type,'w')
#                    f2 = open('../ref'+str(iRef)+'/'+fpath+'basis_functions_vir_LINCS_'+site_type,'w')
#                    f3 = open('../ref'+str(iRef)+'/'+fpath+'basis_functions_KE','w')
#                    f4 = open('../ref'+str(iRef)+'/'+fpath+'basis_functions_vir_novdw','w')
#                    f5 = open('../ref'+str(iRef)+'/'+fpath+'basis_functions_U_novdw','w')
#                    
#                    for ilam, lam in enumerate(self.lam_index[site_type]): 
#                        if ilam < len(self.lam_index[site_type])-1: 
#                            delimit = '\t'
#                        else:
#                            delimit = '\n'
#                        f0.write(str(lam)+delimit)
#                        f1.write(str(lam)+delimit)
#                        f2.write(str(lam)+delimit)
#    
#                    for frame in xrange(nSnaps):
#                        U_basis_vdw = LJsr[:,frame]
#                        Vir_basis_vdw = Vir_1[:,frame]
#                        Vir_basis_LINCS = Vir_0[:,frame] - Vir_1[:,frame] - Vir_2[:,frame]
#                        sumr6lam[:,frame] = np.linalg.solve(Cmatrix,U_basis_vdw)
#                        sumrdr6lam_vdw[:,frame] = np.linalg.solve(Cmatrix,Vir_basis_vdw)
#                        sumrdr6lam_LINCS[:,frame] = np.linalg.solve(Cmatrix,Vir_basis_LINCS)
#                        assert sumr6lam[0,frame] < 0, 'The attractive contribution has the wrong sign'
#                    
#                        for iSet in range(nSets):
#                            if iSet < nSets-1:
#                                f0.write(str(sumr6lam[iSet,frame])+'\t')
#                                f1.write(str(sumrdr6lam_vdw[iSet,frame])+'\t')
#                                f2.write(str(sumrdr6lam_LINCS[iSet,frame])+'\t')
#                            else:
#                                f0.write(str(sumr6lam[iSet,frame])+'\n')
#                                f1.write(str(sumrdr6lam_vdw[iSet,frame])+'\n')
#                                f2.write(str(sumrdr6lam_LINCS[iSet,frame])+'\n')
#                                
#                        f3.write(str(KE[0,frame])+'\n')
#                        f4.write(str(Vir_2[0,frame])+'\n')
#                        f5.write(str(U_novdw_state[iState,frame])+'\n')
#                        
#                    f0.close()
#                    f1.close()
#                    f2.close()
#                    f3.close()
#                    f4.close()
#                    f5.close()
#                    
#                    KE_state[iState,:] = KE[0,:]
#                    Vir_novdw_state[iState,:] = Vir_2[0,:]
#                    sumr6lam_state[iState,:,:] = sumr6lam
#                    sumrdr6lam_vdw_state[iState,:,:] = sumrdr6lam_vdw
#                    sumrdr6lam_LINCS_state[iState,:,:] = sumrdr6lam_LINCS 
#                    
#                    self.check_basis_functions_U(LJsr,sumr6lam,Cmatrix)
#                    Vir_vdw_hat, Vir_LINCS_hat, Vir_total_hat = self.check_basis_functions_Vir(Vir_0,Vir_1,Vir_2,sumrdr6lam_vdw,sumrdr6lam_LINCS,Cmatrix)
#                    pcheck = self.convert_VirialtoP(KE,Vir_0,iState)
#                    assert (np.abs(p-pcheck)< 1e-3).all(), 'Conversion of virial to P has deviations of at most:'+str(np.max(np.abs(p-pcheck)))
#                    self.check_basis_functions_press(p,p_0,p_1,p_2,Vir_vdw_hat, Vir_LINCS_hat, Vir_total_hat,KE,iState)
#                        
#                    #self.sumr6lam[iState], self.sumrdr6lam_vdw[iState], self.sumrdr6lam_LINCS[iState] = sumr6lam, sumrdr6lam_vdw, sumrdr6lam_LINCS
#                        
#                    iState += 1   
#    
#        return KE_state, Vir_novdw_state, sumr6lam_state, sumrdr6lam_vdw_state, sumrdr6lam_LINCS_state, U_novdw_state
#                    
#    def convert_PtoVirial(self,KE,press,iState):
#        """ Calculate the virial
#        KE: kinetic energy [kJ/mol]
#        press: pressure [bar]
#        iState: ITIC state point 
#        
#        Vir: virial [kJ/mol]
#        """
#        
#        rho = self.rho_sim[iState]/Nmol_sim[iState] #[1/nm3]
#        Vol = 1./rho/nm3_to_m3 #[m3]
#        
#        Vir = KE/3. - press/2.*Vol/kJm3tobar*N_A
#        
#        return Vir
#    
#    def convert_VirialtoP(self,KE,Vir,iState):
#        """ Calculate the virial
#        KE: kinetic energy [kJ/mol]
#        Vir: virial [kJ/mol]
#        iState: ITIC state point 
#        
#        press: pressure [bar]
#        """
#        
#        rho = self.rho_sim[iState]/Nmol_sim[iState] #[1/nm3]
#        Vol = 1./rho/nm3_to_m3 #[m3]
#                
#        press = 2./Vol*(KE/3. - Vir)*kJm3tobar/N_A
#        
#        return press
#                    
#    def create_Carray(self,eps_sig_lam,site_type):
#        '''
#        This function creates a single column array of C6, C12, C13... where C6 is always the
#        zeroth column and the rest are 0 except for the column that pertains to lambda.
#        '''
#        
#        iRef, lam_index = self.iRef, self.lam_index
#        
#        eps = eps_sig_lam[0]
#        sig = eps_sig_lam[1]
#        lam = eps_sig_lam[2]
#        
#        Carray = np.zeros([len(lam_index[self.site_types[0]])])
#        
#        C6, Clam = convert_eps_sig_C6_Clam(eps,sig,lam,print_Cit=False)
#        
#        Carray[0] = C6
#               
#        for ilam, lam_basis in enumerate(lam_index[site_type]):
#            if lam == lam_basis:
#                Carray[ilam] = Clam
#                    
##        fpath = "../ref"+str(iRef)+"/"
##        
##        f = open(fpath+'Carrayit','a')
##        
##        for ilam, Ci in enumerate(Carray):
##            f.write(str(Ci)+'\t')
##        
##        f.write('\n')              
##        f.close()
#               
#        return Carray
#    
#    def LJ_tail_corr(self,C6,rho,Nmol,site_type):
#        '''
#        Calculate the LJ tail correction to U using the Gromacs approach (i.e. only C6 is used)
#        '''
#        U_Corr = -2./3. * np.pi * C6 * rc**(-3.)
#        U_Corr *= Nmol * rho * N_inter[site_type]
#        return U_Corr
#                    
#    def UP_basis_functions_site(self,iState,eps_sig_lam,site_type):
#        '''
#        iState: state point (integer)
#        eps_sig_lam: array of epsilon [K], sigma [nm], and lambda
#        
#        LJ_SR: short-range Lennard-Jones energy [kJ/mol]
#        LJ_dc: dispersive correction LJ [kJ/mol]
#        press: pressure [bar]
#        '''
#        
#        rho_state = rho_sim[iState]
#        Nstate = Nmol_sim[iState]        
#        
#        sumr6lam = self.sumr6lam_state[site_type][iState,:,:].transpose()
#        sumrdr6lam_vdw = self.sumrdr6lam_vdw_state[site_type][iState,:,:].transpose()
#        sumrdr6lam_LINCS = self.sumrdr6lam_LINCS_state[site_type][iState,:,:].transpose()
##        KE = self.KE_state[iState,:]
##        Vir_novdw = self.Vir_novdw_state[iState,:]
#        #if iState == 9: print(site_type,sumr6lam[0,0])
#        Carray = self.create_Carray(eps_sig_lam[site_type],site_type)
#        C6 = Carray[0]
#        
#        LJ_SR = np.linalg.multi_dot([sumr6lam,Carray])
#        LJ_dc = np.ones(len(LJ_SR))*self.LJ_tail_corr(C6,rho_state,Nstate,site_type) # Use Carray[0] to convert C6 into LJ_dc
#        
#        Vir_vdw = np.linalg.multi_dot([sumrdr6lam_vdw,Carray])
#        Vir_LINCS = np.linalg.multi_dot([sumrdr6lam_LINCS,Carray])
##        Vir_total = Vir_vdw + Vir_LINCS + Vir_novdw
#        
##        press = self.convert_VirialtoP(KE,Vir_total,iState)
#        #print('The initial LJ energy is '+str(LJ_SR[0]))
##        return LJ_SR, LJ_dc, press
#        return LJ_SR, LJ_dc, Vir_vdw, Vir_LINCS
#    
#    def UP_basis_functions(self,iState,eps_sig_lam):
#        
#        site_types = self.site_types
#        
#        KE = self.KE_state[iState,:]
#        Vir_novdw = self.Vir_novdw_state[iState,:]
#        
##        LJ_SR = {'CH3':[],'CH2':[],'CH3CH2':[]}
##        Vir_vdw_LINCS = {'CH3':[],'CH2':[],'CH3CH2':[]}
#        
#        LJ_total = np.zeros(len(KE))
#        Vir_total = Vir_novdw.copy()
#        
#        eps_sig_lam_sites = {'CH3':eps_sig_lam[0:3],'CH2':eps_sig_lam[3:6],'CH3CH2':eps_sig_lam[6:9]}
#        
#        for site_type in site_types:
#        
#            LJ_SR, LJ_dc, Vir_vdw, Vir_LINCS = self.UP_basis_functions_site(iState,eps_sig_lam_sites,site_type)
#            
#            LJ_total += LJ_SR + LJ_dc
#            
#            Vir_total += Vir_vdw + Vir_LINCS
#        
#        press = self.convert_VirialtoP(KE,Vir_total,iState)
#        
#        return LJ_total, press
#    
#    def UP_basis_states(self,eps_sig_lam):
#        
#        for iState in range(nStates):
#            LJ_total_basis, press_basis = self.UP_basis_functions(iState,eps_sig_lam)
#            
#            if iState == 0:
#                nSnaps = len(LJ_total_basis)
#                LJ_total_basis_rr_state = np.zeros([nStates,nSnaps])
#                press_basis_rr_state = np.zeros([nStates,nSnaps])
#                U_total_basis_rr_state = np.zeros([nStates,nSnaps])
#                
#            LJ_total_basis_rr_state[iState,:] = LJ_total_basis 
#            press_basis_rr_state[iState,:] = press_basis
#            U_total_basis_rr_state[iState,:] = LJ_total_basis_rr_state[iState,:] + self.U_novdw_state[iState,:]
#                             
##        self.LJ_total_basis_rr_state, self.press_basis_rr_state = LJ_total_basis_rr_state, press_basis_rr_state
#        return LJ_total_basis_rr_state, U_total_basis_rr_state, press_basis_rr_state
#
#    def calc_refs_basis(self):
#        eps_sig_lam_refs, nRefs = self.eps_sig_lam_refs, self.nRefs
#        
#        #print(eps_sig_lam_refs)
#        for iiRef, eps_sig_lam_ref in enumerate(eps_sig_lam_refs):
#            
#            #eps_sig_lam_ref = eps_sig_lam_refs[:,iiRef]
#            
#            #print(eps_sig_lam_ref)
#            
#            if iiRef == 0:
#                
#                LJ_total_basis_ref_state, U_total_basis_ref_state, press_basis_ref_state = self.UP_basis_states(eps_sig_lam_ref)
#                nSnaps = LJ_total_basis_ref_state.shape[1]
#                assert nStates == LJ_total_basis_ref_state.shape[0], "Number of states does not match dimension"
#                LJ_total_basis_refs_state = np.zeros([nRefs,nStates,nSnaps])
#                U_total_basis_refs_state = np.zeros([nRefs,nStates,nSnaps])
#                press_basis_refs_state = np.zeros([nRefs,nStates,nSnaps])
#            
#            else:
#            
#                LJ_total_basis_ref_state, U_total_basis_ref_state, press_basis_ref_state = self.UP_basis_states(eps_sig_lam_ref)
#            
#            LJ_total_basis_refs_state[iiRef], U_total_basis_refs_state[iiRef], press_basis_refs_state[iiRef] = LJ_total_basis_ref_state, U_total_basis_ref_state, press_basis_ref_state
#        
#        self.LJ_total_basis_refs_state, self.U_total_basis_refs_state, self.press_basis_refs_state =  LJ_total_basis_refs_state, U_total_basis_refs_state, press_basis_refs_state                        
#    
#    def validate_ref(self):
#        iRef, iRefs = self.iRef, self.iRefs
#
#        for iiiRef, iiRef in enumerate(iRefs):
#
#            if iRef == iiRef:
#
#                LJ_total_basis_ref_state, U_total_basis_ref_state, press_basis_ref_state = self.LJ_total_basis_refs_state[iiiRef], self.U_total_basis_refs_state[iiiRef], self.press_basis_refs_state[iiiRef]
#        
#        g_start = 28 #Row where data starts in g_energy output
#        g_LJ_sr = 1 #Column where LJ (SR) is located
#        g_LJ_dc = 2 #Column where LJ (DC) is located
#        g_en = 3 #Column where the potential energy is located
#        g_p = 5 #Column where p is located
#        
#        APD_U = np.zeros(nStates)
#        APD_LJ = np.zeros(nStates)
#        APD_P = np.zeros(nStates)
#
##        self.UP_basis_states(eps_sig_lam_ref)
#        
##        LJ_total_basis_ref_state = self.LJ_total_basis_rr_state 
##        press_basis_ref_state = self.press_basis_rr_state
#        
#        for iState in range(nStates):
#            
#            LJ_total_basis_ref = LJ_total_basis_ref_state[iState]
#            U_total_basis_ref = U_total_basis_ref_state[iState]
#            press_basis_ref = press_basis_ref_state[iState]
#            
#            fpath = fpath_all[iState]
#            
#            en_p = open('../ref'+str(iRef)+'/'+fpath+'energy_press_ref%srr%s.xvg' %(iRef,iRef),'r').readlines()[g_start:] #Read all lines starting at g_start for "state" k
#
#            nSnaps = len(en_p)
#
#            press_ref = np.zeros(nSnaps)
#            LJ_total_ref = np.zeros(nSnaps)
#            U_total_ref = np.zeros(nSnaps)
#            
#            for frame in range(nSnaps):
#            
#                LJ_sr_ref = float(en_p[frame].split()[g_LJ_sr])
#                LJ_dc_ref = float(en_p[frame].split()[g_LJ_dc])
#                LJ_total_ref[frame] = LJ_sr_ref + LJ_dc_ref
#                U_total_ref[frame] = float(en_p[frame].split()[g_en])
#                press_ref[frame] = float(en_p[frame].split()[g_p])
#            
#            LJ_dev = (LJ_total_basis_ref - LJ_total_ref)/LJ_total_ref*100.
#            U_dev = (U_total_basis_ref - U_total_ref)/U_total_ref*100.
#            press_dev = (press_basis_ref - press_ref)/np.mean(press_ref)*100.
#            
##            for LJ, press in zip(LJ_dev,press_dev):
##                print(LJ,press)
#            APD_LJ[iState] = np.mean(LJ_dev)
#            APD_U[iState] = np.mean(U_dev)
#            APD_P[iState] = np.mean(press_dev)
##            print(np.mean(LJ_dev))
##            print(np.mean(press_dev))
# 
#        print('Average percent deviation in non-bonded energy from basis functions compared to reference simulations: '+str(np.mean(APD_LJ)))           
#        print('Average percent deviation in internal energy from basis functions compared to reference simulations: '+str(np.mean(APD_U)))
#        print('Average percent deviation in pressure from basis functions compared to reference simulations: '+str(np.mean(APD_P)))
#
#    def validate_refs(self):
#        
#        iRef,iRefs = self.iRef,self.iRefs
#        
#        for iiiRef, iiRef in enumerate(iRefs):
#    
#            LJ_total_basis_ref_state, U_total_basis_ref_state, press_basis_ref_state = self.LJ_total_basis_refs_state[iiiRef], self.U_total_basis_refs_state[iiiRef], self.press_basis_refs_state[iiiRef]
#            
#            g_start = 28 #Row where data starts in g_energy output
#            g_LJ_sr = 1 #Column where LJ (SR) is located
#            g_LJ_dc = 2 #Column where LJ (DC) is located
#            g_en = 3 #Column where the potential energy is located
#            g_p = 5 #Column where p is located
#            
#            APD_U = np.zeros(nStates)
#            APD_LJ = np.zeros(nStates)
#            APD_P = np.zeros(nStates)
#    
#    #        self.UP_basis_states(eps_sig_lam_ref)
#            
#    #        LJ_total_basis_ref_state = self.LJ_total_basis_rr_state 
#    #        press_basis_ref_state = self.press_basis_rr_state
#            
#            for iState in range(nStates):
#                
#                LJ_total_basis_ref = LJ_total_basis_ref_state[iState]
#                U_total_basis_ref = U_total_basis_ref_state[iState]
#                press_basis_ref = press_basis_ref_state[iState]
#                
#                fpath = fpath_all[iState]
#                
#                en_p = open('../ref'+str(iRef)+'/'+fpath+'energy_press_ref%srr%s.xvg' %(iRef,iiRef),'r').readlines()[g_start:] #Read all lines starting at g_start for "state" k
#    
#                nSnaps = len(en_p)
#    
#                press_ref = np.zeros(nSnaps)
#                LJ_total_ref = np.zeros(nSnaps)
#                U_total_ref = np.zeros(nSnaps)
#                
#                for frame in range(nSnaps):
#                    
#                    LJ_sr_ref = float(en_p[frame].split()[g_LJ_sr])
#                    LJ_dc_ref = float(en_p[frame].split()[g_LJ_dc])
#                    LJ_total_ref[frame] = LJ_sr_ref + LJ_dc_ref #Previously for ethane I justed used the potential for LJ
#                    U_total_ref[frame] = float(en_p[frame].split()[g_en])
#                    press_ref[frame] = float(en_p[frame].split()[g_p])
#                    
#                LJ_dev = (LJ_total_basis_ref - LJ_total_ref)/LJ_total_ref*100.
#                U_dev = (U_total_basis_ref - U_total_ref)/U_total_ref*100.
#                press_dev = (press_basis_ref - press_ref)/np.mean(press_ref)*100.
#
#    #            for LJ, press in zip(LJ_dev,press_dev):
#    #                print(LJ,press)
#                APD_LJ[iState] = np.mean(LJ_dev)
#                APD_U[iState] = np.mean(U_dev)
#                APD_P[iState] = np.mean(press_dev)
#    #            print(np.mean(LJ_dev))
#    #            print(np.mean(press_dev))
#            #print('iRef= '+str(iRef))
#            #print('iiRef= '+str(iiRef))
#            #print('Average percent deviation in non-bonded energy from basis functions compared to reference simulations: '+str(np.mean(APD_LJ)))           
#            #print('Average percent deviation in internal energy from basis functions compared to reference simulations: '+str(np.mean(APD_U)))
#            #print('Average percent deviation in pressure from basis functions compared to reference simulations: '+str(np.mean(APD_P)))
#            assert np.abs(np.mean(APD_LJ)) < 1e-3, 'Basis function non-bonded energy error too large: '+str(np.mean(APD_LJ))+' for: iRef= '+str(iRef)+' iiRef= '+str(iiRef)+'. The MAPD_LJ is: '+str(np.max(APD_LJ))+'% for '+fpath_all[np.argmax(APD_LJ)]+'where Unb_basis for first frame is: '+str(LJ_total_basis_ref_state[iState][0])+'(kJ/mol)'  
#            assert np.abs(np.mean(APD_U)) < 1e-3, 'Basis function internal energy error too large: '+str(np.mean(APD_U))+' for: iRef= '+str(iRef)+' iiRef= '+str(iiRef)
#            assert np.abs(np.mean(APD_P)) < 1e-1, 'Basis function pressure error too large: '+str(np.mean(APD_P))+' for: iRef= '+str(iRef)+' iiRef= '+str(iiRef)
#
#def UP_basis_mult_refs(basis):
#    
#    nRefs = len(basis) #Rather than providing nRefs as a variable we will just determine it from the size of basis
#    
#    for iiRef in range(nRefs): 
#            
#        LJ_total_basis_ref_state, U_total_basis_ref_state, press_basis_ref_state = basis[iiRef].LJ_total_basis_refs_state, basis[iiRef].U_total_basis_refs_state, basis[iiRef].press_basis_refs_state
#        
#        if iiRef == 0:
#            
#            #print(LJ_total_basis_ref_state.shape)
#            nSnaps = LJ_total_basis_ref_state.shape[2]
#        
#            LJ_total_basis_refs = np.zeros([nRefs,nRefs,nStates,nSnaps])
#            U_total_basis_refs = np.zeros([nRefs,nRefs,nStates,nSnaps])
#            press_basis_refs = np.zeros([nRefs,nRefs,nStates,nSnaps])                                                                                                                                                                           
##            for jRef in range(args.nRefs):
#                       
#        LJ_total_basis_refs[iiRef] = LJ_total_basis_ref_state
#        U_total_basis_refs[iiRef] = U_total_basis_ref_state
#        press_basis_refs[iiRef] = press_basis_ref_state
#                        
#     # Validated that these values agreed with what is expected                   
##        print(LJ_total_basis_refs[0,0,0,:])
##        print(LJ_total_basis_refs[1,0,0,:])
##        print(LJ_total_basis_refs[1,1,0,:])
##        print(LJ_total_basis_refs.shape)
#
#    return LJ_total_basis_refs, U_total_basis_refs, press_basis_refs
#
#def main():
#    
#    parser = argparse.ArgumentParser()
#    parser.add_argument("-nRefs","--nRefs",type=int,help="set the integer value for the number of references")
#    parser.add_argument("-iRef","--iRef",type=int,nargs='+',help="set the integer value for the reference")
#    args = parser.parse_args()
#    
#    eps_low = np.array([88.,42.,np.sqrt(88.*42.)])
#    eps_high = np.array([108.,50.,np.sqrt(108.*50.)])
#    sig_low = np.array([0.365,0.385,0.375])
#    sig_high = np.array([0.385,0.405,0.395])
#    lam_low = np.array([12.,12.,12.])
#    lam_high = np.array([18.,18.,18.])
#    
#    basis = []
#    
##    for i in range(2):
##        
##        basis.append(basis_function(Temp_sim,rho_sim,args.iRef,88.,108.,0.365,0.385,12.,12.))
##        basis[i].validate_ref()
##   
#    if args.nRefs:
#
#        for iiRef in range(args.nRefs):
#            basis.append(basis_function(Temp_sim,rho_sim,iiRef,range(args.nRefs),eps_low,eps_high,sig_low,sig_high,lam_low,lam_high,True))
#            basis[iiRef].validate_refs()
#            
#        LJ_total_basis_refs, U_total_basis_refs, press_basis_refs = UP_basis_mult_refs(basis)
#            
#    if args.iRef:
#        
#        for iiRef in args.iRef:
#        
#            basis.append(basis_function(Temp_sim,rho_sim,iiRef,0,eps_low,eps_high,sig_low,sig_high,lam_low,lam_high,False)) 
##            basis.append(basis_function(Temp_sim,rho_sim,iiRef,eps_low,eps_high,sig_low,sig_high,lam_low,lam_high)) 
#            basis[0].validate_ref()
##            print(np.mean(basis[0].U_novdw_state))
##            print(np.max(np.abs(basis[0].U_novdw_state)))
#    # Testing how much of a difference the points for basis functions make
# 
##    basis.append(basis_function(Temp_sim,rho_sim,args.iRef,88.,108.,0.375,0.375,12.,12.)) 
##    basis[0].validate_ref()
##    basis.append(basis_function(Temp_sim,rho_sim,args.iRef,98.,98.,0.365,0.385,12.,12.))
##    basis[1].validate_ref()
##    basis.append(basis_function(Temp_sim,rho_sim,args.iRef,88.,108.,0.365,0.385,12.,12.))
##    basis[2].validate_ref() 

                
#for frame in xrange(nSnaps):
#    U_basis_vdw = LJsr[:,frame]
#    Vir_basis_vdw = Vir_1[:,frame]
#    Vir_basis_LINCS = Vir_0[:,frame] - Vir_1[:,frame] - Vir_2[:,frame]
#    sumr6lam[:,frame] = np.linalg.solve(Cmatrix,U_basis_vdw)
#    sumrdr6lam_vdw[:,frame] = np.linalg.solve(Cmatrix,Vir_basis_vdw)
#    sumrdr6lam_LINCS[:,frame] = np.linalg.solve(Cmatrix,Vir_basis_LINCS)
                        
#        fpath = "../ref"+str(iRef)+"/"
#        
#        f = open(fpath+'Cmatrix','w')
#        g = open(fpath+'lam_index','w')
#        
#        for ilam, lam in enumerate(lam_index):
#            for jlam in range(len(lam_basis)):
#                f.write(str(Cmatrix[ilam,jlam])+'\t')
#            f.write('\n')
#            g.write(str(lam)+'\t')              
#        
#        f.close()
#        g.close()       
             
#        print(Cmatrix)
#        print(lam_index)
    
        
        
        
        
        
        
        

# Simulation system specifications
#
#rc = 1.4 #[nm]
#N_sites = 3
##N_inter = N_sites**2
#N_inter = {'CH3':4,'CH2':1,'CH3CH2':4}
#
##Read in the simulation specifications
#
#ITIC = np.array(['Isotherm', 'Isochore'])
#Temp_ITIC = {'Isochore':[],'Isotherm':[]}
#rho_ITIC = {'Isochore':[],'Isotherm':[]}
#Nmol = {'Isochore':[],'Isotherm':[]}
#Temps = {'Isochore':[],'Isotherm':[]}
#rhos_ITIC = {'Isochore':[],'Isotherm':[]}
#rhos_mass_ITIC = {'Isochore':[],'Isotherm':[]}
#nTemps = {'Isochore':[],'Isotherm':[]}
#nrhos = {'Isochore':[],'Isotherm':[]}
#
#Temp_sim = np.empty(0)
#rho_sim = np.empty(0)
#Nmol_sim = np.empty(0)
#
##Extract state points from ITIC files
## Move this outside of this loop so that we can just call it once, also may be easier for REFPROP
## Then again, with ITIC in the future Tsat will depend on force field
#
#for run_type in ITIC:
#
#    run_type_Settings = np.loadtxt(run_type+'Settings.txt',skiprows=1)
#
#    Nmol[run_type] = run_type_Settings[:,0]
#    Lbox = run_type_Settings[:,1] #[nm]
#    Temp_ITIC[run_type] = run_type_Settings[:,2] #[K]
#    Vol = Lbox**3 #[nm3]
#    rho_ITIC[run_type] = Nmol[run_type] / Vol #[molecules/nm3]
#    rhos_ITIC[run_type] = np.unique(rho_ITIC[run_type])
#    rhos_mass_ITIC[run_type] = rhos_ITIC[run_type] * Mw / N_A * nm3_to_m3 #[kg/m3]
#    nrhos[run_type] = len(rhos_ITIC[run_type])
#    Temps[run_type] = np.unique(Temp_ITIC[run_type])
#    nTemps[run_type] = len(Temps[run_type]) 
# 
#    Temp_sim = np.append(Temp_sim,Temp_ITIC[run_type])
#    rho_sim = np.append(rho_sim,rho_ITIC[run_type])
#    Nmol_sim = np.append(Nmol_sim,Nmol[run_type])
#
#nTemps['Isochore']=2 #Need to figure out how to get this without hardcoding
#    
#rho_mass = rho_sim * Mw / N_A * nm3_to_m3 #[kg/m3]
##rho_mass = rho_sim * Mw / N_A * nm3_to_ml #[gm/ml]
#
#nStates = len(Temp_sim)
#
## Create a list of all the file paths (without the reference directory, just the run_type, rho, Temp)
#
#fpath_all = []
#
#for run_type in ITIC: 
#
#    for irho  in np.arange(0,nrhos[run_type]):
#
#        for iTemp in np.arange(0,nTemps[run_type]):
#
#            if run_type == 'Isochore':
#
#                fpath_all.append(run_type+'/rho'+str(irho)+'/T'+str(iTemp)+'/NVT_eq/NVT_prod/')
#
#            else:
#
#                fpath_all.append(run_type+'/rho_'+str(irho)+'/NVT_eq/NVT_prod/')
#                
#assert nStates == len(fpath_all), 'Number of states does not match number of file paths'
#
#
#if __name__ == '__main__':
#    '''
#    python basis_function_class.py --nRefs XX --iRef XX
#  
#    "--nRefs XX" or "--iRef XX" flag is required, sets the integer value for nRefs or iRef
#    '''
#
#    main()

hexane_bf = basis_function(iRef=0,iRefs=0,N_basis=6,N_frames=101)
hexane_bf.parity_plot()
hexane_bf.compute_U_theta()