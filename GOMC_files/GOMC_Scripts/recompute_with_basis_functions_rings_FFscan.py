'''
Computes the basis functions for cyclohexane
'''

from __future__ import division
import numpy as np 
import os, sys, argparse, shutil

kb = 1.38064852e-26 #[kJ/K]
NA = 6.02214e23 #[1/mol]

def convert_eps_sig_C6_Clam(eps_K,sig,lam,n=6.,print_Cit=True):
    Ncoef = lam/(lam-n)*(lam/n)**(n/(lam-n))
    eps = eps_K * kb * NA
    C6 = eps * sig ** n
    Clam = eps * sig ** lam
    
    if print_Cit:
    
        f = open('C6_it','w')
        f.write(str(C6))
        f.close()
        
        f = open('Clam_it','w')
        f.write(str(Clam))
        f.close()
        
    else:
        
        return C6, Clam, Ncoef

compound = 'CYC6'
model = 'TraPPE_basis'

root_path = '/home/ram9/'+compound+'/GOMC/GCMC/'+model+'/'

NStates = 7                                    
NHists = 2000
NBasis = 6

ilam_dic = {6:0,12:1,14:2,16:3,18:4,20:5}

eps_ref=52.5
sig_ref=3.91
lam_ref=12
iFF_ref = 0

eps_low_dic = {12:40,14:45,16:50,18:55,20:60}
eps_high_dic = {12:60,14:65,16:70,18:75,20:80}

lam_range = [20]# [12,14,16,18,20]

for lam in lam_range:

    iFF = 1

    eps_range = np.linspace(eps_low_dic[lam],eps_high_dic[lam],21)
    sig_range = np.linspace(3.8,4.0,21)

    for ieps, eps in enumerate(eps_range):

        for isig, sig in enumerate(sig_range):

            C6, Clam, Ncoef = convert_eps_sig_C6_Clam(eps,sig,lam,print_Cit=False)

            Carray = np.zeros(NBasis)

            Carray[0] = C6*Ncoef
            Carray[ilam_dic[lam]] = Clam*Ncoef

            #print(Carray)
            print('iFF = '+str(iFF)+' eps = '+str(eps)+' sig = '+str(sig)+' lam = '+str(lam))

            for iState in range(NStates):
    
                try:

                    f = open(root_path+'lam'+str(lam)+'recomputeFF'+str(iFF)+'/his'+str(iState+1)+'a.dat','w')

                except:

                    os.mkdir(root_path+'lam'+str(lam)+'recomputeFF'+str(iFF))
                    f = open(root_path+'lam'+str(lam)+'recomputeFF'+str(iFF)+'/his'+str(iState+1)+'a.dat','w')

                try:

                    basis_functions = np.loadtxt(root_path+'/basisFunctions/basis'+str(iState+1)+'.txt',skiprows=3)

                    Hist_data = np.loadtxt(root_path+'/histfiles/his'+str(iState+1)+'a.dat',skiprows=1)
                    NSnapshots = len(Hist_data)
                    Hist_header = np.genfromtxt(root_path+'/histfiles/his'+str(iState+1)+'a.dat',skip_footer=NSnapshots)

                    for header_i in Hist_header:

                        f.write(str(header_i)+'\t')

                    f.write('\n')

                except:

                    print('No file for State'+str(iState))

                N_ref = Hist_data[:,0]
                u_ref = Hist_data[:,1]

                u_iFF = np.linalg.multi_dot([basis_functions,Carray])

                for N_j, u_ref_j, u_iFF_j in zip(N_ref,u_ref,u_iFF):

                   f.write(str(int(N_j))+'\t'+str(u_ref_j)+'\t'+str(u_iFF_j)+'\n')

                f.close()

            f = open(root_path+'lam'+str(lam)+'recomputeFF'+str(iFF)+'/eps_sig_lam.txt','w')
            f.write('eps'+'\t'+'sig'+'\t'+'lam'+'\n')
            f.write(str(eps)+'\t'+str(sig)+'\t'+str(lam))
            f.close()

            iFF += 1
