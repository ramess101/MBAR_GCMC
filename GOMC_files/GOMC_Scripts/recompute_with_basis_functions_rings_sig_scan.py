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
model = 'lam20_basis'

root_path = '/home/ram9/'+compound+'/GOMC/GCMC/'+model+'/'

NStates = 7                                    
NHists = 1250


### For model = 'TraPPE_basis'
#NBasis = 6
#
#ilam_dic = {6:0,12:1,14:2,16:3,18:4,20:5}
#
#eps_ref=52.5
#sig_ref=3.91
#lam_ref=12
#iFF_ref = 0
#nsig = 21
#
#lam_range = [12,14,16,18,20]

### For model = 'lam16_basis'
NBasis = 2
#
ilam_dic = {6:0,16:1}
#
eps_ref=70.0
sig_ref=3.89
lam_ref=16
iFF_ref = 0
nsig = 41
#
lam_range = [16]

### For model = 'lam14_basis'
#NBasis = 2

#ilam_dic = {6:0,14:1}

#eps_ref=61.5
#sig_ref=3.93
#lam_ref=14
#iFF_ref = 0
#nsig = 41

#lam_range = [14]

### For model = 'lam18_basis'
#NBasis = 2

#ilam_dic = {6:0,18:1}

#eps_ref=77.0
#sig_ref=3.89
#lam_ref=18
#iFF_ref = 0
#nsig = 41

#lam_range = [18]

### For model = 'lam20_basis'
NBasis = 2

ilam_dic = {6:0,20:1}

eps_ref=84.0
sig_ref=3.88
lam_ref=20
iFF_ref = 0
nsig = 41

lam_range = [20]


sig_low = 3.8
sig_high = 4.0

for lam in lam_range:

    iFF = 1

    eps_range = np.array([eps_ref])
    sig_range = np.linspace(sig_low,sig_high,nsig)

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

                    f = open(root_path+'lam'+str(lam)+'recompute_sig'+str(iFF)+'/his'+str(iState+1)+'a.dat','w')

                except:

                    os.mkdir(root_path+'lam'+str(lam)+'recompute_sig'+str(iFF))
                    f = open(root_path+'lam'+str(lam)+'recompute_sig'+str(iFF)+'/his'+str(iState+1)+'a.dat','w')

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

            f = open(root_path+'lam'+str(lam)+'recompute_sig'+str(iFF)+'/eps_sig_lam.txt','w')
            f.write('eps'+'\t'+'sig'+'\t'+'lam'+'\n')
            f.write(str(eps)+'\t'+str(sig)+'\t'+str(lam))
            f.close()

            iFF += 1
