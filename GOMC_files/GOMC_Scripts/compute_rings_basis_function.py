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

def compute_rings_basis_function(compound,model,NStates,NBasis):
    '''
    Converts histogram files into basis functions
    Outputs to root_path/basisFunctions/
    '''

    root_path = '/home/ram9/'+compound+'/GOMC/GCMC/'+model+'/'

    ### Load in the epsilon, sigma, lambda values

    eps_sig_lam_basis = np.loadtxt(root_path+'eps_sig_lam_basis',skiprows=1)

    assert len(eps_sig_lam_basis) == NBasis, "Number of basis functions is not equal to the dimensions of eps_sig_lam_basis file."

    Cmatrix = np.zeros([NBasis,NBasis])
    Ncoef_all = np.zeros(NBasis)
    lam_all = np.zeros(NBasis)

    for iBasis, eps_sig_lam_i in enumerate(eps_sig_lam_basis):

        eps_i = eps_sig_lam_i[0]
        sig_i = eps_sig_lam_i[1]
        lam_i = eps_sig_lam_i[2]

        C6_i, Clam_i, Ncoef_i = convert_eps_sig_C6_Clam(eps_i,sig_i,lam_i,print_Cit=False)
        Cmatrix[iBasis,0] = C6_i

        if iBasis == 0:

            Cmatrix[iBasis,1] = Clam_i

        else:

            Cmatrix[iBasis,iBasis] = Clam_i

        Ncoef_all[iBasis] = Ncoef_i

        lam_all[iBasis] = lam_i

    lam_all[0] = 6.

    #print(Cmatrix)
    #print(Ncoef_all)

    for iState in range(NStates):
    
        State_path = root_path+'State'+str(iState)+'/'

        try:

            f = open(root_path+'basisFunctions/basis'+str(iState+1)+'.txt','w')

        except:

            os.mkdir(root_path+'basisFunctions')
            f = open(root_path+'basisFunctions/basis'+str(iState+1)+'.txt','w')

        try:

            Hist_data = np.loadtxt(root_path+'/histfiles/his'+str(iState+1)+'a.dat',skiprows=1)
            NSnapshots = len(Hist_data)
            Hist_header = np.genfromtxt(root_path+'/histfiles/his'+str(iState+1)+'a.dat',skip_footer=NSnapshots)

            for header_i in Hist_header:

                f.write(str(header_i)+'\t')

            f.write('\n')

        except:

            print('No file for State'+str(iState))
            
        for ilam in lam_all:

            f.write('C'+str(int(ilam))+'\t')

        f.write('\n'+'Prexponent (Ncoef) not included'+'\n')

        N_ref = Hist_data[:,0]
        u_basis_all = Hist_data[:,1:]

        for iSnap in range(NSnapshots):

           u_basis_i = u_basis_all[iSnap,:]
           uNcoef_basis_i = u_basis_i / Ncoef_all

           #print(u_basis_i)
           #print(uNcoef_basis_i)

           basisFunctions = np.linalg.solve(Cmatrix,uNcoef_basis_i)
           #basisFunctions = np.linalg.lstsq(Cmatrix,uNcoef_basis_i).x

           for basis_i in basisFunctions:

               f.write(str(basis_i)+'\t')

           f.write('\n')

        f.close()

    f = open(root_path+'basisFunctions/Ncoef_basis.txt','w')

    for Ncoef in Ncoef_all:

        f.write(str(Ncoef)+'\t')

    f.close()

def main():
    '''
    Reads in the argument values and calls compute_rings_basis_function
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--compound",type=str,help="Please enter the name of the compound to analyze")
    parser.add_argument("-ff","--forcefield",type=str,help="Please enter the name of the force field/model to analyze")
    parser.add_argument("-NS","--NStates",type=int,help="Please enter the number of states (int), typical values are 7-10")
    parser.add_argument("-NB","--NBasis",type=int,help="Please enter the number of basis parameters rerun (int), typical values are 2-6")
    args = parser.parse_args()

    compound = args.compound
    model = args.forcefield
    NStates = args.NStates
    NBasis = args.NBasis

    compute_rings_basis_function(compound,model,NStates,NBasis)

if __name__ == '__main__':
    '''
    Converts the histogram files into basis function files
    '''
   
    main()
