'''
Compiles the histograms for model and rerun model
'''

from __future__ import division
import numpy as np 
import os, sys, argparse, shutil

def compile_Hist_data(compound,model,NStates,NHists,NBasis):
    '''
    Reads in and compiles all the histogram data for NStates, NHists, and NBasis
    Writes values into root_path/histfiles
    '''

    root_path = '/home/ram9/'+compound+'/GOMC/GCMC/'+model+'/'

    for iState in range(NStates):
    
        State_path = root_path+'State'+str(iState)+'/'

        try:

            f = open(root_path+'histfiles/his'+str(iState+1)+'a.dat','w')

        except:

            os.mkdir(root_path+'histfiles')
            f = open(root_path+'histfiles/his'+str(iState+1)+'a.dat','w')

        try:

            Hist_data = np.loadtxt(State_path+'his_rr1_basis_function_0',skiprows=1)
            NSnapshots = len(Hist_data)
            Hist_header = np.genfromtxt(State_path+'his_rr1_basis_function_0',skip_footer=NSnapshots)

            for header_i in Hist_header:

                f.write(str(header_i)+'\t')

            f.write('\n')

        except:

            print('No file for State'+str(iState))
            
        for iHist in np.arange(1,NHists+1):

            u_basis_all = np.zeros([NSnapshots,NBasis])

            for iBasis in range(NBasis):

                try:
        
                    Hist_basis_i = np.loadtxt(State_path+'his_rr'+str(iHist)+'_basis_function_'+str(iBasis),skiprows=1)             

                    N_basis = Hist_basis_i[:,0]
                    u_basis = Hist_basis_i[:,1]

                    if iBasis == 0: 
                 
                        N_ref = N_basis.copy()
                        u_ref = u_basis.copy()        

                    else:

                        assert len(u_basis) == len(u_ref), 'Number of frames are different, basis has '+str(len(u_basis))+' frames and ref has '+str(len(u_ref))+' frames.'   
                        assert N_basis[0] == N_ref[0], 'First frame N are different, basis has '+str(N_basis[0])+' and ref has '+str(N_ref[0])    
                        assert N_basis[-1] == N_ref[-1], 'Last frame N are different, basis has '+str(N_basis[-1])+' and ref has '+str(N_ref[-1])

                    u_basis_all[:,iBasis] = u_basis

                except:

                    pass

            for N_j, u_basis_j in zip(N_ref,u_basis_all):

                f.write(str(int(N_j))+'\t')

                for iBasis in range(NBasis):

                    f.write(str(u_basis_j[iBasis])+'\t')

                f.write('\n')

        f.close()

def main():
    '''
    Reads in the argument values and calls compile_Hist_data
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--compound",type=str,help="Please enter the name of the compound to analyze")
    parser.add_argument("-ff","--forcefield",type=str,help="Please enter the name of the force field/model to analyze")
    parser.add_argument("-NS","--NStates",type=int,help="Please enter the number of states (int), typical values are 7-10")
    parser.add_argument("-NH","--NHists",type=int,help="Please enter the maximum number of histogram snapshots (int), typical values are 1250 or 2000")
    parser.add_argument("-NB","--NBasis",type=int,help="Please enter the number of basis parameters rerun (int), typical values are 2-6")
    args = parser.parse_args()

    compound = args.compound
    model = args.forcefield
    NStates = args.NStates
    NHists = args.NHists
    NBasis = args.NBasis

    compile_Hist_data(compound,model,NStates,NHists,NBasis)

if __name__ == '__main__':
    '''
    Compile the histogram files for the reruns to create basis functions
    '''
   
    main()
