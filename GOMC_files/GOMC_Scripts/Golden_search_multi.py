"""
Golden section search, derivative free 1-d optimizer
"""

from __future__ import division 
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


R_ratio=0.61803399
C_ratio=1.-R_ratio

def GOLDEN(func,AX,BX,CX,TOL):
    X0 = AX
    X3 = CX
    if np.abs(CX-BX) > np.abs(BX-AX):
        X1 = BX
        X2 = BX + C_ratio*(CX-BX)
    else:
        X2 = BX
        X1 = BX - C_ratio*(BX-AX)
    F1 = func(X1)
    F2 = func(X2) 
    print(X0,X1,X2,X3)
    while np.abs(X3-X0) > TOL*(np.abs(X1)+np.abs(X2)):
    #for i in np.arange(0,50):
        if F2 < F1:
            X0 = X1
            X1 = X2
            X2 = R_ratio*X1 + C_ratio*X3
            F1 = F2
            F2 = func(X2)
            print(X0,X1,X2,X3)
        else:
            X3 = X2
            X2 = X1
            X1 = R_ratio*X2 + C_ratio*X0
            F2 = F1
            F1 = func(X1)
            print(X0,X1,X2,X3)      
    
    if F1 < F2:
        GOLDEN = F1
        XMIN = X1
    else:
        GOLDEN = F2
        XMIN = X2
    
    return XMIN

def GOLDEN_multi(func,guesses,lower_bounds,upper_bounds,TOL,maxit,show_plot=False):
    '''
    Solves multiple functions that are contained within a single matrix
    Designed for implementation with n-dimensional MBAR
    
    inputs:
        guesses: n-dimensional array of guesses
        lower_bounds: n-dimensional array of lower_bounds
        upper_bounds: n-dimensional array of upper_bounds
        TOL: tolerance, singe value
        maxit: maximum number of iterations
    outputs:
        XMIN: n-dimensional array of optimal values
    '''
    ### I need to fix this for when the initial bounds are not adequate
    AX_all = lower_bounds.copy()
    CX_all = upper_bounds.copy()
    BX_all = guesses.copy()

    FA_all = func(AX_all)
    FB_all = func(BX_all)
    FC_all = func(CX_all)
    
    if (FB_all < FA_all).all() and (FB_all < FC_all).all():
        print('This is a well initialized system')
    else:
        recompute_C = False
        recompute_A = False
        recompute_B = False
        initialized = False
        attempts = 0
        max_attempts = 5
        while not initialized:
#            plt.plot(BX_all,FB_all,'k-')
#            plt.plot(AX_all,FA_all,'b-')
#            plt.plot(CX_all,FC_all,'r-')
#            plt.show()
            for i, (ai,bi,ci,fa,fb,fc) in enumerate(zip(AX_all,BX_all,CX_all,FA_all,FB_all,FC_all)):
                if show_plot: plt.plot([ai,bi,ci],[fa,fb,fc])
                if fb > fa and fb < fc: #Guess is greater than lower bound but less than upper bound
                    AX_all[i] = ai - (bi - ai)
                    BX_all[i] = ai
                    CX_all[i] = bi
                    FB_all[i] = fa
                    FC_all[i] = fb
                    recompute_A = True
                    print('Guess is higher than lower bound')
                elif fb > fc and fb < fa: #Guess is greater than upper bound but less than lower bound
                    CX_all[i] = ci + (ci - bi)
                    BX_all[i] = ci
                    AX_all[i] = bi
                    FB_all[i] = fc
                    FA_all[i] = fb
                    recompute_C = True
                    print('Guess is higher than upper bound')
                elif np.abs(fb - 1.) < 1e-5:
                    BX_all[i] = ai + (ci-ai)*np.random.random()
                    recompute_B = True
                    print('Guess is non-informative, equal to 1')
                    attempts += 1
                    if attempts > max_attempts: 
                        print('Could not initialize system')
                        if show_plot: plt.show()
                        recompute_B = False
                        recompute_A = False
                        recompute_C = False
#                        return BX_all
                elif np.abs(fb - fa)/fa < 1e-3 and np.abs(fb - fc)/fc < 1e-3:
                    if np.random.rand() < 0.5:
                        BX_all[i] = (bi + ci)/2.
                    else:
                        BX_all[i] = (bi + ai)/2.
                    recompute_B = True
                    print('Guess is almost equal to bounds')
                elif fb > fa and fb > fc: #Guess is greater than both the lower and upper bound
                    print('This function appears to be multimodal')
                    break
            if show_plot: plt.show()
            if recompute_C:
                FC_all = func(CX_all)
                initialized=False
            if recompute_A:
                FA_all = func(AX_all)
                initialized=False
            if recompute_B:
                FB_all = func(BX_all)
                initialized=False
            if not recompute_C and not recompute_A and not recompute_B:
                initialized=True
                print('This is a well initialized system')
            recompute_C = False
            recompute_A = False
            recompute_B = False
    
#    for i, (ai,bi,ci,fa,fb,fc) in enumerate(zip(AX_all,BX_all,CX_all,FA_all,FB_all,FC_all)):
#        while fb > fa and fb < fc: #Guess is greater than lower bound but less than upper bound
#            CX_all[i] = ai - (bi - ai)
#        while fb > fa and fb > fc: #Guess is greater than both the lower and upper bound
#            print('This function appears to be multimodal')
#            break
#        while fb > fc and fb < fa: #Guess is greater than upper bound but less than lower bound
#            AX_all[i] = ci + (ci - bi)
#        if fb < fa and fb < fc:
#            print('This is a well initialized system')
    
    nfun = len(AX_all) #Number of functions being optimized
    
    XMIN = np.zeros(nfun)
    flagnew_all = np.zeros(nfun)
    Xnew_all = np.zeros(nfun)
    
    X0 = AX_all.copy()
    X3 = CX_all.copy()
    X1 = BX_all.copy()
    X2 = BX_all.copy()
            
    for i, (AX, BX, CX) in enumerate(zip(AX_all,BX_all,CX_all)):
        if np.abs(CX-BX) > np.abs(BX-AX):
            X1[i] = BX
            X2[i] = BX + C_ratio*(CX-BX)
        else:
            X2[i] = BX
            X1[i] = BX - C_ratio*(BX-AX)
    F1_all = func(X1)
    F2_all = func(X2) 
#    print(X0,X1,X2,X3)
    nit = 0
    while (np.abs(X3-X0) > TOL*(np.abs(X1)+np.abs(X2))).any() and nit < maxit:
        for i, (F1, F2) in enumerate(zip(F1_all,F2_all)):
            if F1 == F2: 
                F1 + (np.random.random(1) - 0.5)*10e-10 
                print('F1 and F2 were equal')
            if F2 < F1:
                X0[i] = X1[i]
                X1[i] = X2[i]
                X2[i] = R_ratio*X1[i] + C_ratio*X3[i]
                F1_all[i] = F2
#                F2 = f_eval(X2)
                Xnew_all[i] = X2[i]
                flagnew_all[i] = 2
                #print(str(i)+' has flag =2')
            else:
                X3[i] = X2[i]
                X2[i] = X1[i]
                X1[i] = R_ratio*X2[i] + C_ratio*X0[i]
                F2_all[i] = F1
#                F1 = f_eval(X1)
                Xnew_all[i] = X1[i]
                flagnew_all[i] = 1
                #print(str(i)+' has flag =1')
        Fnew_all = func(Xnew_all) #Only a single function evaluation for new set of Xs
        for i, (fi, Fi) in enumerate(zip(flagnew_all,Fnew_all)):
            if fi == 1:
                F1_all[i] = Fi
            elif fi == 2:
                F2_all[i] = Fi 
        nit += 1
        #print(nit,np.abs(X3-X0) > TOL*(np.abs(X1)+np.abs(X2)))
#        print(X0,X1,X2,X3)           
    for i, (F1, F2) in enumerate(zip(F1_all,F2_all)):
        if F1 < F2:
            XMIN[i] = X1[i]
        else:
            XMIN[i] = X2[i]
    
    if nit < maxit:
        print('Converged within '+str(nit)+' iterations')
    else:
        print('Did not converge within the alloted '+str(maxit)+' iterations')
    
    return XMIN

def main():
    #567114216.81*x**2 - 1089657403.26*x + 523426800.54
    #574154474.878275*(x-0.05)**2 - 1097704524.393190*(x-0.05) + 524667789.414363
    
    f_eval1 = lambda x: 567114216.81*x**2 - 1089657403.26*x + 523426800.54
    f_eval2 = lambda x: 574154474.878275*(x-0.05)**2 - 1097704524.393190*(x-0.05) + 524667789.414363
                                        
    #f_eval = lambda x: [f_eval1(x[0]),f_eval2(x[1])], print('function call')  
    
    def f_eval(x):
        print('function call')     
        return [f_eval1(x[0]),f_eval2(x[1])]                             
    
    guess = np.array([0.95,1.0143,18])
    
    xplot = np.linspace(-20,20,50)
    yplot = f_eval([xplot,xplot])
    
    plt.plot(xplot,yplot[0],label='Function')
    plt.plot(xplot,yplot[1],label='Function')
    plt.legend()
    plt.show()
               
    xmin = GOLDEN_multi(f_eval,np.array([guess[1],0.1*guess[1]]),np.array([guess[0],0.5*guess[0]]),np.array([guess[2],3*guess[2]]),0.0001,100)
    
    print(xmin)
    
if __name__ == '__main__':
    '''
    python Golden_search_multi.py
  
    '''

    main()   
