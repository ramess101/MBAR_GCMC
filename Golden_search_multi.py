"""
Golden section search, derivative free 1-d optimizer
"""

from __future__ import division 
import numpy as np
from scipy.optimize import minimize
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

def GOLDEN_multi(func,guesses,lower_bounds,upper_bounds,TOL,maxit):
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
    
    AX_all = lower_bounds.copy()
    CX_all = upper_bounds.copy()
    BX_all = guesses.copy()
    
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
            else:
                X3[i] = X2[i]
                X2[i] = X1[i]
                X1[i] = R_ratio*X2[i] + C_ratio*X0[i]
                F2_all[i] = F1
#                F1 = f_eval(X1)
                Xnew_all[i] = X1[i]
                flagnew_all[i] = 1
        Fnew_all = func(Xnew_all) #Only a single function evaluation for new set of Xs
        for i, (fi, Fi) in enumerate(zip(flagnew_all,Fnew_all)):
            if fi == 1:
                F1_all[i] = Fi
            elif fi == 2:
                F2_all[i] = Fi 
        nit += 1
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
    
    guess = np.array([0.8,1.0143,1.2])
    
    xplot = np.linspace(min(guess),max(guess),50)
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
