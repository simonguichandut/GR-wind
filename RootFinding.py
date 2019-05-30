import numpy as np
from numpy.linalg import norm
from numpy import array
import sys
import os

from wind_GR import MakeWind
from IO import load_roots

def Jacobian(func,za,zb,*args):

    ''' First order numerical jacobian of a two input, two output function '''

    dx,dy = zb[0]-za[0] , zb[1]-za[1]
    faa,fba,fab = func(za,*args) , func([zb[0],za[1]],*args) , func([za[0],zb[1]],*args) 
    J11 = (fba[0]-faa[0])/dx
    J12 = (fab[0]-faa[0])/dy
    J21 = (fba[1]-faa[1])/dx
    J22 = (fab[1]-faa[1])/dy
    
    return [[J11,J12],[J21,J22]]


def JacobianUpdate(J0,za,zb,fa,fb,*args):
    
    ''' Jacobian update from Broyden's method '''
    
    J0,za,zb,fa,fb = array(J0),array(za),array(zb),array(fa),array(fb)
    dz,df = zb-za,fb-fa
    Jnew = J0 + 1/norm(dz)**2*np.outer(df-np.matmul(J0,dz),dz)
    return Jnew
    

def Newton_Raphson_2D(func,z0,z1,limits,*args,tol=1e-3,flagcatch=0):

    ''' Newton-Raphson root finding for a two input, two output function
        Two initial guesses have to be provided. Will soften iterations if 
        going out of bounds [xmin,xmax,ymin,ymax], or if obtaining invalid
        values. '''
    
    diff = 2*tol
    counter,f0,f1 = 0,func(z0,*args),func(z1,*args)
    Jold = [[0.00001,0.00001],[0.00001,-0.00001]]
    nitermax = 50
    while (diff>tol or norm(f1)>tol):

        J = Jacobian(func,z0,z1,*args)
        if counter != 0 and (True in np.isnan(J) or np.linalg.det(J)==0.0):
            print('Using Jacobian update')
            J = JacobianUpdate(Jold,z0,z1,f0,f1)
        
        pillow = 1
        znew = np.array(z1) - pillow*np.matmul(np.linalg.inv(J),f1)
        
        #  Out of bounds, catch level 1
        while znew[0]<limits[0] or znew[0]>limits[1] or znew[1]<limits[2] or znew[1]>limits[3] or (True in np.isnan(func(znew,*args))) :
            pillow/=3
            print('pillow update : ',pillow)
            znew = np.array(z1) - pillow*np.matmul(np.linalg.inv(J),f1)
            
        fnew = func(znew,*args)
            
        # solution space is not necesarily rectangular, catch level 2
        if flagcatch != 0:
            while flagcatch in fnew or (True in np.isnan(fnew)):
                pillow/=3
                print('pillow update v2: ',pillow)
                znew = np.array(z1) - pillow*np.matmul(np.linalg.inv(J),f1)
                fnew = func(znew,*args)
                if pillow < 1e-20 : sys.exit('Fell into a poorly behaved region, exiting..')
        
        z0,f0 = z1[:],f1[:]
        z1,f1 = znew[:],fnew[:]
        Jold = J
        print(counter+1,' : update [Edot,Ts]=',z1,'\n\t errors=',fnew,'\n\n')
        
        diff = abs(norm(z1)-norm(z0))  # difference between two iterations
        counter += 1
        
        
        if counter>nitermax:
            sys.exit('Not able to find a root after %d iterations, exiting..'%nitermax)

    print('Root found at : ',z1,'\n\n')        
        
    return z1


def RootFinder(logMdot,logTs0=7.4,box='on',Verbose=0,usefile=1):  
    
    ''' Finds the error-minimizing set of parameters (Edot,Ts) for a wind 
        with a given Mdot '''

    print('\n Starting root finding algorithm for logMdot = %.2f'%logMdot)

    # First check if we can start from a root in file

    find_first_root = 1
    if usefile:
        logMDOTS,roots = load_roots()
        if logMdot in logMDOTS:
            print('First root from file')
            z0 = roots[logMDOTS.index(logMdot)]
            z1 = [z0[0]+0.001,z0[1]+0.01]
            find_first_root = 0

    if find_first_root:
                    
        Edotmin = 1.01 # in units of LEdd. There are likely no solutions with a lower Edot
        
        # But this Edot might not converge
        err=MakeWind([Edotmin,logTs0],logMdot,Verbose=Verbose)
        while 100 in err:
            Edotmin += 0.002
            print('\nEdotmin: ',Edotmin)
            err=MakeWind([Edotmin,logTs0],logMdot,Verbose=Verbose)
            
            if Edotmin>1.1:
                sys.exit('Inadequate initial Ts0, exiting..')
                
        # Initial guesses (the second one just needs to be closer to the solution)
        z0,z1 = [Edotmin+0.002,logTs0] , [Edotmin+0.0025,logTs0-0.01]

    # It happens that z1 does not work..
    err = MakeWind(z1,logMdot,Verbose=Verbose)
    i=1
    while 100 in err:
        print('z1 inadequate (no solution)')
        z1 = [Edotmin+0.0025 , logTs0 + (-1)**i *i*0.01] # Alternating between going below and above Ts0
        err = MakeWind(z1,logMdot,Verbose=Verbose)
        i += 1  
        if i == 10:
            sys.exit('Inadequate initial Ts0, exiting..')

    # Check that first jacobian to compute is non-singular
    J = Jacobian(MakeWind,z0,z1,logMdot)
    i=1
    while np.linalg.det(J)==0.0:
        print("z1 inadequate (singular jacobian)")
        z1 = [Edotmin+0.0025 , logTs0 + (-1)**i *i*0.01] # Alternating between going below and above Ts0
        J = Jacobian(MakeWind,z0,z1,logMdot)
        i += 1  
        if i == 10:
            sys.exit('Inadequate initial Ts0, exiting..')

    limits = [1,1.1,6.5,8]
    print('\nStarting Rootfinding with first two iterations of (Edot,Ts) : ',z0,z1)
    root = Newton_Raphson_2D(MakeWind,z0,z1,limits,logMdot,flagcatch=100)

    return root




import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
from IO import save_root,clean_rootfile


# A single root
logMdot = 17.3
logTs0 = 7.4
root=RootFinder(logMdot,logTs0=logTs0,usefile=0)
save_root(logMdot,root)
    

## Mutliple roots

# logmdots = np.round(np.arange(19,17.1,-0.05),decimals=2)
# # logMDOTS = np.arange(18.05,19,0.1)
# # logMDOTS = np.arange(17.95,17,-0.05)
# roots = []
# problems = []

# usefile=0

# for logMDOT in logmdots:
   
#    logTs0=7.5 if logMDOT<18 else 7.1   # maybe don't need this anymore now that error2 doesnt have nans?
       
#    try:
#        root = RootFinder(logMDOT,logTs0=logTs0,usefile=usefile)
#        roots.append(root)
#        save_root(logMDOT,root)
#    except:
#        problems.append(logMDOT)
#        print('PROBLEM WITH LOGMDOT = ',logMDOT)
       
# print('There were problems for these values:')
# print(problems)
        

        
