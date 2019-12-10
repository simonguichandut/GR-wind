import numpy as np
from numpy.linalg import norm
from numpy import array
import sys
import os
import IO
from wind_GR import MakeWind


def Jacobian(func,za,zb,*args,**kwargs):

    ''' First order numerical jacobian of a two input, two output function '''

    dx,dy = zb[0]-za[0] , zb[1]-za[1]
    faa,fba,fab = func(za,*args,**kwargs) , func([zb[0],za[1]],*args,**kwargs) , func([za[0],zb[1]],*args,**kwargs) 
    J11 = (fba[0]-faa[0])/dx
    J12 = (fab[0]-faa[0])/dy
    J21 = (fba[1]-faa[1])/dx
    J22 = (fab[1]-faa[1])/dy
    
    return [[J11,J12],[J21,J22]]


def JacobianUpdate(J0,za,zb,fa,fb):
    
    ''' Jacobian update from Broyden's method '''
    
    J0,za,zb,fa,fb = array(J0),array(za),array(zb),array(fa),array(fb)
    dz,df = zb-za,fb-fa
    Jnew = J0 + 1/norm(dz)**2*np.outer(df-np.matmul(J0,dz),dz)
    return Jnew
    

def Newton_Raphson_2D(func,z0,z1,limits,*args,tol=1e-4,flagcatch=None,**kwargs):

    ''' Newton-Raphson root finding for a two input, two output function
        Two initial guesses have to be provided. Will soften iterations if 
        going out of bounds [xmin,xmax,ymin,ymax], or if obtaining invalid
        values. '''
    
    diff = 2*tol
    counter,f0,f1 = 0,func(z0,*args,**kwargs),func(z1,*args,**kwargs)
    Jold = [[0.00001,0.00001],[0.00001,-0.00001]]
    nitermax = 100
    while (diff>tol or norm(f1)>tol):

        try:
            J = Jacobian(func,z0,z1,*args,**kwargs)
        except:
            print('I was not able to calculate the Jacobian.  The function cannot evaluate at one of the points')       # To help me bugfix
            raise

        if counter != 0 and (True in np.isnan(J) or np.linalg.det(J)==0.0):
            print('Using Jacobian update')
            J = JacobianUpdate(Jold,z0,z1,f0,f1)
        
        pillow = 1
        znew = np.array(z1) - pillow*np.matmul(np.linalg.inv(J),f1)
        print('Trying new z : ',znew)
        
        #  Out of bounds. Catch level 1
        while znew[0]<limits[0] or znew[0]>limits[1] or znew[1]<limits[2] or znew[1]>limits[3] or (True in np.isnan(func(znew,*args,**kwargs))) :
            pillow/=3
            print('pillow update : ',pillow)
            znew = np.array(z1) - pillow*np.matmul(np.linalg.inv(J),f1)
            
        fnew = func(znew,*args,**kwargs)
            
        # solution space is not necesarily rectangular. Catch level 2
        if flagcatch is not None:
            while (not set(flagcatch).isdisjoint(fnew)) or (True in np.isnan(fnew)):
                print('NaN or flag caught in fnew : ',fnew)
                pillow/=3
                print('pillow update v2: ',pillow)
                znew = np.array(z1) - pillow*np.matmul(np.linalg.inv(J),f1)
                fnew = func(znew,*args,**kwargs)
                if pillow < 1e-20 : sys.exit('Fell into a poorly behaved region, exiting..')

        # Root is unstable : we're close but we keep oscilating around it. Catch level 3
        if norm(f1[0])<tol*10 or norm(f1[1])<tol*10: # i.e we're close
            print('getting close')
            while True in list((array(f1)*array(fnew))<0): # sign change
                pillow/=3
                print('pillow update v3 :',pillow)
                znew = np.array(z1) - pillow*np.matmul(np.linalg.inv(J),f1)
                fnew = func(znew,*args,**kwargs)


        
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


def RootFinder(logMdot,logTs0=7.4,box='on',verbose=0,usefile=1):  
    
    ''' Finds the error-minimizing set of parameters (Edot,Ts) for a wind 
        with a given Mdot '''

    print('\nStarting root finding algorithm for logMdot = %.2f'%logMdot)

    # First check if we can start from a root in file

    find_first_root = 1
    if usefile:
        
        try:
            logMDOTS,roots = IO.load_roots()

            if logMdot in logMDOTS:
                print('First root from file')
                z0 = roots[logMDOTS.index(logMdot)]
                
            elif round(logMdot+0.05,2) in logMDOTS:
                print('First root from file (adjacent Mdot)')
                z0 = roots[logMDOTS.index(round(logMdot+0.05,2))]
            
            elif round(logMdot-0.05,2) in logMDOTS:
                print('First root from file (adjacent Mdot)')
                z0 = roots[logMDOTS.index(round(logMdot-0.05,2))]
                
            z1 = [z0[0]+0.001,z0[1]+0.01]
            find_first_root = 0

        except:
            print('Root file does not exist.  Will try to find an appropriate root to start from.')

    if find_first_root:
                    
        Edotmin = 1.01 # in units of LEdd. There are likely no solutions with a lower Edot
        # But this Edot might not converge
        err=MakeWind([Edotmin,logTs0],logMdot,Verbose=verbose)
        
        while not set((100,200,300,400)).isdisjoint(err):
            Edotmin += 0.002
            print('\nEdotmin: ',Edotmin)
            err=MakeWind([Edotmin,logTs0],logMdot,Verbose=verbose)
            
            if Edotmin>1.1:
                sys.exit('Inadequate initial Ts0, exiting..')
                
        # Initial guesses 
        z0,z1 = [Edotmin+0.002,logTs0] , [Edotmin+0.0025,logTs0-0.01]

    # It happens that z1 does not work..
    err = MakeWind(z1,logMdot,Verbose=verbose)
    i=1
    while not set((100,200,300,400)).isdisjoint(err):
        print('z1 inadequate (no solution)')
        z1 = [Edotmin+0.0025 , logTs0 + (-1)**i *i*0.01] # Alternating between going below and above Ts0
        err = MakeWind(z1,logMdot,Verbose=verbose)
        i += 1  
        if i == 10:
            sys.exit('Inadequate initial Ts0, exiting..')

    # Check that first jacobian to compute is non-singular
    J = Jacobian(MakeWind,z0,z1,logMdot,Verbose=verbose)
    i=1
    while np.linalg.det(J)==0.0:
        print("z1 inadequate (singular jacobian)")
        z1 = [Edotmin+0.0025 , logTs0 + (-1)**i *i*0.01] # Alternating between going below and above Ts0
        J = Jacobian(MakeWind,z0,z1,logMdot,Verbose=verbose)
        i += 1  
        if i == 10:
            print('Inadequate initial Ts0, exiting..')
            raise ValueError

    limits = [1,1.1,6.5,8]
    print('\nStarting Rootfinding with first two iterations of (Edot,Ts) : ',z0,z1)
    root = Newton_Raphson_2D(MakeWind,z0,z1,limits,logMdot,flagcatch=(100,200,300,400),Verbose=verbose)

    return root




###################################### Driver ########################################

def driver(logmdots,usefile=1):

    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    from IO import save_root,clean_rootfile

    if logmdots == 'all':
        logmdots = np.round(np.arange(19,17,-0.05),decimals=2)

    roots = []
    problems,success = [],[]

    for logMDOT in logmdots:
    
        logTs0=7.5 if logMDOT<18 else 7.1       # An educated guess, works well for helium. Might have to tune or just leave as user input.  Don't like magic numbers
        
        try:
            root = RootFinder(logMDOT,logTs0=logTs0,usefile=usefile)
            roots.append(root)
            success.append(logMDOT)
            save_root(logMDOT,root)
        except:
            problems.append(logMDOT)
            print('\nPROBLEM WITH LOGMDOT = ',logMDOT,'\nTrying again with verbose...\n\n')
            try : RootFinder(logMDOT,logTs0=logTs0,usefile=usefile,verbose=1)
            except: pass
        
    print('\n\n*********************  SUMMARY *********************')
    print('Found roots for these values :',success)
    print('There were problems for these values :',problems)

    if len(success)>=1 and input('\nClean (overwrite) updated root file? (0 or 1) '):
        clean_rootfile(warning=0)
    print('\n\n')
    
    


# Command line call
if len(sys.argv)>1:
    
    if sys.argv[1]!='all' and ' ' in sys.argv[1]:           # probably need a better parser..
        sys.exit('Give logmdots as a,b,c,...')

    if sys.argv[1]=='all':
        logmdots='all'
    elif ',' in sys.argv[1]:
        logmdots = eval(sys.argv[1])
    else:
        logmdots = [eval(sys.argv[1])]

    if len(sys.argv)<3:
        driver(logmdots)
    else:
        if sys.argv[2]=='1' or sys.argv[2]=='True':
            driver(logmdots, usefile = True)
        else:
            driver(logmdots, usefile = False)
        
        

