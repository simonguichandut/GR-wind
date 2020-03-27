import numpy as np
import sys
import os
import IO
from scipy.interpolate import InterpolatedUnivariateSpline as IUS 
from scipy.optimize import fsolve,brentq
from wind_GR_FLD import *

"""
Rootfinding is done differently in FLD than optically thick.  For every (Mdot,Edot), there is a single value of Ts 
that goes to infinity.  But it's not possible to find the exact value and shoot to infiniy.  Instead, for every Edot, 
we find two close (same to high precision) values of Ts that diverge in opposite direction, and declare either to be 
the correct one.  This gives a Edot-Ts relation, on which we can rootfind based on the inner boundary error.
"""

M, RNS, y_inner, tau_out, comp, EOS_type, FLD, mode, save, img = IO.load_params()

def run_outer(logMdot,Edot_LEdd,logTs,Verbose=0):  # full outer solution from sonic point
    global Mdot,Edot,Ts,verbose,rs
    Mdot, Edot, Ts, verbose = setup_globals([Edot_LEdd,logTs],logMdot,Verbose=Verbose,return_them=True)
    rs = rSonic(Ts)
    if rs<RNS and Verbose: print('warning: sonic point less than 12km')

    # High rmax to ensure we find two solutions that diverge separatly in rootfinding process
    return outerIntegration(r0=rs, T0=Ts, phi0=2.0, rmax=1e12)

def run_inner(logMdot,Edot_LEdd,logTs,Verbose=0,solution=False):  # full inner solution from sonic point
    global Mdot,Edot,Ts,verbose,rs
    Mdot, Edot, Ts, verbose = setup_globals([Edot_LEdd,logTs],logMdot,Verbose=Verbose,return_them=True)
    rs = rSonic(Ts)
    sol_inner1 = innerIntegration_r()
    T95,phi95 = sol_inner1.sol(0.95*rs)
    _,rho95,_,_ = calculateVars_phi(0.95*rs, T95, phi=phi95, subsonic=True)

    if solution:
        return innerIntegration_rho(rho95, T95, returnResult=True)
    else:
        return innerIntegration_rho(rho95, T95)



def get_TsEdotrel(logMdot,tol=1e-6,Verbose=0):

    # find the value of Ts that allow a solution to go to inf (to tol precision), for each value of Edot

    if Verbose: print('\nLOGMDOT = %.2f\n'%logMdot)

    Edotvals = linspace(1.01,1.05,15)
    Tsvals = []
    a,b = 6.1,8
    
    for Edot_LEdd in Edotvals:
        print('\nFinding Ts for Edot/LEdd = %.3f'%Edot_LEdd)

        logTsvals = np.linspace(a,b,10)
        cont = True

        while abs(b-a)>tol and cont:
            print('%.6f    %.6f'%(a,b))

            for logTs in logTsvals[1:]:
                if Verbose: print(logTs)

                print('Current: %.6f'%logTs,end="\r")

                res = run_outer(logMdot,Edot_LEdd,logTs,Verbose)
                if rs<RNS: cont = 0 # stop if rs smaller than RNS

                if res.status==1:
                    a = logTs
                elif res.status==0:
                    print('Reached end of integration interval (r=%.3e) without diverging!'%res.t[-1])
                    raise
                else:
                    b = logTs
                    break

            logTsvals = np.linspace(a,b,6)      

        # Take final sonic point temperature to be bottom value (the one that leads to Mach 1.  We know the real value is in between a and a+tol)
        # Tsvals.append(a)

        if cont==0:
            print('At this Edot (and higher), the Ts required to go to inf is too high (sonic point smaller than NS radius). Exiting here.')
        elif a==b:
            print('border values equal (did not hit rs<RNS, maybe allow higher Ts). Exiting')
            break

        # Save one at a time
        IO.save_EdotTsrel(logMdot,[Edot_LEdd],[a],[b])

        a,b = a,8  # next Edot, Ts will certainly be higher than this one
        print('ok'.ljust(20))

    # Or save all at the end
    # IO.save_EdotTsrel(logMdot,Edotvals,Tsvals,np.array(Tsvals)+tol)

# get_TsEdotrel(17.75)


def RootFinder(logMdot,checkrel=True,Verbose=False):

    """ Find the (Edot,Ts) pair that minimizes the error on the inner boundary condition """

    print('\nStarting root finding algorithm for logMdot = %.2f'%logMdot)

    # Check if Edot,Ts file exists
    rel = IO.load_EdotTsrel(logMdot)
    if rel[0] is False:
        print('Edot-Ts relation file does not exist, creating..')
        get_TsEdotrel(logMdot,Verbose=Verbose)
        rel = IO.load_EdotTsrel(logMdot)
        print('\nDone!')

    IO.clean_EdotTsrelfile(logMdot)
    if Verbose: print('Loaded Edot-Ts relation from file')
    _,Edotvals,TsvalsA,TsvalsB = rel

    # Check if file is correct, i.e the two Ts values diverge in different directions
    if checkrel:
        print('Checking if relation is correct')
        for i in (0,-1):
            sola = run_outer(logMdot,Edotvals[i],TsvalsA[i])
            solb = run_outer(logMdot,Edotvals[i],TsvalsB[i])
            if sola.status == solb.status:
                print('Problem with EdotTsrel file at Edot/LEdd=%.3f ,logTs=%.3f'%(Edotvals[i],TsvalsA[i]))
                print(sola.message)
                print(solb.message)
                sys.exit()
        print(' EdotTsrel file ok')



    # Now do a 1D search on the interpolated line based on the inner BC error
    rel_spline = IUS(Edotvals,TsvalsA)
    
    def Err(Edot_LEdd):
        if isinstance(Edot_LEdd,np.ndarray): Edot_LEdd=Edot_LEdd[0]
        logTs=rel_spline(Edot_LEdd)
        E = run_inner(logMdot,Edot_LEdd,logTs)
        print("Looking for root... Edot/LEdd=%.6f \t logTs=%.6f \t Error=%.6f"%(Edot_LEdd,logTs,E),end="\r")
        return E


    # Testing
    # print(Err(1.025))
    # print(Err((Edotvals[0]+Edotvals[-1])/2))

    if Verbose: print('Searching root on Edot,Ts relation based on inner boundary error')

    # Check if root is present in the interval
    erra = Err(Edotvals[0])
    errb = Err(Edotvals[-1])
    if erra*errb > 0: #same sign
        print('No root in the interval')
        raise

    else:
        x = brentq(Err,Edotvals[0],Edotvals[-1])
        root = [x,rel_spline(x).item(0)]
        print('Found root : ',root,'. Error on NS radius: ',Err(x))
        return root

# RootFinder(18,checkrel=True)




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

        try:
            root = RootFinder(logMDOT,checkrel=False)
            roots.append(root)
            success.append(logMDOT)
            save_root(logMDOT,root)
        except:
            problems.append(logMDOT)
            print('\nPROBLEM WITH LOGMDOT = ',logMDOT,'\nTrying again with verbose...\n\n')
            try : RootFinder(logMDOT,checkrel=True,Verbose=True)
            except: pass
    
        
    print('\n\n*********************  SUMMARY *********************')
    print('Found roots for these values :',success)
    print('There were problems for these values :',problems)

    if len(success)>=1 and input('\nClean (overwrite) updated root file? (0 or 1) '):
        clean_rootfile(warning=0)
    print('\n\n')
    
    

# Command line call
if len(sys.argv)>1:
    
    if sys.argv[1]!='all' and ' ' in sys.argv[1]:          
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
        
        

