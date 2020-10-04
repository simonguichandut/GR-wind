import numpy as np
import sys
import os
import IO
from scipy.interpolate import InterpolatedUnivariateSpline as IUS 
from scipy.optimize import fsolve,brentq
from wind_GR_FLD import *

# import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning) 
from IO import save_root,clean_rootfile

"""
Rootfinding is done differently in FLD than optically thick.  For every (Mdot,Edot), there is a single value of Ts 
that goes to infinity.  But it's not possible to find the exact value and shoot to infiniy.  Instead, for every Edot, 
we find two close (same to high precision) values of Ts that diverge in opposite direction, and declare either to be 
the correct one.  This gives a Edot-Ts relation, on which we can rootfind based on the inner boundary error.
"""

# M, RNS, y_inner, tau_out, comp, EOS_type, FLD, mode, save, img = IO.load_params(as_dict=False)

def run_outer(logMdot,Edot_LEdd,logTs,Verbose=0):  # full outer solution from sonic point
    global Mdot,Edot,Ts,verbose,rs
    Mdot, Edot, Ts, rs, verbose = setup_globals([Edot_LEdd,logTs],logMdot,Verbose=Verbose,return_them=True)
    
    if rs<IO.load_params()['R']*1e5: 
        raise Exception('sonic point less than RNS')

    # High rmax to ensure we find two solutions that diverge separatly in rootfinding process
    return outerIntegration(r0=rs, T0=Ts, phi0=2.0, rmax=1e11)

def run_inner(logMdot,Edot_LEdd,logTs,Verbose=0,solution=False):  # full inner solution from sonic point
    global Mdot,Edot,Ts,verbose,rs
    Mdot, Edot, Ts, rs, verbose = setup_globals([Edot_LEdd,logTs],logMdot,Verbose=Verbose,return_them=True)
    rs = rSonic(Ts)
    sol_inner1 = innerIntegration_r()
    T95,phi95 = sol_inner1.sol(0.95*rs)
    _,rho95,_,_ = calculateVars_phi(0.95*rs, T95, phi=phi95, subsonic=True)

    if solution:
        return innerIntegration_rho(rho95, T95, returnResult=True)
    else:
        return innerIntegration_rho(rho95, T95)



def get_TsEdotrel(logMdot,tol=1e-5,Verbose=0,Edotmin=1.01,Edotmax=1.04,npts=10):

    # find the value of Ts that allow a solution to go to inf (to tol precision), for each value of Edot

    if Verbose: print('\nLOGMDOT = %.2f\n'%logMdot)

    Edotvals = np.linspace(Edotmin,Edotmax,npts)
    Edotvals = np.round(Edotvals,10)
    
    Tsvals = []
    cont = True

    if IO.load_EdotTsrel(logMdot)[0] is True: #relation already exists, will use it to predict initial a,b bounds on Ts
        _,Edotrel,Tsrel,_ = IO.load_EdotTsrel(logMdot)

        if Edotrel[0]>Edotmin:
            b=Tsrel[0]
            a=b-0.5
        else:
            ix= [i for i in range(len(Edotrel)) if Edotrel[i]<Edotmin][-1] # this gives the index of the Edot in Edotvals which is closest (but lower than) Edotmin
            a = Tsrel[ix]
            b = a+0.5
        npts_Ts = 50 # if we are here we are really refining the param space

    else:
        a,b = 6.1,8
        nps_Ts=10

    for Edot_LEdd in Edotvals:
        print('\nFinding Ts for Edot/LEdd = %.10f'%Edot_LEdd)

        logTsvals = np.linspace(a,b,npts_Ts)
        logTsvals = np.round(logTsvals,9)

        while abs(b-a)>tol and cont:
            print('%.8f    %.8f'%(a,b))

            for logTs in logTsvals[1:-1]:

                print('Current: %.8f'%logTs, end="\r")

                try:
                    res = run_outer(logMdot,Edot_LEdd,logTs,Verbose)
                except Exception as E:
                    print(E)
                    print('Exiting...')
                    cont = False
                    break

                else:
                    if res.status==1:
                        a = logTs
                    elif res.status==0:
                        raise Exception('Reached end of integration interval (r=%.3e) without diverging!'%res.t[-1])
                    else: # res.status=-1
                        b = logTs
                        break

            logTsvals = np.linspace(a,b,6)      

            # To save time, check that where we are about to search is worth it.
            # We can take previous roots, if they exist, interpolate Edot and Ts
            # and evaluate the interpolation at the current Edot. If the interval
            # Ts interval [a,b]that we are checking now is very far from the 
            # interpolation, we can exit here to save time
            try:
                logMdots,roots = IO.load_roots()
                elts = [i for i in range(len(logMdots)) if logMdots[i]>logMdot]
                if len(elts)>=2:
                    x1,x2 = logMdots[elts[0]], logMdots[elts[1]]
                    y1,y2 = roots[elts[0]][1], roots[elts[1]][1] # grabbing the Ts vals
                    # let's interpolate a line to predict logTs for our Mdot
                    y = (y2-y1)/(x2-x1) * (logMdot-x1) + y1
                    # Now check if our bottom bound (a) is very far from the prediction)
                    # in logspace, 0.5 is very far, too far for the root to be there
                    if a-y>0.5:
                        print('Bottom Ts (%.2f) too far from where the root will realistically be (prediction from two other Mdots is logTs=%.2f'%(a,y))
                        cont=False
                        break
            except:
                pass

        # Take final sonic point temperature to be bottom value (the one that leads to Mach 1.  We know the real value is in between a and a+tol)
        # Tsvals.append(a)

        if cont==False:
            break

        if a==b:
            print('border values equal (did not hit rs<RNS, maybe allow higher Ts). Exiting')
            break

        # Save one at a time
        IO.save_EdotTsrel(logMdot,[Edot_LEdd],[a],[b])

        a,b = a,8  # next Edot, Ts will certainly be higher than this one
        print('ok'.ljust(20))

    IO.clean_EdotTsrelfile(logMdot,warning=0)



def RootFinder(logMdot,checkrel=True,Verbose=False,depth=1):

    """ Find the (Edot,Ts) pair that minimizes the error on the inner boundary condition """

    print('\nStarting root finding algorithm for logMdot = %.2f'%logMdot)

    # Check if Edot,Ts file exists
    rel = IO.load_EdotTsrel(logMdot)
    if rel[0] is False:
        print('Edot-Ts relation file does not exist, creating..')

        if logMdot < 18.0:
            # At low Mdots (~<18, high Edots go to high Ts quickly, hitting
            # the rs = RNS line and causing problems in rootfinding)
            get_TsEdotrel(logMdot,Verbose=Verbose,Edotmax=1.03)
        else:
            get_TsEdotrel(logMdot,Verbose=Verbose)

        rel = IO.load_EdotTsrel(logMdot)
        print('\nDone!')

    
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

        print(' EdotTsrel file ok')

    # Now do a 1D search on the interpolated line based on the inner BC error
    if len(Edotvals)<=3: # need at least 4 points for cubic spline
        print('Not enough points to interpolate spline, re-interpolating')
        if len(Edotvals)==1:
            get_TsEdotrel(logMdot,Edotmin=Edotvals[0]-0.001,Edotmax=Edotvals[0]+0.001,npts=5)
        else:
            get_TsEdotrel(logMdot,Edotmin=Edotvals[0]-0.001,Edotmax=Edotvals[-1]+0.001,npts=5)
        raise Exception('Call Again')

    else:
        rel_spline = IUS(Edotvals,TsvalsA)
    
    def Err(Edot_LEdd):
        if isinstance(Edot_LEdd,np.ndarray): Edot_LEdd=Edot_LEdd[0]
        logTs=rel_spline(Edot_LEdd)
        E = run_inner(logMdot,Edot_LEdd,logTs)
        print("Looking for root... Edot/LEdd=%.6f \t logTs=%.6f \t Error=%.6f"%(Edot_LEdd,logTs,E),end="\r")
        return E


    if Verbose: print('Searching root on Edot,Ts relation based on inner boundary error')

    erra = Err(Edotvals[0])
    for Edot in Edotvals[:0:-1]: # cycle through the values in reverse order until the 2nd one
        errb = Err(Edot)
        if erra*errb < 0: # different sign, means a root is in the interval
            print('\nroot present')
            break

    if erra*errb > 0: # same sign (we'll enter this if we didn't break last loop)
        diff = Edotvals[1]-Edotvals[0]
        if erra<0:
            print('\nOnly negative errors (rb<RNS)') # need smaller Ts (smaller Edot)
            get_TsEdotrel(logMdot,Edotmin=Edotvals[0]-diff,Edotmax=Edotvals[0]-1e-8,npts=5*depth)
        else:
            print('\nOnly positive errors (rb>RNS)') # need higher Ts (higher Edot)
            get_TsEdotrel(logMdot,Edotmin=Edotvals[-1]+1e-8,Edotmax=Edotvals[-1]+diff,npts=5*depth)
#        raise Exception('No root in the interval')
        raise Exception('Call Again')

    else:
        x = brentq(Err,Edotvals[0],Edot) # Edot is the last value before break above
        root = [x,rel_spline(x).item(0)]
        print('Found root : ',root,'. Error on NS radius: ',Err(x))
        return root


def ImproveRoot(logMdot, eps=0.001):

    ''' Rootfinding with the inner b.cond is done on a spline interpolation of 
    the Edot-Ts relation. When that root is obtained, it's not exact because of 
    interpolation errors, meaning that root can't be carried out to infinity. 
    In the main code (wind_GR_FLD), a new Ts bound is found to do the bisection,
    which changes the sonic point. At low mdot (or in general when the Edot-Ts)
    relation doesn't have enough points, the change in sonic point can be 
    significant enough that the base (r(y8)) is not close at all to RNS anymore.
    The purpose of this function is to resolve the Edot-Ts relation around the 
    root found initially and then re-do the rootfinding.
    '''

    logMdots,roots = IO.load_roots()
    if logMdot not in logMdots:
        sys.exit("root doesn't exist yet")
    root = roots[logMdots.index(logMdot)]

    # Search between the two Edot values that bound the root
    _,Edots,_,_ = IO.load_EdotTsrel(logMdot)
    i = np.argwhere(np.array(Edots)>root[0])[0][0]
    diff = Edots[i]-Edots[i-1]
    bound1 = Edots[i-1] + 0.1*diff
    bound2 = Edots[i] - 0.1*diff

    get_TsEdotrel(logMdot,Edotmin=bound1,Edotmax=bound2,npts=8)
    #get_TsEdotrel(logMdot,Edotmin=root[0]-eps,Edotmax=root[0]+eps,npts=8)
    IO.clean_EdotTsrelfile(logMdot,warning=0)
    root = RootFinder(logMdot,checkrel=False)
    IO.save_root(logMdot,root)
    IO.clean_rootfile(warning=0)



###################################### Driver ########################################

def recursor(logMdot, depth=1, max_depth=5):
    # will call RootFinder many times by recursion, as long as it returns the "Call Again" exception

    if depth==max_depth:
        print('Reached max recursion depth')
        return None 

    try:
        root = RootFinder(logMdot,checkrel=False,depth=depth)
        return root

    except Exception as E:

        if E.__str__() == 'Call Again':
            print('\nGoing into recursion depth %d'%(depth+1))
            root = recursor(logMdot, depth=depth+1)
            return root

        else:
            print(E)
            raise Exception('Problem')


def driver(logmdots):

    if logmdots == 'all':
        logmdots = np.round(np.arange(19,17,-0.1),decimals=1)

    success, max_recursed, problems = [],[],[]

    for logMdot in logmdots:

        try: 
            root = recursor(logMdot)

            if root is None:
                max_recursed.append(logMdot)

            else:
                success.append(logMdot)
                save_root(logMdot,root)

        except:
                problems.append(logMdot)
                print('\n',e)
                print('\nPROBLEM WITH LOGMDOT = ',logMdot,'\nTrying again with verbose and checking EdotTs rel...\n\n')
                try : RootFinder(logMDOT,checkrel=True,Verbose=True)
                except: pass


    print('\n\n*********************  SUMMARY *********************')
    print('Found roots for :',success)
    print('Reached recursion limit trying to find roots for :',max_recursed)
    print('There were problems for :',problems)

    if len(success)>=1: #and input('\nClean (overwrite) updated root file? (0 or 1) '):
       clean_rootfile(warning=0)
    


# def driver(logmdots,usefile=1, recursion_depth=0):

#     recursion_flag = 0 if recursion_depth==0 else 1

#     roots = []
#     problems,success = [],[]

#     if recursion_depth>5:
#         print('Reached recursion limit, exiting. Did not find root for logMdot=',logmdots)
#     if recursion_depth<5:

#         if logmdots == 'all':
#             logmdots = np.round(np.arange(19,17,-0.1),decimals=1)
    
#         for logMDOT in logmdots:
    
#             try:
#                 root = RootFinder(logMDOT,checkrel=False)
#                 roots.append(root)
#                 success.append(logMDOT)
#                 save_root(logMDOT,root)
#             except Exception as e:
    
#                 if e.__str__() == 'Call Again':
#                     print('Calling again.. (recursion depth %d)'%(recursion_depth+1))
#                     driver([logMDOT],recursion_depth=recursion_depth+1)
# #                    recursion_flag = 1
    
#                 else:
#                     problems.append(logMDOT)
#                     print('\n',e)
#                     print('\nPROBLEM WITH LOGMDOT = ',logMDOT,'\nTrying again with verbose and checking EdotTs rel...\n\n')
#                     try : RootFinder(logMDOT,checkrel=True,Verbose=True)
#                     except: pass
            
#     if recursion_flag == 0:  
#         # This will only print after the mother loop is over (won't print at each recursion. just for aesthetics :)
#         print('\n\n*********************  SUMMARY *********************')
#         print('Found roots for these values :',success)
#         print('There were problems for these values :',problems)
    
#     if len(success)>=1: #and input('\nClean (overwrite) updated root file? (0 or 1) '):
#        clean_rootfile(warning=0)
    
    

# Command line call
if __name__ == "__main__":
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
            
            

