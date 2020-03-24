import numpy as np
import sys
import os
import IO
from wind_GR_FLD import *


"""
Rootfinding is done differently in FLD than optically thick.  For every (Mdot,Edot), there is a single value of Ts 
that goes to infinity.  But it's not possible to find the exact value and shoot to infiniy.  Instead, for every Edot, 
we find two close (same to high precision) values of Ts that diverge in opposite direction, and declare either to be 
the correct one.  This gives a Edot-Ts relation, on which we can rootfind based on the inner boundary error.
"""

def run_outer(logMdot,Edot_LEdd,logTs):
    global Mdot,Edot,Ts,verbose,rs
    Mdot, Edot, Ts, verbose = setup_globals([Edot_LEdd,logTs],logMdot,Verbose=0,return_them=True)
    rs = rSonic(Ts)
    return outerIntegration(returnResult=True)


def get_TsEdotrel(logMdot,tol=1e-6):

    # find the value of Ts that allow a solution to go to inf (to tol precision), for each value of Edot

    print('\nLOGMDOT = %.2f\n'%logMdot)

    Edotvals = linspace(1.01,1.05,10)
    Tsvals = []
    
    for Edot_LEdd in Edotvals:

        print('\nFinding Ts for Edot/LEdd = %.3f'%Edot_LEdd)

        a,b = 6.3,7.8
        logTsvals = np.linspace(a,b,15)

        while abs(b-a)>tol:

            print('%.6f\t%.6f'%(a,b))
        
            for logTs in logTsvals:
            
                # global Mdot,Edot,Ts,verbose,rs
                # Mdot, Edot, Ts, verbose = setup_globals([Edot_LEdd,logTs],logMdot,Verbose=0,return_them=True)
            
                # rs = rSonic(Ts)
                # res = outerIntegration(returnResult=True)

                res = run_outer(logMdot,Edot_LEdd,logTs)

                if res.status==1:
                    a = logTs
                else:
                    b = logTs
                    break

            logTsvals = np.linspace(a,b,5)      

        if a==b:
            print('Edot likely too high? Exiting')
            break

        # Take final sonic point temperature to be bottom value (the one that leads to Mach 1.  We know the real value is in between a and a+tol)
        # Tsvals.append(a)

        # Save one at a time
        IO.save_EdotTsrel(logMdot,[Edot_LEdd],[a],[b])

    # Or save all at the end
    # IO.save_EdotTsrel(logMdot,Edotvals,Tsvals,np.array(Tsvals)+tol)


# Edotvals,Tsvals = get_TsEdotrel(17.75)
# print(Edotvals)
# print('\n\n')
# print(Tsvals)


def RootFinder(logMdot,checkrel=True):

    """ Find the (Edot,Ts) pair that minimizes the error on the inner boundary condition """

    # Check if Edot,Ts file exists

    rel = IO.load_EdotTsrel(logMdot)
    if rel[0] is False:
        print('Edot-Ts relation file does not exist, creating..')
        get_TsEdotrel(logMdot)
        rel = IO.load_EdotTsrel(logMdot)

    print('Loaded Edot-Ts relation from file')
    _,Edotvals,TsvalsA,TsvalsB = rel

    # Check if file is correct, i.e the two Ts values diverge in different directions
    if checkrel:
        print('Checking if relation is correct')
        for i in (0,-1):
            sola = run_outer(logMdot,Edotvals[i],TsvalsA[i])
            solb = run_outer(logMdot,Edotvals[i],TsvalsB[i])
            if sola.status == solb.status:
                sys.exit('Problem with EdotTsrel file')
        print(' EdotTsrel file ok')


RootFinder(18)




###################################### Driver ########################################

# def driver(logmdots,usefile=1):

#     import warnings
#     warnings.filterwarnings("ignore", category=RuntimeWarning) 
#     from IO import save_root,clean_rootfile

#     if logmdots == 'all':
#         logmdots = np.round(np.arange(19,17,-0.05),decimals=2)

#     roots = []
#     problems,success = [],[]

#     for logMDOT in logmdots:
    

        
#     print('\n\n*********************  SUMMARY *********************')
#     print('Found roots for these values :',success)
#     print('There were problems for these values :',problems)

#     if len(success)>=1 and input('\nClean (overwrite) updated root file? (0 or 1) '):
#         clean_rootfile(warning=0)
#     print('\n\n')
    
    

# # Command line call
# if len(sys.argv)>1:
    
#     if sys.argv[1]!='all' and ' ' in sys.argv[1]:           # probably need a better parser..
#         sys.exit('Give logmdots as a,b,c,...')

#     if sys.argv[1]=='all':
#         logmdots='all'
#     elif ',' in sys.argv[1]:
#         logmdots = eval(sys.argv[1])
#     else:
#         logmdots = [eval(sys.argv[1])]

#     if len(sys.argv)<3:
#         driver(logmdots)
#     else:
#         if sys.argv[2]=='1' or sys.argv[2]=='True':
#             driver(logmdots, usefile = True)
#         else:
#             driver(logmdots, usefile = False)
        
        

