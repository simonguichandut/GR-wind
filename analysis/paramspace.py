import sys
sys.path.append(".")

# my idea is to understand what happens in different parts of the Ts,Edot space
# for example where does the sonic point become smaller than RNS
# what is the progression of the Edot-Ts curves that allow integration to inf

import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import IO 
params = IO.load_params()

# currently only with FLD on
assert params['FLD'] == True

from wind_GR_FLD import setup_globals


def get_sonicpoints(logMdot):

    n=50
    sonicpoints = [[] for i in range(n)]
    Edotvals = np.linspace(1.01,1.05,n)
    logTsvals = np.linspace(6.5,8,n)
    
    for i,logTs in enumerate(logTsvals):   
        # print('\n\n LOG TS = %.3f \n\n'%logTs)
        for Edot_LEdd in Edotvals:
            try:
                Mdot,Edot,Ts,rs,_ = setup_globals([Edot_LEdd,logTs],logMdot,Verbose=0,return_them=True) 
                # print('Edot/LEdd=%.3f : logrs = %.3f'%(Edot_LEdd,np.log10(rs)))
            except:
                rs=1
                # print('Edot/LEdd=%.3f : ERROR'%(Edot_LEdd))

            sonicpoints[i].append(rs)
        
    return Edotvals,logTsvals,sonicpoints 

def plot_map(logMdot):

    Edotvals,logTsvals,sonicpoints = get_sonicpoints(logMdot)

    fig,ax = plt.subplots(1,1)    
    ax.patch.set_color('.25')

    cmap = plt.get_cmap('hot')

    plt.title(r'Sonic point radius for $\dot{M}=10^{%.1f}$'%logMdot)

    levs=np.logspace(5,9,50)
    im = ax.contourf(Edotvals,logTsvals,sonicpoints,levs,cmap=cmap,norm=colors.LogNorm())
    fig.colorbar(im,ax=ax,ticks=[1e5,1e6,1e7,1e8,1e9])

    ax.set_xlabel(r'$\dot{E}/L_{Edd}$')
    ax.set_ylabel(r'log $T_s$ (K)')

    ax.set_xlim([Edotvals[0],Edotvals[-1]])
    ax.set_ylim([logTsvals[0],logTsvals[-1]])  

    plt.show()

def get_RNSline(logMdot):
    # At every Edot, returns the value of Ts for which rs=RNS
    Edotvals = np.linspace(1.01,1.05,50)
    Edot_values, logTs_values = [],[]

    for Edot_LEdd in Edotvals:

        def err(logTs):
            _,_,_,rs,_ = setup_globals([Edot_LEdd,logTs],logMdot,Verbose=0,return_them=True) 
            return rs - params['R']*1e5

        logTsvals = np.linspace(7,9,100)
        logTskeep1,logTskeep2 = 0,0

        for logTs in logTsvals:
            e = err(logTs)
            if e>0:
                logTskeep1 = logTs            
            else:
                logTskeep2 = logTs
                break

        if logTskeep1!=0 and logTskeep2!=0:
            Edot_values.append(Edot_LEdd)
            logTs_values.append(brentq(err,logTskeep1,logTskeep2))
        
    return Edot_values,logTs_values

def plot_paramspace():

    ## Plot for all the Mdots the sonic point RNS line, the Edot-Ts relation
    # to integrate to infinity, and the roots

    fig,ax = plt.subplots(1,1)
    ax.set_xlim([1.01,1.05])
    ax.set_ylim([6.5,8.5])

    logMdots,roots = IO.load_roots()

    
    for logMdot in logMdots:

        if logMdot in (17.75,18,18.25,18.5,18.75,19):

            # Edot_Ts rel to integrate to infinity
            _,Edotvals,TsvalsA,TsvalsB = IO.load_EdotTsrel(logMdot)
            ax.plot(Edotvals,TsvalsA,'.-',label=str(logMdot))

            # Root
            root = roots[logMdots.index(logMdot)]

    ax.legend()

    plt.show()





