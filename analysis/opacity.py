''' Checking importance of terms in opacity ''' 
import sys
sys.path.append(".")

import numpy as np
import matplotlib.pyplot as plt
from numpy import log10, pi, array, linspace, logspace, where, exp, argwhere
from scipy.interpolate import interp1d

from wind_GR import *
from IO import *

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

# Parameters
M, RNS, y_inner, tau_out, comp, mode, save, img = load_params()

# Constants
c = 2.99292458e10
h = 6.62e-27
kappa0 = 0.2
GM = 6.6726e-8*2e33*M
LEdd = 4*pi*c*GM/kappa0
ZZ = (1-2*GM/(c**2*RNS*1e5))**(-1/2) # redshift
g = GM/(RNS*1e5)**2 * ZZ



# Opacities 

# Thompson
def kappa_es1(T):
    return kappa0/(1.0+(2.2e-9*T)**0.86)

def kappa_es2(rho,T):
    return kappa0/( (1.0+(2.2e-9*T)**0.86) * (1+2.7e11*rho/T**2) )

# Free-free
def kappa_ff(rho,T):
    return 1e23 * 2**2/(4*2) * rho * T**(-7/2)


# Import Wind 10^18.1
logmdot = 18.1
logMDOTS,roots = load_roots()
root = roots[argwhere(array(logMDOTS)==logmdot)[0][0]]

R, T, Rho, u, Phi, Lstar, L, LEdd_loc, E, P, cs, tau, rs, Edot, Ts = MakeWind(root, logmdot, mode='wind')


kes1 = kappa_es1(T)
kes2 = kappa_es2(Rho,T)
kff = kappa_ff(Rho,T)

fig,ax = plt.subplots(1,1)
ax.semilogy(log10(Rho),kes2,label=r'$\kappa_{es}$')
ax.semilogy(log10(Rho),kes1,label=r"$\kappa_{es}'$")    
# ax.semilogy(Rho,kes1/kes2,label=r'ratio') 
ax.semilogy(log10(Rho),kff,label=r"$\kappa_{ff}$")  
ax.set_xlabel(r'log $\rho$')
ax.set_ylabel(r'$\kappa$')
ax.legend()


def logrho_to_logT(logrho):
    func = interp1d(log10(Rho),log10(T))
    logtemp = func(logrho)
    return ["%.1f" % z for z in logtemp]


top_ticks = [-5,-3,-1,1,3]

ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(top_ticks)
ax2.set_xticklabels(logrho_to_logT(top_ticks))

ax2.set_xlabel(r'log T')


# fig2,ax2=plt.subplots(1,1)
# plt.loglog(Rho,T)
plt.show()



