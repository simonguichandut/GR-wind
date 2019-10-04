import sys
sys.path.append(".")

import numpy as np
from numpy.linalg import norm
from numpy import array
import pickle    
import matplotlib.pyplot as plt

from wind_GR import *

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
# warnings.filterwarnings("ignore", category=ODEintWarning) 


logMDOT = 17.25
from IO import load_roots
logMDOTS,roots = load_roots()
z = roots[logMDOTS.index(logMDOT)]


R, T, Rho, u, Phi, Lstar, L, LEdd_loc, E, P, cs, tau, rs, Edot, Ts = MakeWind(z,logMDOT,mode='wind')


def electronsfull(rho,T):  # From Paczynski (1983) semi-analytic formula : ApJ 267 315

    Ye = 0.5
    rY = rho*Ye
    pednr = 9.91e12 * (rho*Ye)**(5/3)     
    pedr = 1.231e15 * (rho*Ye)**(4/3)
    ped = 1/np.sqrt((1/pedr**2)+(1/pednr**2))
    pend = kB/mp*rY*T
    pe = np.sqrt(ped**2 + pend**2) # pressure

    f = 5/3*(ped/pednr)**2 + 4/3*(ped/pedr)**2
    Ue = pe/(f-1)
    
    Ue2 = 3/2*pe

    return pe,Ue,pednr,pedr,ped,pend,Ue2


pe,Ue,pednr,pedr,ped,pend,Ue2 = electronsfull(Rho,T)
prad = arad*T**4/3.0 
pgas = Rho*cs2(T)

fig,ax = plt.subplots(1,1)
ax.loglog(R,P,'k',lw=0.6,label='Total')
ax.loglog(R,prad,'r',lw=0.6,label='Radiation')
ax.loglog(R,pgas,'b',lw=0.6,label='Ideal gas')
ax.loglog(R,pe,'m',lw=0.6,label='Electrons')
ax.loglog(R,pend,'m--',lw=0.6,label='Electrons (non-deg)')
ax.loglog(R,ped,'m-.',lw=0.6,label='Electrons (deg)')

ax.legend()
ax.set_xlabel('r (cm)')
ax.set_ylabel('Pressure (dyne/cm2)')

ax2=ax.twinx()
ax2.loglog(R,Rho,'g-')
ax2.set_ylabel('Density (g/cm3)')
ax2.yaxis.label.set_color('g')
ax2.set_ylim([1e-3,1e5])

ax.set_title(r'$\dot{M}=10^{%s}$ g/s'%logMDOT)


fig2,ax2 = plt.subplots(1,1)
ax2.loglog(Rho,P,'k',lw=0.6,label='Total')
ax2.loglog(Rho,prad,'r',lw=0.6,label='Radiation')
ax2.loglog(Rho,pgas,'b',lw=0.6,label='Ideal gas')
ax2.loglog(Rho,pe,'m',lw=0.6,label='Electrons')
ax2.loglog(Rho,pend,'m--',lw=0.6,label='Electrons (non-deg)')
ax2.loglog(Rho,ped,'m-.',lw=0.6,label='Electrons (deg)')
ax2.loglog(Rho,pednr,'g-.',lw=0.6,label='Electrons (deg n.r)')
ax2.loglog(Rho,pedr,'g--',lw=0.6,label='Electrons (deg r.)')

ax2.legend()
ax2.set_xlabel('Density (g/cm3)')
ax2.set_ylabel('Pressure (dyne/cm2)')
ax2.set_title(r'$\dot{M}=10^{%s}$ g/s'%logMDOT)


plt.show()