import sys
sys.path.append(".")

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from numpy import log10, pi, array, linspace, logspace, where, exp, argwhere

from wind_GR import *
from IO import *

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Parameters
M, RNS, y_inner, tau_out, comp, mode, save, img = load_params()

# Constants
c = 2.99792458e10
h = 6.62e-27
kappa0 = 0.2
GM = 6.6726e-8*2e33*M
LEdd = 4*pi*c*GM/kappa0
FEdd = LEdd/(4*pi*(RNS*1e5)**2)
ZZ = (1-2*GM/(c**2*RNS*1e5))**(-1/2)  # redshift
g = GM/(RNS*1e5)**2 * ZZ
arad = 7.5657e-15
P_inner = g*y_inner
Trad = (3*P_inner/arad)**(1/4)
# print('g/1e14 = ',g/1e14)

#######################################################################################################################################
# Data

# Import Winds data
clean_rootfile()
logMDOTS, roots = load_roots()

# Extracted terms
Lb,Tb,Lbs,rhob,Pb,Eb,ub,Rb,rsonic,uphot,Tphot,Lph,Lphs,rhophot,Rphot = [[] for i in range(15)]
stuff = (Lb,Tb,Lbs,rhob,Pb,Eb,ub,Rb,rsonic,uphot,Tphot,Lph,Lphs,rhophot,Rphot)

for logMdot in logMDOTS:
    R, u, cs, rho, T, P, phi, L, Lstar, E, tau, rs = read_from_file(logMdot)
    for a, b in zip(stuff, 
            (L[0],T[0],Lstar[0],rho[0],P[0],E[0],u[0],R[0],rs,u[-1],T[-1],L[-1],Lstar[-1],rho[-1],R[-1])):
        a.append(b)

Lb,Tb,Lbs,rhob,Pb,Eb,ub,Rb,rsonic,uphot,Tphot,Lph,Lphs,rhophot,Rphot = [array(a) for a in stuff]

# Base Luminosity
Fb = Lb/(4*pi*(RNS*1e5)**2)  # non-redshifted base flux
Fbs = Lbs/(4*pi*(RNS*1e5)**2)  # redshifted base flux
Mdots = 10**array(logMDOTS)
Edots = array([l[0] for l in roots])

# Contributions to Edot
w = (Eb+Pb)/rhob
Ek = (ub)**2/2
Eg = GM/RNS/1e5

Edots_GR = Lbs + Mdots*H_e(rhob,Tb)*Y(Rb,ub)
Edots_newt =  Lb + Mdots*(c**2+Ek+w-Eg)

#fig,ax=plt.subplots(1,1)
#ax.plot(logMDOTS,Edots_GR/1e38,label='GR',lw=1.5,c='r')
#ax.plot(logMDOTS,Edots_newt/1e38,label='Newtonian',lw=1.5,c='b')
#ax.set_xlabel(r'log $\dot{M}$ (g/s)',fontsize=14)
#ax.set_ylabel(r'$\dot{E}$ (10$^{38}$ erg s$^{-1}$)',fontsize=14)
#ax.legend(fontsize=14)
#plt.tight_layout()

#frac_L = Lb/Edots_newt
#frac_c2 = Mdots*c**2/Edots_newt
#frac_Ek = Mdots*Ek/Edots_newt
#frac_w = Mdots*w/Edots_newt
#frac_Eg = -Mdots*Eg/Edots_newt
#print(frac_L+frac_c2+frac_Ek+frac_w+frac_Eg) # should be 1

f = (Lbs-LEdd)/(Eg*Mdots)
YY=(1-2*GM/c**2/12e5)**(0.5)  # purely gravitaitonnal Y=(1+z)^(-1) , because (v/c)^2<<1

#######################################################################################################################################
#%% Energetics
plt.close('all')

from matplotlib.patches import Rectangle
#logmdots = (17.25,17.5,17.75,18,18.25,18.5,18.75,19)
logmdots = np.arange(17.2,19,0.1)

fig,ax = plt.subplots(1,1,figsize=(6,6))
ax.axhline(0,c='k',lw=0.5)
ax.set_xticks(np.arange(17.2,19,0.2))
ax.set_xlabel(r'log $\dot{M}$ (g/s)',fontsize=14)
#ax.set_title(r'$\dot{E}$ distribution',fontsize=14)

norm = Edots_newt[-1]
ax.set_ylim([-Eg*Mdots[-1]/norm-0.05,1+0.05])
ax.set_xlim([17,19.25])


def plot_one_column(logMdot,normalize='biggest'):
    i = logMDOTS.index(logMdot)
#    rect_grav = Rectangle((logMdot-0.1,-Eg*Mdots[i]/norm),0.2,0.005, facecolor='k')
    
    if normalize=='biggest':
        norm = Edots_newt[-1]
        fg,fc2,fL,fw,fk = -Eg*Mdots[i]/norm , Mdots[i]*c**2/norm , Lb[i]/norm , Mdots[i]*w[i]/norm, Mdots[i]*(ub[i])**2/2/norm
        ax.set_yticks((0,0.5e40/norm,1e40/norm))
        ax.set_yticklabels(('0',r'5$\times 10^{39}$','$10^{40}$'))
    elif normalize=='each':
        norm = Edots_newt
        fg,fc2,fL,fw,fk = -Eg*Mdots[i]/norm[i] , Mdots[i]*c**2/norm[i] , Lb[i]/norm[i] , Mdots[i]*w[i]/norm[i], Mdots[i]*(ub[i])**2/2/norm[i]
        ax.set_yticks((0,0.5,1))
#        ax.set_yticklabels(('0','0.5','1'))
    
    
    ax.plot([logMdot-0.04,logMdot+0.04],[fg,fg],color='b',label=r'$-GM\dot{M}/R$')
    rect_c2 = Rectangle((logMdot-0.04,fg),0.08,fc2, facecolor='gray',label='$\dot{M}c^2$')
    rect_L = Rectangle((logMdot-0.04,fg+fc2),0.08,fL, facecolor='r',alpha=0.5,label='$L$')
    rect_w = Rectangle((logMdot-0.04,fg+fc2+fL),0.08,fw, facecolor='g',alpha=0.5,label='$\dot{M}w$')
    rect_k = Rectangle((logMdot-0.04,fg+fc2+fL+fw),0.08,fk, facecolor='m',alpha=0.5,label='$\dot{M}u^2/2$')
    
    Y
    for rect in (rect_c2,rect_L,rect_w,rect_k):
        ax.add_patch(rect)
    
for i,x in enumerate(logmdots.round(decimals=1)):
#    normalize = 'each'
    normalize = 'biggest'
    plot_one_column(x,normalize=normalize)    
    if i==0 and normalize=='biggest': 
        leg = ax.legend(loc=2,fontsize=14)

#    r'$\dot{E}=L+\dot{M}(c^2+v^2/2+w-GM/r)$'

plt.tight_layout()

#%% factor f relation and fits
plt.close('all')


fig,ax = plt.subplots(1,1)
ax.set_xlabel(r'log $\dot{M}$ (g/s)',fontsize=14)
#ax.set_ylabel(r'$(L_b^*-L_{Edd})/(GM\dot{M}/R)$',fontsize=14)
ax.set_ylabel(r'$f$     ',fontsize=14,rotation='horizontal',ha='right')
#ax.set_title(r'$f=(L-L_{Edd})/(GM\dot{M}/R)$',fontsize=14)
ax.plot(logMDOTS,f,'k.')
ax.axhline(1,c='k',ls='--',lw=0.5)
ax.set_xlim([17.05,19.15])
plt.tight_layout()


fig2,ax2=plt.subplots(1,1)
ax2.set_xlabel(r'$\dot{M}$ (10$^{18}$ g/s)',fontsize=14)
ax2.set_ylabel(r'$\dot{E}$ (10$^{39}$ erg s$^{-1}$)',fontsize=14)
ax2.plot(Mdots/1e18,Edots_GR/1e39,'k.')
x=linspace(0,1e19,100)
plt.tight_layout()


# fit what seems to be almost linear
def fit_func(x,m):
    return LEdd + 1e21*m*x
from scipy.optimize import curve_fit
popt, pcov = curve_fit(fit_func, Mdots, Edots_GR)
print('Best linear fit : Edot = LEdd + %.3f x 10^(21) * Mdot'%(popt[0]))
f_fit1 = 1.1-YY*w/Eg
f_fit1b = 1.1-0*w
ax2.plot(x/1e18,fit_func(x,popt[0])/1e39,'b-',lw=0.5,label=(r'$\dot{E}=L_{Edd}+(%.2f\times10^{21})\dot{M}$')%(popt[0]))
#ax.plot(logMDOTS,f_fit1b,'k-',label='linear fit, w not included',lw=0.9)
#ax.plot(logMDOTS,f_fit1,'b-',label='linear fit, w included',lw=1)

ax.plot(logMDOTS,f_fit1,'b-',label='linear fit',lw=1)






## Maybe it's not quite linear?
#def fit_func2(x,m,a):
#    return LEdd + 1e21*m*x**a
#popt, pcov = curve_fit(fit_func2, Mdots[:-10], Edots_GR[:-10])
#m,a = popt
#print('Best non-linear fit : Edot = LEdd + %.3f x 10^(21) * Mdot^%.3f'%(m,a))
##ax2.plot(x/1e18,fit_func2(x,m,a)/1e39,'b-',lw=0.5,label=(r'$\dot{E}\sim L_{Edd}+(%.3f\times10^{21})\dot{M}^{%.3f}$')%(m,a))
#
#f_fit2 = (m*1e21/Eg)*Mdots**(a-1) - YY*(c**2+w)/Eg
##f_fit2 = 1/(Eg*Mdots)*(m*1e21*Mdots**a) - c**2*YY/Eg   # same thing
#ax.plot(logMDOTS,f_fit2,'r-',label='non-linear fit',lw=0.8)



# Maybe it is linear but zero is not exactly Eddington
#def fit_func3(x,b,m):
#    return b*LEdd + 1e21*m*x
#popt, pcov = curve_fit(fit_func3, Mdots[:-10], Edots_GR[:-10])
#b,m = popt
#print('Best linear, non-Edd zero fit : Edot = %.3fLEdd + %.3f x 10^(21) * Mdot'%(b,m))
#f_fit3 = m*1e21/Eg + (b-1)*LEdd/Eg/Mdots - YY*(c**2+w)/Eg 
##f_fit3 = 1/(Eg*Mdots)*(Edots_GR-LEdd) - YY*(c**2)/Eg 
#ax.plot(logMDOTS,f_fit3,'g-',label='linear fit, non-LEdd zero',lw=0.8)
#



### Combine all ideas
#def fit_func4(x,b,m,a):
#    return b*LEdd + 1e21*m*x**a
#popt, pcov = curve_fit(fit_func4, Mdots[:-10], Edots_GR[:-10])
#b,m,a = popt
#print('Best non-linear, non-Edd zero fit : Edot = %.3fLEdd + %.3f x 10^(21) * Mdot^%.3f'%(b,m,a))
#f_fit4 = m*1e21/Eg*Mdots**(a-1) + (b-1)*LEdd/Eg/Mdots - YY*(c**2+w)/Eg 
##f_fit3 = 1/(Eg*Mdots)*(Edots_GR-LEdd) - YY*(c**2)/Eg 
#ax.plot(logMDOTS,f_fit4,'m-',label='non-linear fit, non-LEdd zero',lw=0.8)
#
#

## Combine all ideas, but don't fit the last 10 points (where enthalpy takes over)
#def fit_func4(x,b,m,a):
#    return b*LEdd + 1e21*m*x**a
#popt, pcov = curve_fit(fit_func4, Mdots[:-12], Edots_GR[:-12])
#b,m,a = popt
#
#
#f_fit4 = m*1e21/Eg*Mdots**(a-1) + (b-1)*LEdd/Eg/Mdots - YY*(c**2+w)/Eg 
#f_fit4b = m*1e21/Eg*Mdots**(a-1) + (b-1)*LEdd/Eg/Mdots - YY*(c**2)/Eg 
#ax.plot(logMDOTS,f_fit4b,'k-',label='w not included',lw=0.7)
#ax.plot(logMDOTS,f_fit4,'b-',label='w included',lw=1)
#
#


#ax.legend()
ax2.legend(fontsize=13)




#%% Other f factors

f = (Lbs-LEdd)/(Eg*Mdots)
YY=(1-2*GM/c**2/12e5)**(0.5)
f2 = (Lbs-LEdd)/(Mdots*c**2*(1-YY))

fig,ax = plt.subplots(1,1)
ax.set_xlabel(r'log $\dot{M}$ (g/s)',fontsize=14)
#ax.set_ylabel(r'$(L_b^*-L_{Edd})/(GM\dot{M}/R)$',fontsize=14)
ax.set_ylabel(r'$f$     ',fontsize=14,rotation='horizontal',ha='right')
#ax.set_title(r'$f=(L-L_{Edd})/(GM\dot{M}/R)$',fontsize=14)
ax.plot(logMDOTS,f2,'k.')
ax.axhline(1,c='k',ls='--',lw=0.5)
ax.set_xlim([17.05,19.15])
plt.tight_layout()

plt.show()