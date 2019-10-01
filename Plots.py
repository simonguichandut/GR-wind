''' Plotting various quantities from wind solutions ''' 

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
from numpy import log10, pi, array, linspace, logspace, floor

rc('text', usetex = True)
# rc('font', family = 'serif')
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
mpl.rcParams.update({'font.size': 15})

from wind_GR import *
from IO import *

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

# Parameters
M, RNS, y_inner, tau_out, comp, mode, save, img = load_params()

# constants
c = 2.99792458e10
kappa0 = 0.2
GM = 6.6726e-8*2e33*M
LEdd = 4*pi*c*GM/kappa0
g = GM/(RNS*1e5)**2

# Make directories for data and plots
make_directories()

# Import Winds
clean_rootfile()
logMDOTS,roots = load_roots()


save=0


########## PLOTS ###########
def set_style():
    plt.style.use(['seaborn-talk'])
def beautify(fig,ax):
    set_style()
    ax.tick_params(which='both',direction='in')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    fig.tight_layout()


# Radius-Luminosity (fig. 2)
fig1, ax1 = plt.subplots(1, 1)
ax1.set_xlabel(r'$r$ (cm)')
ax1.set_ylabel(r'$L_{\infty}/L_{Edd}$')    

# Radius-Temperature (fig. 4)
fig2, ax2 = plt.subplots(1, 1)
ax2.set_xlabel(r'$r$ (cm)')
ax2.set_ylabel(r'$T$ (K)')

# Density-Temperature (fig. 5)
fig3, ax3 = plt.subplots(1, 1)
ax3.set_xlabel(r'$\rho$ (g cm$^{-3}$)')
ax3.set_ylabel(r'$T$ (K)')

# Radius-Velocity (fig. 6)
fig4, ax4 = plt.subplots(1, 1)
ax4.set_xlabel(r'$r$ (cm)')
ax4.set_ylabel(r'$v$ (cm s$^{-1}$)')

# Radius-Pressure
fig5, ax5 = plt.subplots(1, 1)
ax5.set_xlabel(r'$r$ (cm)')
ax5.set_ylabel(r'$P$ (g cm$^{-1}$ s$^{-2}$)')

# Density-Opacity
fig7, ax7 = plt.subplots(1, 1)
ax7.set_xlabel(r'$\rho$ (g cm$^{-3}$)')
ax7.set_ylabel(r'$\kappa$')

# Density-Optical depth
fig8, ax8 = plt.subplots(1, 1)
ax8.set_xlabel(r'$\rho$ (g cm$^{-3}$)')
ax8.set_ylabel(r'$\tau$')
ax8.axhline(1)

Lbs = []  # base luminosity, redshifted to infinity
colors = ['r', 'b', 'g', 'k', 'm']

i = 0
for logMdot, root in zip(logMDOTS, roots):

    print(logMdot)
    global Mdot, verbose
    Mdot, verbose = 10**logMdot, 0

    R, T, Rho, u, Phi, Lstar, L, LEdd_loc, E, P, cs, tau, rs, Edot, Ts = MakeWind(
        root, logMdot, mode='wind')

    Lbs.append(Lstar[0])

    if save:
        data = [R, T, Rho, u, Phi, Lstar, L, E, P, cs, tau, rs]
        write_to_file(logMdot, data)

    if logMdot in (17.25, 17.5, 17.75, 18, 18.25 , 18.5, 18.75, 19):

        c = colors[int(np.floor(i/2)-1)]
        ls = '-' if i%2==0 else '--'

        ax1.semilogx(R, Lstar/LEdd,color=c, lw=0.8,  label=('%.2f' % (log10(Mdot))), linestyle = ls)
        beautify(fig1,ax1)

        def myloglogplot(ax,x,y):
                ax.loglog(x , y, color=c, lw=0.8,  label=('%.2f' % (log10(Mdot))), linestyle = ls)
        def draw_sonicpoint(ax,x,y):
                sonic = list(R).index(rs)
                ax.loglog([x[sonic]] , [y[sonic]], marker='.', color='k')
        for fig,ax,x,y in zip((fig2,fig3,fig4,fig5,fig7,fig8),(ax2,ax3,ax4,ax5,ax7,ax8) , (R,Rho,R,R,Rho,Rho) , (T,T,u,P,kappa(Rho,T),tau)):
                myloglogplot(ax,x,y)
                if i%2==0:draw_sonicpoint(ax,x,y)
                beautify(fig,ax)

        i += 1

for ax in (ax1,ax2,ax3,ax4,ax5,ax7,ax8):
        ax.legend(title=r'log $\dot{M}$ (g/s)', loc='best')

# Additionnal plots

Fbs = array(Lbs)/(4*pi*(RNS*1e5)**2)  # redshifted base flux
fig6, ax6 = plt.subplots(1, 1)
ax6.set_xlabel(r'$F_{b,\infty}$ (10$^{25}$ erg s$^{-1}$ cm$^{-2}$)')
ax6.set_ylabel(r'log $\dot{M}$ (g/s)')
ax6.plot(Fbs/1e25,logMDOTS,'k.-',lw=0.8)
beautify(fig6,ax6)


if save: 
    save_plots([fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8],['Luminosity','Temperature1','Temperature2','Velocity','Pressure','Flux_Mdot','Opacity','Optical_depth'],img)
else:
    plt.show()