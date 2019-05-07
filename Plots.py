''' Plotting various quantities from wind solutions ''' 

import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import log10, pi, array, linspace, logspace

from wind_GR import MakeWind
from IO import *

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

# Parameters
M, RNS, y_inner, tau_out, comp, mode, save, img = load_params()

# Constants
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


########## PLOTS ###########

# Radius-Luminosity (fig. 2)
fig1, ax1 = plt.subplots(1, 1)
# ax1.set_xlabel(r'log $r$ (cm)', fontsize=14)
# ax1.set_ylabel(r'log $L^*/L_{Edd}$', fontsize=14)
ax1.set_xlabel(r'$r$ (cm)', fontsize=14)
# ax1.set_ylabel(r'$L^*/L_{Edd}$', fontsize=14)
ax1.set_ylabel(r'$L/L_{Edd}$', fontsize=14)     # Not writing L* for clarity (but keep in mind this is still lum at inf)

# Radius-Temperature (fig. 4)
fig2, ax2 = plt.subplots(1, 1)
# ax2.set_xlabel(r'log $r$ (cm)', fontsize=14)
# ax2.set_ylabel(r'log $T$ (K)', fontsize=14)
ax2.set_xlabel(r'$r$ (cm)', fontsize=14)
ax2.set_ylabel(r'$T$ (K)', fontsize=14)

# Density-Temperature (fig. 5)
fig3, ax3 = plt.subplots(1, 1)
# ax3.set_xlabel(r'log $\rho$ (g cm$^{-3}$)', fontsize=14)
# ax3.set_ylabel(r'log $T$ (K)', fontsize=14)
ax3.set_xlabel(r'$\rho$ (g cm$^{-3}$)', fontsize=14)
ax3.set_ylabel(r'$T$ (K)', fontsize=14)

# Radius-Velocity (fig. 6)
fig4, ax4 = plt.subplots(1, 1)
# ax4.set_xlabel(r'log $r$ (cm)', fontsize=14)
# ax4.set_ylabel(r'log $v$ (cm s$^{-1}$)', fontsize=14)
ax4.set_xlabel(r'$r$ (cm)', fontsize=14)
ax4.set_ylabel(r'$v$ (cm s$^{-1}$)', fontsize=14)

# Radius-Pressure
fig5, ax5 = plt.subplots(1, 1)
# ax5.set_xlabel(r'log $r$ (cm)', fontsize=14)
# ax5.set_ylabel(r'log $P$ (g cm$^{-1}$ s$^{-2}$)', fontsize=14)
ax5.set_xlabel(r'$r$ (cm)', fontsize=14)
ax5.set_ylabel(r'$P$ (g cm$^{-1}$ s$^{-2}$)', fontsize=14)

colors = ['r', 'b', 'g', 'k', 'm']

i = 0
for logMdot, root in zip(logMDOTS, roots):

    print(logMdot)
    global Mdot, verbose
    Mdot, verbose = 10**logMdot, 0

    R, T, Rho, u, Phi, Lstar, L, LEdd_loc, E, P, cs, tau, rs, Edot, Ts = MakeWind(
        root, logMdot, mode='wind')

    if save:
        data = [R, T, Rho, u, Phi, Lstar, L, E, P, cs, tau]
        write_to_file(logMdot, data)

    if logMdot in (17, 17.5, 18, 18.5,18.25,18.75):
        # ax1.plot(log10(R), log10(Lstar/LEdd),
        #          c=colors[i], lw=0.8, label=('%.2f' % (log10(Mdot))))
        # ax2.plot(log10(R), log10(T), c=colors[i], lw=0.8, label=(
        #     '%.2f' % (log10(Mdot))))
        # ax3.plot(log10(Rho), log10(T),
        #          c=colors[i], lw=0.8, label=('%.2f' % (log10(Mdot))))
        # ax4.plot(log10(R), log10(u), c=colors[i], lw=0.8, label=(
        #     '%.2f' % (log10(Mdot))))
        # ax5.plot(log10(R), log10(P), c=colors[i], lw=0.8, label=(
        #     '%.2f' % (log10(Mdot))))
        ax1.semilogx(R, Lstar/LEdd,
                 c=colors[i], lw=0.8, label=('%.2f' % (log10(Mdot))))
        ax2.loglog(R, T, c=colors[i], lw=0.8, label=(
            '%.2f' % (log10(Mdot))))
        ax3.loglog(Rho, T,
                 c=colors[i], lw=0.8, label=('%.2f' % (log10(Mdot))))
        ax4.loglog(R, u, c=colors[i], lw=0.8, label=(
            '%.2f' % (log10(Mdot))))
        ax5.loglog(R, P, c=colors[i], lw=0.8, label=(
            '%.2f' % (log10(Mdot))))

        i += 1


ax1.legend(title=r'log $\dot{M}$ (g/s)', loc=1)
# ax1.set_xlim([5.8, 9.2])
# ax1.set_ylim([-0.1, 0.9])
ax2.legend(title=r'log $\dot{M}$ (g/s)', loc=1)
# ax2.set_xlim([5.8, 9.2])
# ax2.set_ylim([5.6, 10])
ax3.legend(title=r'log $\dot{M}$ (g/s)', loc=4)
# ax3.set_xlim([-9, 8])
# ax3.set_ylim([5.6, 10])
ax4.legend(title=r'log $\dot{M}$ (g/s)', loc=4)
# ax4.set_xlim([5.8, 9.2])
# ax4.set_ylim([5, 9])
ax5.legend(title=r'log $\dot{M}$ (g/s)', loc=1)
# ax5.set_xlim([5.8, 9.2])

if save: 
    save_plots([fig1,fig2,fig3,fig4,fig5],['Luminosity','Temperature1','Temperature2','Velocity','Pressure'],img)
else:
    plt.show()