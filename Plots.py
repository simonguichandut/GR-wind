import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import log10, pi, array, linspace, logspace

from wind_GR import MakeWind
from output_data import write_to_file

img_format = '.png'
save=1

c = 2.99792458e10
kappa0 = 0.2
GM = 6.6726e-8*2e33*1.4
LEdd = 4*pi*c*GM/kappa0

# composition,M/Msun,R(km),tau_outer,log10rho_inner
setup = ['He', 1.4, 10, 3, 4]

savedir = '_'.join([str(setup[i]) for i in range(len(setup))])
path = 'results/' + savedir
if not os.path.exists(path):   # Assuming code is being run from main directory
    os.mkdir(path)
    os.mkdir(path+'/data')
    os.mkdir(path+'/plots')

# Import Winds
logMdots = []
roots = []
with open('solutions/sols_He_1.4_10_3.txt', 'r') as f:
    next(f)
    next(f)
    for line in f:
        stuff = line.split()
        logMdots.append(float(stuff[0]))
        roots.append([float(stuff[1]), float(stuff[2])])

# Radius-Luminosity (fig. 2)
fig1, ax1 = plt.subplots(1, 1)
ax1.set_xlabel(r'log $r$ (cm)', fontsize=13)
ax1.set_ylabel(r'log $L^*/L_{Edd}^*$', fontsize=13)

# Radius-Temperature (fig. 4)
fig2, ax2 = plt.subplots(1, 1)
ax2.set_xlabel(r'log $r$ (cm)', fontsize=13)
ax2.set_ylabel(r'log $T$ (K)', fontsize=13)

# Density-Temperature (fig. 5)
fig3, ax3 = plt.subplots(1, 1)
ax3.set_xlabel(r'log $\rho$ (g cm$^{-3}$)', fontsize=13)
ax3.set_ylabel(r'log $T$ (K)', fontsize=13)

# Radius-Velocity (fig. 6)
fig4, ax4 = plt.subplots(1, 1)
ax4.set_xlabel(r'log $r$ (cm)', fontsize=13)
ax4.set_ylabel(r'log $v$ (cm s$^{-1}$)', fontsize=13)

# Radius-Pressure
fig5, ax5 = plt.subplots(1, 1)
ax5.set_xlabel(r'log $r$ (cm)', fontsize=13)
ax5.set_ylabel(r'log $P$ (g cm$^{-1}$ s$^{-2}$)', fontsize=13)

colors = ['r', 'b', 'm', 'g', 'o']

L_error = []
Lstar_error = []
Lstar_error2 = []
Lb = []
Lbs = []

i = 0
for logMdot, root in zip(logMdots, roots):

    print(logMdot)
    global Mdot, verbose
    Mdot, verbose = 10**logMdot, 0

    R, T, Rho, u, Phi, Lstar, L, LEdd_loc, E, P, cs, tau = MakeWind(
        root, logMdot, mode='wind')

    if save:
        data = [R, T, Rho, u, Phi, Lstar, L, E, P, cs, tau]
        write_to_file(path+'/data', logMdot, data)

    if logMdot in (17, 17.5, 18, 18.5, 19):
        ax1.plot(log10(R), log10(Lstar/LEdd),
                 c=colors[i], lw=0.6, label=('%.1f' % (log10(Mdot))))
        ax2.plot(log10(R), log10(T), c=colors[i], lw=0.6, label=(
            '%.1f' % (log10(Mdot))))
        ax3.plot(log10(Rho), log10(T),
                 c=colors[i], lw=0.6, label=('%.1f' % (log10(Mdot))))
        ax4.plot(log10(R), log10(u), c=colors[i], lw=0.6, label=(
            '%.1f' % (log10(Mdot))))
        ax5.plot(log10(R), log10(P), c=colors[i], lw=0.6, label=(
            '%.1f' % (log10(Mdot))))
        i += 1

    Lb.append(L[0])
    Lbs.append(Lstar[0])


ax1.legend(title=r'log $\dot{M}$', loc=1)
ax1.set_xlim([5.8, 9.2])
ax1.set_ylim([-0.1, 0.9])
ax2.legend(title=r'log $\dot{M}$', loc=1)
ax2.set_xlim([5.8, 9.2])
ax2.set_ylim([5.6, 10])
ax3.legend(title=r'log $\dot{M}$', loc=1)
ax3.set_xlim([-9, 8])
ax3.set_ylim([5.6, 10])
ax4.legend(title=r'log $\dot{M}$', loc=1)
ax4.set_xlim([5.8, 9.2])
ax4.set_ylim([5, 9])
ax5.legend(title=r'log $\dot{M}$', loc=1)
ax5.set_xlim([5.8, 9.2])

if save:
    fig1.savefig(path+'/plots/'+'Luminosity'+img_format)
    fig2.savefig(path+'/plots/'+'Temperature1'+img_format)
    fig3.savefig(path+'/plots/'+'Temperature2'+img_format)
    fig4.savefig(path+'/plots/'+'Velocity'+img_format)
    fig5.savefig(path+'/plots/'+'Pressure'+img_format)


# Base Luminosity
Lb, Lbs = array(Lb), array(Lbs)
Mdots = 10**array(logMdots)

fig6, (ax61, ax62) = plt.subplots(1, 2, figsize=(15, 8))
ax61.plot(logMdots, Lb/LEdd, 'ro-', label=r'Local Luminosity')
ax61.plot(logMdots, Lbs/LEdd, 'bo-', label=r'Luminosity at $\infty$')
ax61.plot([logMdots[0], logMdots[-1]], [1, 1], 'k--')
ax61.set_xlabel(r'log $\dot{M}$', fontsize=14)
ax61.set_ylabel(r'$L_b/L_{Edd}$', fontsize=14)
ax61.legend()
ax62.plot(logMdots, Lb/Lbs, 'ko-')
ax62.set_xlabel(r'log $\dot{M}$', fontsize=14)
ax62.set_ylabel(r'$Local/Infinity$', fontsize=14)
ax62.set_ylim([1, 2])


L_error, Lstar_error = (Lb-LEdd)/(GM*Mdots/1e6), (Lbs-LEdd)/(GM*Mdots/1e6)
fig7, ax7 = plt.subplots(1, 1)
ax7.plot(logMdots, L_error, 'ro-', label=r'Comoving Luminosity')
ax7.plot(logMdots, Lstar_error, 'bo-', label=r'Luminosity at $\infty$')
ax7.plot([logMdots[0], logMdots[-1]], [1, 1], 'k--')
ax7.set_xlabel(r'log $\dot{M}$', fontsize=14)
ax7.set_ylabel(r'$(L_b-L_{Edd})/(GM\dot{M}/R)$', fontsize=14)
ax7.legend()

# plt.show()
