import sys
sys.path.append(".")

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from numpy import log10, pi, array, linspace, logspace, where, exp, argwhere

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

# Constants
c = 2.99292458e10
h = 6.62e-27
kappa0 = 0.2
GM = 6.6726e-8*2e33*M
LEdd = 4*pi*c*GM/kappa0
FEdd = LEdd/(4*pi*(RNS*1e5)**2)
ZZ = (1-2*GM/(c**2*RNS*1e5))**(-1/2)  # redshift
g0 = GM/(RNS*1e5)**2 * ZZ
g = 1.61e14 # hard coded
# print(g0,g) 
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
Lb = []  # base luminosity
Tb = []  # base temperature
Lbs = []  # base luminosity at infinity
Tfull, Pfull, Rfull, rhofull = [[] for i in range(4)]  # full temperature, pressure, radius and density arrays
sonicpoints = []

Rphot = []

for logMdot in logMDOTS:
    R, u, cs, rho, T, P, phi, L, Lstar, E, tau, rs = read_from_file(logMdot)
    for a, b in zip((Lb, Lbs, Tb, Tfull, Pfull, Rfull, rhofull, Rphot), (L[0], Lstar[0], T[0], T, P, R, rho, R[-1])):
        a.append(b)
    sonicpoints.append(rs)
    # print('Travel time = ',trapz(u**(-1),R))


# Write some stuff to a text file 
# with open('../GR_results.txt','w') as f:
#     f.write('{:<7s} \t {:<10s} \t {:<10s} \t {:<10s} \t {:<10s}\n'.format('logMdot','Rphot(km)','Tb','Lb','Lbinf'))
#     for i in range(len(logMDOTS)):
#         f.write('{:<7.2f} \t {:<10.5e} \t {:<10.5e} \t {:<10.5e} \t {:<10.5e} \n'.format(
#          logMDOTS[i] , Rphot[i] , Tb[i] , Lb[i] , Lbs[i]) )


# Base Luminosity
Lb, Lbs, Tb = array(Lb), array(Lbs), array(Tb)
Fb = Lb/(4*pi*(RNS*1e5)**2)  # non-redshifted base flux
Fbs = Lbs/(4*pi*(RNS*1e5)**2)  # redshifted base flux
Mdots = 10**array(logMDOTS)

def LerrorPlot(save=1):
    Lerror = (Lbs-LEdd)/(GM*Mdots/RNS/1e5)
    fig,ax = plt.subplots(1,1)
    ax.plot(logMDOTS,Lerror,'k.-')
    if save:
        fig.savefig('misc_plots/Lerror.png')
    else:
        plt.show()
LerrorPlot(save=0)

# For a given base flux, what should be the wind
fluxfunc = interp1d(Fb, logMDOTS, kind='linear')

# Other roots
root1, root2 = [], []
for root in roots:
    root1.append(root[0])
    root2.append(root[1])
rootfunc1 = interp1d(logMDOTS, root1, kind='cubic')
# these two functions will give the root (input for makewind) for a given logMdot
rootfunc2 = interp1d(logMDOTS, root2, kind='cubic')


# Import Light curve data

# Extract luminosity as a function of time
data = np.loadtxt('light_curve/prof', usecols=(0, 1))
t = data[:, 0]
L = data[:, 1]

# Extract base temperature as a function of time, keep y and T arrays
t2, Tb2, Fb2 = [], [], []
lc_y, lc_T, lc_kappa = [], [], []


def append_vars(line, varz, cols):  # take line of file and append its values to variable lists
    l = line.split()
    for var, col in zip(varz, cols):
        var.append(float(l[col]))


with open('light_curve/out', 'r') as out:

    # Number of grid points
    ngrid = int(out.readline())

    # Looping through datafile
    count = 0
    y, T, FF, k = [], [], [], []  #
    for line in out:
        l = line.split()

        if len(l) == 1:
            if count > 0:
                # base temperature & flux corresponds to y=1e8 (first value)
                Tb2.append(T[0])
                Fb2.append(FF[0])
                if count < 2:
                    lc_y = y
                lc_T.append(T)
                lc_kappa.append(k)
                ngrid = len(y)
                y, T, FF, k = [], [], [], []       # reset arrays at every timestep

            count += 1
            t2.append(float(l[0]))

        else:
            append_vars(line, [y, T, FF, k], [0, 1, 2, 9])

            # # opacity check (at top of grid)
            # if len(y)==1:
            #     kappa_burstcool = l[9]
            #     kappa_wind = kappa(T[-1])
            #     print(log10(y[-1]),T[-1]/1e9,kappa_burstcool,kappa_wind)


# Import envelope (with same gravity as wind)

# Extract yT grids for different fluxes
# data_env = np.loadtxt('envelope/grid')
# Fvals_env = np.unique(data_env[:, 2])
# yvals_env = np.unique(data_env[:,0])

def env_F(logF):

    if logF not in Fvals_env:
        sys.exit('Flux value not in envelope grid')
    else:
        ind = (data_env[:, 2] == logF)
        y = data_env[ind, 0]
        T = data_env[ind, 1]
        rho = data_env[ind, 4]
        kappa = data_env[ind,5]
        kappa_es = data_env[ind,6]
        kappa_ff = data_env[ind,7]
        kappa_rad = data_env[ind,8]
        kappa_cond = data_env[ind,9]
        return y, T, rho, kappa, kappa_es, kappa_ff, kappa_rad, kappa_cond

def env_y(logy):

    if logy not in yvals_env:
        sys.exit('y value not in envelope grid')
    else:
        ind = (data_env[:,0] == logy)
        T = data_env[ind,1]
        F = data_env[ind,2]
        return T,F

# Compare envelope with correct gravity to envelope with wrong gravity corrected in burstcool
# fig,ax =plt.subplots(1,1)
# ax.loglog(array(Fb2),array(Tb2),label='burstcool')
# # ax.plot(array(Fb2)/1e25,array(Tb2)/1e9,label='burstcool')

# data_env_y8 = np.loadtxt('envelope/y8.txt')
# logTvals_y8 = array(data_env_y8[:,1])
# logFvals_y8 = array(data_env_y8[:,2])

# ax.loglog(10**logFvals_y8,10**logTvals_y8,label='envelope')
# # ax.plot(10**logFvals_y8/1e25,10**logTvals_y8/1e9,label='envelope')
# # ax.set_xlim([3,9.5])
# # ax.set_ylim([1.2,1.6])
# ax.legend()
# ax.set_xlabel(r'F$_{b}$ (10$^{25}$ erg s$^{-1}$ cm$^{-2}$)', fontsize=14)
# ax.set_ylabel(r'T$_b$ (10$^{9}$ K) ', fontsize=14)

# plt.show()

#######################################################################################################################################
# Typical plots

def lc_plot(setup='12'):

    if setup == '12':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    elif setup == '22':
        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        ax1, ax2 = axes[0]
        ax3, ax4 = axes[1]

    ax1.plot(t, L/1e38, 'b--', label='Calculated light curve')
    ax1.set_xlim([0, 100])
    ax1.set_xlabel('t (s)', fontsize=14)
    #ax1.set_ylabel(r'L$_{b,red}$ (10$^{38}$ erg s$^{-1}$)', fontsize=14)
    ax1.set_ylabel(r'L (10$^{38}$ erg s$^{-1}$)', fontsize=14)

    # observed light curve : doesnt go above Edd
    L2 = []
    for i in range(len(L)):
        if L[i] > LEdd:
            L2.append(LEdd)
        else:
            L2.append(L[i])
    L2 = array(L2)

    ax1.axhline(LEdd/1e38, color='m', linestyle=':')
    ax1.text(180, LEdd/1e38-0.3, r'L$_{Edd}$', fontsize=14)
    ax1.plot(t, L2/1e38, 'b-', label='Observed light curve')
    ax1.legend()

    if setup == '12':
        return fig, ax1, ax2
    elif setup == '22':
        return fig, ax1, ax2, ax3, ax4


def FT_Plot(ax, show_radT=1, xaxis='flux', fulltext_pos=18):

    if xaxis == 'flux':
        x, xlabel = Fb/1e25, r'F$_{b}$ (10$^{25}$ erg s$^{-1}$ cm$^{-2}$)'
    elif xaxis == 'luminosity':
        x, xlabel = Lb/1e38, r'L$_{b}$ (10$^{38}$ erg s$^{-1}$)'
    elif xaxis == 'luminosity_redshifted':
        x, xlabel = Lbs/1e38, r'L$_{b,red}$ (10$^{38}$ erg s$^{-1}$)'

    ax.plot(x, Tb/1e9, 'k', lw=0.5, alpha=0.5)
    ax.set_ylabel(r'T$_b$ (10$^{9}$ K) ', fontsize=14)
    ax.set_xlabel(xlabel, fontsize=14)

    write_number = [17.3, 17.8, 17.9, 18, 18.1, 18.2, 18.3, 18.4, 18.5, 18.6]
    write_number.remove(fulltext_pos)

    for i in range(len(logMDOTS)):

        if logMDOTS[i] == fulltext_pos:
            ax.text(x[i]+0.1, Tb[i]/1e9-0.01,
                    (r'log $\dot{M}$ = '+str(logMDOTS[i])), fontweight='bold', color='r',fontsize=12)
            ax.plot(x[i], Tb[i]/1e9, 'ko', markerfacecolor='w')
        elif logMDOTS[i] in write_number:
            ax.text(x[i]+0.1, Tb[i]/1e9-0.01, str(logMDOTS[i]),
                    fontweight='bold', color='r',fontsize=12)
            ax.plot(x[i], Tb[i]/1e9, 'ko', markerfacecolor='w')

    if show_radT:
        ax.axhline(Trad/1e9, color='k', linestyle='--')
        ax.text(x[0], Trad/1e9-0.015, 'radiation temperature')

    func = interp1d(x, Tb, kind='cubic')
    x2 = linspace(x[0], x[-1], 400)
    y = func(x2)
    ax.plot(x2, y/1e9, 'k-',label='Winds')

    ax.set_xlim(x[0]-0.2, x[-1]+0.1)
    ax.set_ylim(Tb[0]/1e9-0.02, Trad/1e9+0.02)


def column_depth_plot(ax, logMdot, yaxis='temperature',color='k'):

    # Need to run makewind if logMdot not a pre-solved root
    if logMdot in logMDOTS:
        # q = argwhere(array(logMDOTS) == logMdot)[0][0]
        q = logMDOTS.index(logMdot)
        P, T = Pfull[q], Tfull[q]
    else:
        root = [rootfunc1(logMdot), rootfunc2(logMdot)]
        Rad, T, Rho, vel, Phi, Lstar, Lwind, LEdd_loc, E, P, cs, tau, rs, Edot, Ts = MakeWind(
            root, logMdot, mode='wind')

    y = P/g  # column depth
    if yaxis == 'temperature':
        line = ax.loglog(y, T,color=color)
        ax.set_ylabel(r'T (K)', fontsize=14)
    elif yaxis == 'density':
        line = ax.loglog(y,Rho,color=color)
        ax.set_ylabel(r'$\rho$ (g cm$^{-3}$)')
    elif yaxis == 'opacity':
        line = ax.semilogx(y, kappa(T),color=color)
        ax.set_ylabel(r'$\kappa$', fontsize=14)

    ax.set_xlabel(r'y (g cm$^{-2}$)', fontsize=14)

    return line




#######################################################################################################################################
# Envelope vs Wind

# fluxrange = linspace(log10(Fb[0]),log10(Fb[-1]),10)
# def closest(arr,val):  # returns number in array closest to val
    # return arr[np.argmin(np.abs(arr-val))]
# fluxrange2 = np.unique([closest(Fvals_env,f) for f in fluxrange if Fb[0]<10**closest(Fvals_env,f)<Fb[-1]])

def yTrho_plots(save=0):
    fig,(ax,ax2) = plt.subplots(1,2,figsize=(15,7))
    cmap = plt.cm.get_cmap('winter', len(fluxrange2))
    # print('\nF(1e25) logMdot:')
    for i,logF in enumerate(fluxrange2):
        y_env,T_env,rho_env,kappa_env,kappa_env_es,kappa_env_ff,kappa_env_rad,kappa_env_cond = env_F(logF)
        l1=ax.loglog(10**y_env,10**T_env,color=cmap(i),linestyle='--',lw=0.8,alpha=0.9)
        l2=column_depth_plot(ax, fluxfunc(10**logF),color=cmap(i))
        ax.set_yscale('linear')

        l3=ax2.loglog(10**y_env,rho_env,color=cmap(i),linestyle='--',lw=0.8,alpha=0.9)
        l4=column_depth_plot(ax2, fluxfunc(10**logF),yaxis='density',color=cmap(i))

        # print('%.3f\t%.3f'%(10**logF/1e25,fluxfunc(10**logF)))

        if i==0:
            l1[0].set_label('Envelope')
            l2[0].set_label('Wind')
            ax.legend()

            l3[0].set_label('Envelope')
            l4[0].set_label('Wind')
            ax2.legend()

        ax.set_xlim([1e7,1e9])
        ax.set_ylim([9e8,2e9])
        ax2.set_xlim([1e7,1e9])
        # ax2.set_ylim([9e8,2e9])
    if save:
        fig.savefig('misc_plots/env_vs_wind.png')
    else:
        plt.show()

# yTrho_plots()


def FT_Plot_with_envelopes(save=0,plot_error=1):   # ALSO CREATES NEW ENVELOPE FILE

    rc('text', usetex = False)
    mpl.rcParams.update({'font.size': 15})

    #Flux temperature relation
    Ty8,Fy8 = env_y(8)
    Ty8,Fy8 = 10**Ty8,10**Fy8

    if plot_error:
        fig,(ax1,ax2) = plt.subplots(1,2, figsize=(15, 6))
    else:
        fig,ax1 = plt.subplots(1,1,figsize=(8,7))

    FT_Plot(ax1)
    ax1.plot(Fy8/1e25,Ty8/1e9,'b-',lw=1,label='Hydrostatic')
    ax1.legend()
    ax1.grid()

    func_F_wind = interp1d(Fb/1e25,Tb/1e9,kind='linear')
    func_F_env = interp1d(Fy8/1e25,Ty8/1e9,kind='linear')
    x = linspace(Fb[0]/1e25, Fb[-1]/1e25, 400)

    err = (func_F_env(x)-func_F_wind(x))/func_F_wind(x)*100
    if plot_error:
        ax2.plot(x,err)
        ax2.set_ylabel(r'Temperature Error (%)')
        ax2.set_xlabel(r'F$_{b}$ (10$^{25}$ erg s$^{-1}$ cm$^{-2}$)')
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylim([0,max(err)+2])

    # Build new FT file. envelope values until error>0.4%, then wind
    wind_begin = argwhere(Fy8 < Fb[0])[-1][0]
    Fy8_new,Ty8_new = Fy8[:wind_begin],Ty8[:wind_begin]

    use_wind = 0
    for Fval in x:
        Fy8_new = np.append(Fy8_new,Fval*1e25)
        if (func_F_env(Fval)-func_F_wind(Fval))/func_F_wind(Fval)*100 < 0.4 and use_wind==0:
            Ty8_new = np.append(Ty8_new,func_F_env(Fval)*1e9)
        else:
            Ty8_new = np.append(Ty8_new,func_F_wind(Fval)*1e9)
            use_wind = 1

    # ax1.plot(Fy8_new/1e25,Ty8_new/1e9,'k--')
    # import pickle
    # with open('data_to_smooth.pkl','wb') as f:
    #     pickle.dump([Fy8_new,Ty8_new],f)

    
    # Have to smooth out the curve a bit
    x,y = Fy8_new,Ty8_new
    derivs = [(y[i+1]-y[i])/(x[i+1]-x[i]) for i in range(len(x)-1)]
    while True in (np.sign(derivs)!=np.sign(np.average(derivs))):   # the derivatives should stay the same sign
        i = list(np.sign(derivs)!=np.sign(np.average(derivs))).index(True)
        # y[i] = (y[i-1]+y[i+1])/2
        x,y=np.delete(x,i),np.delete(y,i)
        derivs = [(y[i+1]-y[i])/(x[i+1]-x[i]) for i in range(len(x)-1)]

    # ax1.plot(x/1e25,y/1e9,'b.')
    Fy8_new,Ty8_new = x,y

    with open('He_grid_with_winds','w') as f:
        for i in range(len(Fy8_new)):
            f.write('8 %lg %lg\n'%(log10(Ty8_new[i]),log10(Fy8_new[i]))) 


    if save:
        fig.savefig('misc_plots/wind_vs_envelope_FT.png')
    else:
        plt.show()


# FT_Plot_with_envelopes(save=0,plot_error=0)

###################### Opacity block ######################################################################################
# fluxrange = linspace(log10(Fb[0]),log10(Fb[-1]),6)
# fluxrange2 = np.unique([closest(Fvals_env,f) for f in fluxrange if Fb[0]<10**closest(Fvals_env,f)<Fb[-1]])
# fig,ax = plt.subplots(1,1)
# cmap = plt.cm.get_cmap('jet', len(fluxrange2))
# for i,logF in enumerate(fluxrange2):

#     y_env,T_env,rho_env,kappa_env,kappa_env_es,kappa_env_ff,kappa_env_rad,kappa_env_cond  = env_F(logF)

#     l1=ax.semilogx(10**y_env,kappa_env,color=cmap(i),linestyle='-',lw=1.5)
#     l2=ax.semilogx(10**y_env,kappa_env_es,color=cmap(i),linestyle='--',lw=0.8,alpha=0.9)
#     l3=ax.semilogx(10**y_env,kappa_env_ff,color=cmap(i),marker='v',linestyle='None',lw=0.8,alpha=0.9)
#     l4=ax.semilogx(10**y_env,kappa_env_rad,color=cmap(i),marker='o',lw=0.8,alpha=0.9)
#     l5=ax.semilogx(10**y_env,kappa_env_es+kappa_env_ff,color=cmap(i),linestyle='-.',lw=0.8,alpha=0.9)


    
#     def kappa1(T):
#         return kappa0/(1.0+(2.2e-9*T)**0.86)
#     def kappa2(T):
#         return kappa0/(1.0+(T/4.5e8)**0.86)
#     def kappa3(T,rho):
#         return kappa0/( (1.0+(T/4.5e8)**0.86) * (1+2.7e11*rho/T**2) )
#     lx=ax.semilogx(10**y_env,kappa1(10**T_env),color=cmap(i),marker='x',lw='0.8')
#     lx2=ax.semilogx(10**y_env,kappa2(10**T_env),color=cmap(i),marker='+',lw='0.8')
#     lx3=ax.semilogx(10**y_env,kappa3(10**T_env,rho_env),color=cmap(i),marker='*',lw=0.8)

#     if i==0:
#         l1[0].set_label(r'$\kappa_{tot}$')
#         l2[0].set_label(r'$\kappa_{es}$')
#         l3[0].set_label(r'$\kappa_{ff}$')
#         l4[0].set_label(r'$\kappa_{rad}$')
#         l5[0].set_label(r'$\kappa_{es}+\kappa_{ff}$')

#         lx[0].set_label(r'$\kappa(T)_1$')
#         lx2[0].set_label(r'$\kappa(T)_2$')
#         lx3[0].set_label(r'$\kappa(T,\rho)$')
#         ax.legend()

#     ax.set_xlim([1e7,1e9])
#     ax.set_ylim([0.045,0.065])
# plt.show()
############################################################################################################


# Mass above and below photosphere
# Min,Mout = [],[]
# from scipy.integrate import quad
# for i,logMdot in enumerate(logMDOTS):
#     rhofunc = interp1d(Rfull[i]*1e5,rhofull[i],kind='cubic')
#     # x = linspace(Rfull[i][0],Rfull[i][-1],10000)*1e5
#     # plt.loglog(Rfull[i]*1e5,rhofull[i],'k-')
#     # plt.loglog(x,rhofunc(x),'b--')

#     def mass_in_shell(r):
#         return 4*pi*rhofunc(r)*r**2

#     mass_below_sonic,err1 = quad(mass_in_shell , Rfull[i][0]*1e5 ,  sonicpoints[i])
#     mass_above_sonic,err2 = quad(mass_in_shell , sonicpoints[i] , Rfull[i][-1]*1e5)
#     Min.append(mass_below_sonic)
#     Mout.append(mass_above_sonic)

# fig=plt.figure()
# plt.plot(logMDOTS,log10(Min),label='below sonic')
# plt.plot(logMDOTS,log10(Mout),label='above sonic')
# plt.xlabel(r'log $\dot{M}$', fontsize=14)
# plt.ylabel(r'log $\Delta M$', fontsize=14)
# plt.legend()
# plt.show()
# fig.savefig('mass_distribution')









#######################################################################################################################################
# Movies


def movie1(save=0):

    # Follow light curve and see where it lands on the wind base flux/temperature plot

    fig, ax1, ax2 = lc_plot()
    FT_Plot(ax2)

    i, ti = 0, 0
    while ti < 100:

        ti = t2[i]
        time, Lum = t[where(t < ti)], L[where(t < ti)]
        ax1.plot(time, Lum/1e38, 'g-')

        if i > 1 and Lum[-1] > LEdd:
            p = ax2.plot([Fb2[i]/1e25], [Tb2[i]/1e9], 'g.')

        plt.pause(0.01)
        if save:
            fig.savefig(('light_curve/png/%06d.png' % i))

        if i > 1 and Lum[-1] > LEdd:
            # p.pop(0).remove()
            pass

        i += 1

# movie1(save=1)


def moviex(save=0):

    fig, ax1, ax2, ax3, ax4 = lc_plot(setup='22')
    FT_Plot(ax2)

    for m in (17.5, 18, 18.5, 19):
        line = column_depth_plot(ax3, m)
        line[0].set_label(str(m))

    ax3.legend(loc=1, title=r'log $\dot{M}$')
    ax3.set_xlim([1e6, 1e12])
    ax3.set_ylim([1e8, 3e9])
    ax4.set_xlim(ax3.get_xlim())
    ax4.set_ylim(ax3.get_ylim())
    ax4.set_xlabel(ax3.get_xlabel(), fontsize=14)
    ax4.set_ylabel(ax3.get_ylabel(), fontsize=14)
    ax4.text(0.5, 0, '(matching to closest flux)', horizontalalignment='center',
             verticalalignment='bottom', transform=ax4.transAxes)

    if save:
        plt.close()

    for i in range(len(Fb2)):

        ti = t2[i]
        if ti > 100:
            break
        time, Lum = t[where(t < ti)], L[where(t < ti)]
        ax1.plot(time, Lum/1e38, 'g-')

        if Fb2[i] > FEdd:

            po = ax2.plot([Fb2[i]/1e25], [Tb2[i]/1e9], 'g.', alpha=0.8)
            p = ax2.plot([Fb2[i]/1e25], [Tb2[i]/1e9], 'k.')
            l = ax3.loglog(lc_y, lc_T[i], 'k')
            l2 = ax4.loglog(lc_y, lc_T[i], 'k', label='cool')

            # calculate wind
            if Fb2[i] < Fb[0]:
                flux = Fb[0]
            else:
                flux = Fb2[i]

            logmdot = fluxfunc(flux)
            root = [rootfunc1(logmdot), rootfunc2(logmdot)]
            Rad, Temp, Rho, vel, Phi, Lstar, Lwind, LEdd_loc, E, P, cs, tau, rs, Edot, Ts = MakeWind(
                root, logmdot, mode='wind')

            y_wind = P/g
            l3 = ax4.loglog(y_wind, Temp, 'k--', label='wind')
            ax4.legend(loc=1)

        plt.pause(0.01)
        if save:
            fig.savefig(('light_curve/png/%06d.png' % i))

        if Fb2[i] > FEdd:
            for x in (p, l, l2, l3):
                x.pop(0).remove()

        i += 1

# moviex(save=0)


def movie_kappa(save=0):

    fig, ax1, ax2 = lc_plot()
    ax2.set_xlabel(r'y (g cm$^{-2}$)', fontsize=14)
    ax2.set_ylim([0, kappa0])

    for i in range(len(Fb2)):

        ti = t2[i]
        time, Lum = t[where(t < ti)], L[where(t < ti)]
        ax1.plot(time, Lum/1e38, 'g-')

        if Fb2[i] > FEdd:
            l = ax2.semilogx(lc_y, lc_kappa[i], 'b', label='burstcool kappa')
            l2 = ax2.semilogx(lc_y, kappa(
                array(lc_T[i])), 'r', label='wind kappa')
            ax2.legend()

        plt.pause(0.01)
        if save:
            fig.savefig(('light_curve/png/%06d.png' % i))

        if Fb2[i] > FEdd:
            l.pop(0).remove()
            l2.pop(0).remove()

        i += 1

# movie_kappa(save=1)
