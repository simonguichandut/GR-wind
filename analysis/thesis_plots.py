import sys
sys.path.append(".")

from wind_GR import *
import IO

from scipy.optimize import fsolve

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.rcParams.update({

    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    # Non-italic math
    "mathtext.default": "regular",
    # Tick seetings
    "xtick.direction" : "in",
    "ytick.direction" : "in",
    "xtick.top" : True,
    "ytick.right" : True
})


kB = 1.380658e-16
arad = 7.5657e-15
c = 2.99792458e10
mp = 1.67e-24


figsize=(5.95, 3.68) # according to document size


def Make_profiles_plot(lw = 0.8):

    fig,axes = plt.subplots(3,2,figsize=(figsize[0],figsize[1]*3/2), sharex=True)
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0.3)
    (ax1,ax2),(ax3,ax4),(ax5,ax6) = axes

    ax5.set_xlabel(r'r (cm)',labelpad=5)
    ax6.set_xlabel(r'r (cm)',labelpad=5)
    ax5.tick_params(axis='x',pad=4)
    ax6.tick_params(axis='x',pad=4)

    ax1.set_ylabel(r'$T$ (K)')
    ax2.set_ylabel(r'$\rho$ (g cm$^{-3}$)')
    ax3.set_ylabel(r'$u/c$')
    ax4.set_ylabel(r'$\Phi$')
    ax5.set_ylabel(r'$L_\infty/L_{Edd}$')
    ax6.set_ylabel(r'$\tau^*$')

    ax4.axhline(2,color='k',ls=':',lw=0.7)
    ax5.axhline(1,color='k',ls=':',lw=0.7)
    ax6.axhline(3,color='k',ls=':',lw=0.7)

    for ax in (ax1,ax2,ax3,ax4,ax5,ax6):
        ax.grid(alpha=0.5)
    
    colors = ['r','b','g', 'm']

    for i,logMdot in enumerate((17.25, 17.5, 17.75, 18, 18.25 , 18.5, 18.75, 19)):

        w = IO.read_from_file(logMdot)

        ls = '-' if i%2==0 else '--'
        col = colors[int((i-i%2)/2)]

        Linf = w.L*Swz(w.r)

        ax1.loglog(w.r,w.T,lw=lw,ls=ls,color=col,label=str(logMdot))
        ax2.loglog(w.r,w.rho,lw=lw,ls=ls,color=col)
        ax3.loglog(w.r,w.u/c,lw=lw,ls=ls,color=col)
        ax4.loglog(w.r,w.phi,lw=lw,ls=ls,color=col)
        ax5.semilogx(w.r,Linf/LEdd,lw=lw,ls=ls,color=col)
        ax6.loglog(w.r,w.taus,lw=lw,ls=ls,color=col)

        irs = list(w.r).index(w.rs)
        ax1.loglog([w.rs],[w.T[irs]],'.',color=col,ms=4)
        ax2.loglog([w.rs],[w.rho[irs]],'.',color=col,ms=4)
        ax3.loglog([w.rs],[w.u[irs]/c],'.',color=col,ms=4)
        ax4.loglog([w.rs],[w.phi[irs]],'.',color=col,ms=4)
        ax5.semilogx([w.rs],[Linf[irs]/LEdd],'.',color=col,ms=4)
        ax6.loglog([w.rs],[w.taus[irs]],'.',color=col,ms=4)



    # leg = ax1.legend(title=r'log$\dot{M}$ (g s$^{-1}$)',frameon=False, ncol=4,
            # bbox_to_anchor=(1e7,1e11), bbox_transform=ax1.transData)    
    # leg.get_title().set_position((0.1, 1))
    
    # ax1.legend(frameon=False, ncol=4,  bbox_to_anchor=(1e7,1e11), bbox_transform=ax1.transData)
    ax1.legend(frameon=False, ncol=4,  bbox_to_anchor=(0.85,0.97), bbox_transform=fig.transFigure)
    ax1.text(0.18,0.93,(r'log$\dot{M}$ (g s$^{-1}$)'),fontsize=9,transform=fig.transFigure,
            ha='left',va='center')

    # for better clarity
    ax3.set_ylim([1e-12,10**(-1.5)])
    ax5.set_ylim(0.9,3.5)
    ax6.set_ylim(1,10**10.5)

    fig.savefig('analysis/thesis_plots/wind_profiles.pdf',bbox_inches='tight',format='pdf')


def Make_rootsplot():

    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(224,sharex=ax2)
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0.3)

    for ax in (ax1,ax2,ax3):
        ax.grid(alpha=0.5)

    ax1.set_xlabel(r'$(\dot{E}-\dot{M}c^2)/L_{Edd}$')
    ax3.set_xlabel(r'log$\dot{M}$ (g s$^{-1}$)')
    ax1.set_ylabel(r'log$T_c$')
    ax2.set_ylabel(r'$r_c$ (cm)')
    ax3.set_ylabel(r'$L_b^\infty/L_{Edd}$')

    logMdots,roots = IO.load_roots()
    Edotvals = [root[0] for root in roots]
    logTsvals = [root[1] for root in roots]

    ax1.plot(Edotvals,logTsvals,'k-',lw=1)
    ax1.set_xlim([1.015,1.03])
    ax1.set_xticks(np.arange(1.015,1.03,0.003))

    Lbinf,rsonic = [],[]

    for i,logMdot in enumerate(logMdots):
        
        w = IO.read_from_file(logMdot)
        Lbinf.append(w.Lstar[0]) # it's not quite Lstar because of the doppler terms but they are very small near the base so it should be ok
        rsonic.append(w.rs)

        if logMdot in (logMdots[0],logMdots[-1],17.5,18,18.5,18.75):

            ax1.plot([Edotvals[i]],logTsvals[i],'o',mfc='w',mec='k',ms=4)

            if logMdot==18.75:
                s = (r'log$\dot{M}$='+str(logMdot))
            else:
                s = str(logMdot)
            
            ax1.text(Edotvals[i]+0.0005,logTsvals[i],s,fontsize=8,transform=ax1.transData,ha='left',va='center')

    ax2.semilogy(logMdots,rsonic,'k-',lw=1)
    ax3.plot(logMdots,np.array(Lbinf)/LEdd,'k-',lw=1)
    
    fig.savefig('analysis/thesis_plots/wind_roots.pdf',bbox_inches='tight',format='pdf')

    # plt.show()


def Make_density_temperature_plot():

    fig,ax = plt.subplots(1,1,figsize=figsize)
    ax.grid(alpha=0.5)

    ax.set_xlabel(r'$\rho$ (g cm$^{-3}$)')
    ax.set_ylabel(r'$T$ (K)')
    
    ax.set_xlim([1e-8,1e6])
    ax.set_ylim([1e6,5e9])
    
    colors = ['r','b','g', 'm']
    for i,logMdot in enumerate((17.25, 17.5, 17.75, 18, 18.25 , 18.5, 18.75, 19)):

        w = IO.read_from_file(logMdot)
        ls = '-' if i%2==0 else '--'
        col = colors[int((i-i%2)/2)]
        ax.loglog(w.rho,w.T,ls=ls,color=col,lw=0.8)

        irs = list(w.r).index(w.rs)
        ax.loglog([w.rho[irs]],[w.T[irs]],'.',color=col,ms=4)


        # Pressure lines
        Rho = np.logspace(-6,10,100)
        Knr,Kr = 9.91e12/eos.mu_e**(5/3), 1.231e15/eos.mu_e**(4/3)

        # Prad = Pg (non-degen)
        T1 = (3*kB*Rho/(arad*eos.mu*mp))**(1/3)

        # # Prad = Pgas (all contribs)
        # T1b = []
        # for rho in Rho:
        #     def Err(t):
        #         pe,_,_ = eos.electrons(rho,t)
        #         return pe + rho*eos.cs2_I(t) - arad*t**4/3
        #     T1b.append(fsolve(Err,x0=1e12)[0])
            

        # Pednr = Pend (non-degen) : Knr rho**(5/3) = kTrho/mu_e*mp
        T2 = Knr*eos.mu_e*mp/kB * Rho**(2/3)

        # Pedr = Pednr
        rho_rel = (Kr/Knr)**3

        ax.loglog(Rho,T1,'k-',lw=0.3)
        
        ax.loglog(Rho,T2,'k-',lw=0.3)
        ax.axvline(rho_rel,color='k',lw=0.7)

        # ax.loglog(Rho,T1b,'b-',lw=0.5)

        ax.text(Rho[np.argmin(np.abs(T1-2e6))]*2,2e6,(r'$P_r=P_g$'),
            transform=ax.transData,ha='left',va='center')

        ax.text(Rho[np.argmin(np.abs(T2-2e6))]*2,2e6,(r'$P_{end}=P_{ed}$'),
            transform=ax.transData,ha='left',va='center')


#    plt.show()
    fig.savefig('analysis/thesis_plots/wind_rho_T.pdf',bbox_inches='tight',format='pdf')

def Make_density_luminosity_plot():

    fig,ax = plt.subplots(1,1,figsize=figsize)
    ax.grid(alpha=0.5)

    ax.set_xlabel(r'$\rho$ (g cm$^{-3}$)')
    ax.set_ylabel(r'$L/L_{cr}$')

    ax.tick_params(axis='x',pad=4)
    
    colors = ['r','b','g', 'm']
    for i,logMdot in enumerate((17.25, 17.5, 17.75, 18, 18.25 , 18.5, 18.75, 19)):

        w = IO.read_from_file(logMdot)
        ls = '-' if i%2==0 else '--'
        col = colors[int((i-i%2)/2)]
        ax.semilogx(w.rho,w.L/Lcrit(w.r,w.rho,w.T),ls=ls,color=col,lw=0.8)
        irs = list(w.r).index(w.rs)
        ax.semilogx([w.rho[irs]],[w.L[irs]/Lcrit(w.rs,w.rho[irs],w.T[irs])],'.',color=col,ms=4)

    # plt.show()

    fig.savefig('analysis/thesis_plots/wind_L_Lcrit.pdf',bbox_inches='tight',format='pdf')



Make_profiles_plot()
Make_rootsplot()
Make_density_temperature_plot()
Make_density_luminosity_plot()
