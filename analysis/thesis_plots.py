import sys
sys.path.append(".")

from wind_GR import *
import IO

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


figsize=(5.95, 3.68) # according to document size


def Make_profiles_plot(lw = 0.8):

    fig,axes = plt.subplots(3,2,figsize=(figsize[0],figsize[1]*3), sharex=True)
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0.3)
    (ax1,ax2),(ax3,ax4),(ax5,ax6) = axes

    ax5.set_xlabel(r'r (cm)',labelpad=5)
    ax6.set_xlabel(r'r (cm)',labelpad=5)
    ax5.tick_params(axis='x',pad=4)
    ax6.tick_params(axis='x',pad=5)

    ax1.set_ylabel(r'$T$ (K)')
    ax2.set_ylabel(r'$\rho$ (g cm$^{-3}$)')
    ax3.set_ylabel(r'$u/c$')
    ax4.set_ylabel(r'$\Phi$')
    ax5.set_ylabel(r'$L/L_{Edd}$')
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

        ax1.loglog(w.r,w.T,lw=lw,ls=ls,color=col,label=str(logMdot))
        ax2.loglog(w.r,w.rho,lw=lw,ls=ls,color=col)
        ax3.loglog(w.r,w.u/c,lw=lw,ls=ls,color=col)
        ax4.loglog(w.r,w.phi,lw=lw,ls=ls,color=col)
        ax5.semilogx(w.r,w.L/LEdd,lw=lw,ls=ls,color=col)
        ax6.loglog(w.r,w.taus,lw=lw,ls=ls,color=col)

        irs = list(w.r).index(w.rs)
        ax1.loglog([w.rs],[w.T[irs]],'.',color=col,ms=4)
        ax2.loglog([w.rs],[w.rho[irs]],'.',color=col,ms=4)
        ax3.loglog([w.rs],[w.u[irs]/c],'.',color=col,ms=4)
        ax4.loglog([w.rs],[w.phi[irs]],'.',color=col,ms=4)
        ax5.semilogx([w.rs],[w.L[irs]/LEdd],'.',color=col,ms=4)
        ax6.loglog([w.rs],[w.taus[irs]],'.',color=col,ms=4)



    # leg = ax1.legend(title=r'log$\dot{M}$ (g s$^{-1}$)',frameon=False, ncol=4,
            # bbox_to_anchor=(1e7,1e11), bbox_transform=ax1.transData)    
    # leg.get_title().set_position((0.1, 1))
    
    # ax1.legend(frameon=False, ncol=4,  bbox_to_anchor=(1e7,1e11), bbox_transform=ax1.transData)
    ax1.legend(frameon=False, ncol=4,  bbox_to_anchor=(0.85,0.94), bbox_transform=fig.transFigure)
    ax1.text(0.17,0.91,(r'log$\dot{M}$ (g s$^{-1}$)'),fontsize=9,transform=fig.transFigure,
            ha='left',va='center')

    # for better clarity
    ax3.set_ylim([1e-12,10**(-1.5)])
    ax5.set_ylim(0.9,5.2)
    ax6.set_ylim(1,10**10.5)

    fig.savefig('analysis/thesis_plots/wind_profiles.pdf',bbox_inches='tight',format='pdf')

Make_profiles_plot()