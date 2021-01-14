import sys
sys.path.append('.')  

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl 

mpl.rcParams.update({

    # Use LaTeX to write all text
    # "text.usetex": True,
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
    # Tick settings
    "xtick.direction" : "in",
    "ytick.direction" : "in",
    "xtick.top" : True,
    "ytick.right" : True,
    # Short dash sign
    "axes.unicode_minus" : True
})


from IO import read_from_file, load_params, load_roots
assert load_params()['FLD'] == True
from photosphere import Rphot_Teff



model_name = 'He_IG_M1.4_R12_y8_FLD'
LEdd = 4*np.pi*3e10*6.6726e-8*2e33*1.4/0.2


def rho_v_plot():

    fig,(ax1,ax2) = plt.subplots(1,2, figsize=(6.97, 4.31))
    for ax in (ax1,ax2):
        ax.set_xlabel('r (cm)')
        ax.set_xlim([3e6,1e9])
    # ax1.set_ylabel('T (K)')
    ax1.set_ylabel(r'$\rho$ (g cm$^{-3}$)')
    ax2.set_ylabel(r'v (cm s$^{-1}$)')
    ax1.set_ylim([1e-9,1e-1])
    ax2.set_ylim([1e7,3e8])


    for logMdot in (17.5,18,18.5):

        w_simple = read_from_file(logMdot, specific_file=('results/'+model_name+'/data/%.2f.txt'%logMdot))
        
        rph1  = Rphot_Teff(logMdot, wind=w_simple)
        iphot1 = list(w_simple.r).index(rph1)
        rs1 = w_simple.rs
        isonic1 = list(w_simple.r).index(rs1)

        x = w_simple.r
        # for y,ax in zip((w_simple.T, w_simple.rho, w_simple.u),(ax1,ax2,ax3)):
        for y,ax in zip((w_simple.rho, w_simple.u),(ax1,ax2)):
            label = (r'$P_R = aT^4/3$') if logMdot==17.5 else None
            ax.loglog(x, y, 'k-', lw=0.7, label=label)
            ax.loglog(x[isonic1], y[isonic1], 'kx', ms=2)
            ax.loglog(x[iphot1], y[iphot1], 'k.', ms=3)


        w_exact = read_from_file(logMdot, specific_file=('results/'+model_name+'_exact/data/%.2f.txt'%logMdot))

        rph2  = Rphot_Teff(logMdot, wind=w_exact)
        iphot2 = list(w_exact.r).index(rph2)
        rs2 = w_exact.rs
        isonic2 = list(w_exact.r).index(rs2)

        x = w_exact.r
        # for y,ax in zip((w_exact.T, w_exact.rho, w_exact.u),(ax1,ax2,ax3)):
        for y,ax in zip((w_exact.rho, w_exact.u),(ax1,ax2)):
            label = (r'$P_R = (\lambda+\lambda^2R^2)aT^4$') if logMdot==17.5 else None
            ax.loglog(x, y, 'b-', lw=0.7, label=label)
            ax.loglog(x[isonic2], y[isonic2], 'bx', ms=2) 
            ax.loglog(x[iphot2], y[iphot2], 'b.', ms=3)


        print('At logMdot=%.2f, the sonic point radii are off by %.2f %% , the photospheres are off by %.2f %%'%
                                (logMdot,abs(rs1-rs2)/rs1*100,abs(rph1-rph2)/rph1*100))

    ax1.legend(frameon=False)
    fig.savefig('analysis/Prad_exact_or_simple_profiles.png', bbox_inches='tight', dpi=300)
    # fig.savefig('analysis/Prad_exact_or_simple.pdf', bbox_inches='tight')
    # plt.show()

# rho_v_plot()



def Lb_rph_plot():

    fig,ax = plt.subplots(1,1,figsize=(6,4))
    ax.set_xlabel(r'$L_b^\infty/L_{E}$')
    ax.set_ylabel(r'r (km)')

    for Prad,col in zip(('simple','exact'),('k','b')):

        name = model_name
        if Prad == 'exact': 
            name+='_exact'
            label = (r'$P_R = (\lambda+\lambda^2R^2)aT^4$') 
        else:
            label = (r'$P_R = aT^4/3$') 

        logMdots,_ = load_roots(specific_file='roots/roots_'+name+'.txt')

        Lbs,rs,rph = [],[],[]

        for logMdot in logMdots:

            try:
                w = read_from_file(logMdot,specific_file='results/'+name+'/data/%.2f.txt'%logMdot)

                Lbs.append(w.Lstar[0])
                rs.append(w.rs)
                rph.append(Rphot_Teff(logMdot, wind=w))
            except:
                pass

        ax.semilogy(np.array(Lbs)/LEdd,np.array(rs)/1e5,color=col,ls='--',lw=0.8)
        ax.semilogy(np.array(Lbs)/LEdd,np.array(rph)/1e5,color=col,ls='-',lw=0.8,label=label)

    ax.text(2,370,r'$r_{ph}$',ha='left',va='center')
    ax.text(2,105,r'$r_{s}$',ha='left',va='center')
    ax.legend(frameon=False)
    fig.savefig('analysis/Prad_exact_or_simple_Lbs.png',bbox_inches='tight', dpi=300)



Lb_rph_plot()