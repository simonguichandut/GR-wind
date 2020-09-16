import numpy as np
import matplotlib.pyplot as plt 
import IO
import physics
from scipy.integrate import trapz

assert IO.load_params()['FLD'] == True


arad = 7.5657e-15
c = 2.99792458e10
GM = 6.6726e-8*2e33*IO.load_params()['M']

eos = physics.EOS(IO.load_params()['comp'])


def Swz(r):  # Schwartzchild metric term
    return (1-2*GM/c**2/r)    

def Rphot_Teff(logMdot):
    # defines the photosphere as the location where T=Teff, i.e. L=4pi Rph^2 sigmaT^4

    w = IO.read_from_file(logMdot)
    F = w.L/(4*np.pi*w.r**2)
    alpha = F/(arad*c*w.T**4)

    return w.r[np.argmin(abs(alpha - 0.25))]



def Rphot_trapz(logMdot):
    # integrates in to find the photosphere with trapz

    w = IO.read_from_file(logMdot)
    kapparho = eos.kappa(w.rho,w.T) * w.rho / Swz(w.r)**(0.5)
    i = 2
    while True:
        tau = trapz(kapparho[-i:],x=w.r[-i:])

        # will keep updating until given tau is reached. break at maximum tau

        if tau<2/3:
            Rph23 = w.r[-i]  

        if tau<1:
            Rph1 = w.r[-i]

        # if tau<2:
        #     Rph2 = w.r[-i]

        # if tau<3:
        #     Rph3 = w.r[-i]
    
        else:
            break

        i+=1
        

    # Rph2 = w.r[-i]

    # which?
    # Rph = Rph1

    return Rph23,Rph1#,Rph2,Rph3

def Rphot_tau_twothirds(logMdot):
    return Rphot_trapz(logMdot)[0]


def Rphot_interpolate(logMdot):
    # extrapolating rho as a power law from the maximum computed 
    # radius to infinity

    w = IO.read_from_file(logMdot)

    # return Rph


def Rphot_pac(logMdot):
    # taustar = 3
    w = IO.read_from_file(logMdot)
    return w.r[np.argmin(abs(w.taus-3))]


def compare_definitions():
    # makes a plot to compare: r(taustar=3),r(tau=1),r(tau=2/3),rs for all Mdots

    import matplotlib.pyplot as plt 

    logMdots,_ = IO.load_roots()

    rtaus3, rtau23, rtau1, rsonic = [[] for i in range(4)]

    for logMdot in logMdots:
        Rph23,Rph1 = Rphot_trapz(logMdot)
        rtau23.append(Rph23)
        rtau1.append(Rph1)

        rtaus3.append(Rphot_pac(logMdot))

        rsonic.append(IO.read_from_file(logMdot).rs)

    fig,ax=plt.subplots(1,1)
    ax.set_xlabel(r'$\dot{M}$ (g/s)',fontsize=15)
    ax.set_ylabel(r'r (cm)',fontsize=15)

    Mdots = 10**np.array(logMdots)

    ax.loglog(Mdots,rsonic,'ko-',mfc='w',label=r'$r_s$')
    ax.loglog(Mdots,rtaus3,'ro-',mfc='w',label=r'$r(\tau^*=3)$')
    ax.loglog(Mdots,rtau1,'bo-',mfc='w',label=r'$r(\tau=1)$')
    ax.loglog(Mdots,rtau23,'bo--',mfc='w',label=r'$r(\tau=2/3)$')
    ax.legend(frameon=False)

    return fig,ax

# fig,ax = compare_definitions()
# plt.tight_layout()
# plt.show()
