# custom script for adding the true optical depth tau to the save files without rerunning the winds

import IO
from wind_GR_FLD import *


def update_savefile(logMdot):

    x,z=IO.load_roots()
    Edot,Ts = z[list(x).index(logMdot)]

    R,u,cs,rho,T,P,phi,L,Lstar,E,taus,lam,rs = IO.read_from_file(logMdot)

    # calculate tau
    rend = R[-1]
    fT,frho = IUS(R,T),IUS(R,rho)
    def integrand(d):
        r = rend-d
        return eos.kappa(frho(r),fT(r))*frho(r)
    
    tau = []
    for r in R:
        tau.append(quad(integrand,0,rend-r,limit=100,epsabs=1-3,epstol=1e-3)[0])

    w = Wind(R, T, rho, u, phi, Lstar, L, E, P, cs, tau, taus, lam, rs, Edot, Ts)


update_savefile(17.75)