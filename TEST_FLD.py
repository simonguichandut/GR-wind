from wind_GR import *
import matplotlib.pyplot as plt
import numpy as np

global logMdot,Mdot,Edot
logMdot = 18.5
Mdot = 10**18.5
Edot = 1.025426*LEdd # from solutions

from IO import *
R0,u0,cs0,rho0,T0,P0,phi0,L0,Lstar0,E0,tau0,rs0=read_from_file(logMdot)
r0,u0,cs0=R0*1e5,u0*1e5,cs0*1e5



# Constants
kB = 1.380658e-16
arad = 7.5657e-15
c = 2.99792458e10
mp = 1.67e-24
kappa0 = 0.2


## FLD
def Rfunc(E,dE,rho,T):
    return abs(dE)/(E*kappa(rho,T)*rho)

def lamfunc(E,dE,rho,T):
    rr = Rfunc(E,dE,rho,T)
    return (2+rr)/(6+3*rr+rr**2)


# first pass through the data
RR0,lam0 = [],[]
for i,ri in enumerate(r0[1:]):
    E=arad*T0[i]**4
    Eprev=arad*T0[i-1]**4
    dE=(E-Eprev)/(r0[i]-r0[i-1])
    RR0.append(Rfunc(E,dE,rho0[i],T0[i]))
    lam0.append(lamfunc(E,dE,rho0[i],T0[i]))

# fig,ax=plt.subplots(1,1)
# ax.semilogx(r0[1:]/1e5,RR0,'k-')
# ax.set_xlabel('r (km)')
# ax.set_ylabel(r'$R=|\nabla E_R|/(\rho\kappa E_R)$',fontsize=14)
# ax2=ax.twinx()
# ax2.semilogx(r0[1:]/1e5,lam0,'b-')
# ax2.set_ylabel(r'$\lambda=(2+R)/(6+3R+R^2)$',color='b',fontsize=14)
# plt.tight_layout()
# plt.show()


# """
def calcvars2(r,T,u):
    rho = Mdot/(4*pi*r**2*u*Y(r, u))
    Lstar = Edot-Mdot*H(rho, T)*Y(r, u) + Mdot*c**2 
    L = Lstar/(1+u**2/c**2)/Y(r, u)**2
    return rho,Lstar,L

def drFLD(r, T, u, Tprev, rprev):

    ''' applying FLD and assuming relativstic terms are small '''

    rho,Lstar,L = calcvars2(r,T,u)

    E,Eprev = arad*T**4, arad*Tprev**4
    dE = (E-Eprev)/(r-rprev)
    Lam = lamfunc(E,dE,rho,T)
    # print(Lam)

    dlnT_dlnr = -kappa(rho,T)*rho*L/(16*pi*r*arad*c*T**4*Lam)
    # dlnu_dlnr = ( GM/r + (B(T)+4*arad*T**4/3/rho)*dlnT_dlnr - 2*B(T) ) / (B(T) - u**2*(A(T)+B(T)/c**2) )+ gamma(u)**2*u**2/r/c**2 * 4*arad*T**4/3/rho ) 
    dlnu_dlnr = ( GM/r/Swz(r) + (B(T)+4*arad*T**4/3/rho)*dlnT_dlnr - 2*B(T) ) / (B(T) - u**2 + gamma(u)**2*u**2/r/c**2 * 4*arad*T**4/3/rho ) 

    dlnu_dlnr_pac = gamma(u)**(-2) * (GM/r/Swz(r) * (A(T)-B(T)/c**2) - C(Lstar, T, r, rho, u) - 2*B(T)) / (B(T)-u**2*A(T))

    # keeping all terms
    num = (c**2+2.5*B(T)+4*arad*T**4/3/rho)*GM/r/c**2/Swz(r) + (B(T)+4*arad*T**4/3/rho)*dlnT_dlnr - B(T)*(2+GM/r/c**2/Swz(r))
    denom = (B(T) - u**2*(A(T)+B(T)/c**2) + gamma(u)**2*u**2/r/c**2 * 4*arad*T**4/3/rho )
    dlnu_dlnr2 = num/denom

    print('\n %.3f \t %.3f \t %.3f'%(dlnu_dlnr,dlnu_dlnr_pac,dlnu_dlnr2))
    print(Swz(r))

    # dlnT_dlnr_other = -Tstar(Lstar, T, r, rho, u) - 1/Swz(r) * GM/c**2/r
    # dlnT_dlnr_other = -3*kappa(rho,T)*rho*L/(16*pi*r*arad*c*T**4)
    # print(dlnT_dlnr,dlnT_dlnr_other)

    dT_dr = T/r * dlnT_dlnr
    du_dr = u/r * dlnu_dlnr

    return dT_dr,du_dr,Lam



fig,[[ax1,ax2,ax5],[ax3,ax4,ax6]] = plt.subplots(2,3,figsize=(16,10))
ax1.loglog(r0,T0,'k-',linewidth=0.8)
ax2.loglog(r0,rho0,'k-',linewidth=0.8)
ax5.loglog(r0,u0,'k-',linewidth=0.8)
ax3.semilogx(r0,L0/LEdd,'k-',linewidth=0.8)
ax4.loglog(r0,tau0,'k-',linewidth=0.8)
ax6.semilogx(r0[1:],lam0,'k-',linewidth=0.8)
ax3.set_xlabel(r'$r$ (cm)',fontsize=14)
ax4.set_xlabel(r'$r$ (cm)',fontsize=14)
ax6.set_xlabel(r'$r$ (cm)',fontsize=14)
ax1.set_ylabel(r'$T$ (K)',fontsize=14)
ax2.set_ylabel(r'$\rho$ (g cm$^{-3}$)',fontsize=14)
ax3.set_ylabel(r'$L/L_{E}$',fontsize=14)
ax4.set_ylabel(r'$\tau^*=\kappa\rho r$',fontsize=14)
ax5.set_ylabel(r'$u$ (cm/s)',fontsize=14)
ax6.set_ylabel(r'$\lambda=(2+R)/(6+3R+R^2)$',fontsize=14)
for ax in (ax1,ax2,ax3,ax4,ax5,ax6):
    ax.axvline(rs0,alpha=0.5)

# i0 = np.argwhere(r0==rs0)[0][0]  # first try from sonic point
# i0 = np.argmin(np.abs(r0-(rs0+10e5)))  # second try 10 km above sonic point
i0 = np.argmin(np.abs(r0-(rs0+100e5)))  # third try 100 km above sonic point



r,T,u,rho,L,Lam,tau= [r0[i0-1],r0[i0]], [T0[i0-1],T0[i0]] , [u0[i0-1],u0[i0]] , [rho0[i0-1],rho0[i0]] , [L0[i0-1],L0[i0]], [lam0[i0-1],lam0[i0]], [tau0[i0-1],tau0[i0]]

dr = 1e3     # 10m stepsize
while r[-1]<1000e5: # go to 3000km

    dT_dr,du_dr,LLam = drFLD(r[-1],T[-1],u[-1],T[-2],r[-2])
    r.append(r[-1]+dr)
    T.append(T[-1]+dT_dr*dr)
    u.append(u[-1]+du_dr*dr)
    Lam.append(LLam)

    rhoi,Lstari,Li = calcvars2(r[-1],T[-1],u[-1])
    rho.append(rhoi)
    L.append(Li)
    tau.append(rhoi*kappa(rhoi,T[-1])*r[-1])

    if T[-1]<0:
        break

    # ax1.loglog(r[-1],T[-1],'b.')
    # ax2.loglog(r[-1],rho[-1],'b.')
    # ax3.semilogx(r[-1],L[-1]/LEdd,'b.')
    # ax4.loglog(r[-1],tau[-1],'b.')
    # plt.pause(0.01)

    print('r = %.2f km  -  T = %.2e  - rho = %.2e '%(r[-1]/1e5,T[-1],rho[-1]))


ax1.loglog(r,T,'b-')
ax2.loglog(r,rho,'b-')
ax3.semilogx(r,np.array(L)/LEdd,'b-')
ax4.loglog(r,tau,'b-')
ax5.loglog(r,u,'b.-')
ax6.semilogx(r,Lam,'b-')

plt.tight_layout()
plt.show()

# """