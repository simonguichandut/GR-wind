import sys
sys.path.append(".")

import numpy as np 
import matplotlib.pyplot as plt 
from IO import * 

# Constants
kB = 1.380658e-16
arad = 7.5657e-15
c = 2.99792458e10
mp = 1.67e-24
kappa0 = 0.2
sigmarad = 0.25*arad*c

# Parameters
M, RNS, y_inner, tau_out, comp, mode, save, img = load_params()

if comp == 'He':
    # mu = 4.0/3.0  
    mu = 4
GM = 6.6726e-8*2e33*M
LEdd = 4*np.pi*c*GM/kappa0

ZZ = (1-2*GM/(c**2*RNS*1e5))**(-1/2) # redshift
g = GM/(RNS*1e5)**2 * ZZ
P_inner = g*y_inner




R,u,cs,rho,T,P,phi,L,Lstar,E,tau,rs = read_from_file(17.5)
R,u,cs = R*1e5,u*1e5,cs*1e5



# from wind_GR import Tstar

# def cs2p(T):  # ideal gas sound speed  c_s^2  (PROTONS ONLY)
#     return kB*T/(4/3*mp)

# def Ap(T):  # eq 5a
#     return 1 + 1.5*cs2p(T)/c**2

# def Betap(rho, T):  # pressure ratio (ions)
#     Pg = rho*cs2p(T)
#     Pr = arad*T**4/3.0
#     return Pg/(Pg+Pr)

# def Cp(Lstar, T, r, rho, v):  # eq 5c
#     return Tstar(Lstar, T, r, rho, v) * (4-3*Betap(rho, T))/(1-Betap(rho, T)) * arad*T**4/(3*rho)


# A_pac,B_pac,C_pac = Ap(T) , cs2p(T) , Cp(Lstar, T, R, rho, u)

# from wind_GR import *
# # A_pac,B_pac,C_pac = A(rho,T) , B(rho,T) , C(Lstar, T, R, rho, u)


# # Modified from considering degenerate electron gas
# def electrons_full(rho,T):  # From Paczynski (1983) semi-analytic formula : ApJ 267 315

#     if comp == 'He': Ye = 0.5
#     rY = rho*Ye
#     pednr = 9.91e12 * (rho*Ye)**(5/3)     
#     pedr = 1.231e15 * (rho*Ye)**(4/3)
#     ped = 1/sqrt((1/pedr**2)+(1/pednr**2))
#     pend = kB/mp*rY*T
#     pe = sqrt(ped**2 + pend**2) # pressure
    
#     f = 5/3*(ped/pednr)**2 + 4/3*(ped/pedr)**2
#     Ue = pe/(f-1)               # energy density (erg cm-3)

#     alpha1,alpha2 = (pend/pe)**2 , (ped/pe)**2

#     return pe,Ue,f,pednr,pedr,ped,pend,alpha1,alpha2


# pe,Ue,f,pednr,pedr,ped,pend,alpha1,alpha2 = electrons_full(rho,T)

# A_new = 1 + 1.5*cs2(T)/c**2 + pe/(rho*c**2)*(f/(f-1) - alpha1)

# B_new = cs2(T) + pe/rho*(alpha1+alpha2*f)

# C_new = Tstar(Lstar, T, R, rho, u) * ((4-3*Beta(rho, T))/(1-Beta(rho, T)) + 3*pe*alpha1/(arad*T**4)) * arad*T**4/(3*rho)
# # C_new2 = Lstar/LEdd * kappa(T)/kappa0 * GM/(4*R) * ((4-3*Beta(rho, T))/(1-Beta(rho, T)) + 3*pe*alpha1/(arad*T**4)) * (1+(u/c)**2)**(-1) * Y(R, u)**(-3)
# # C_new2 = Tstar(Lstar, T,R,rho,u) * (cs2(T)+4/3*arad*T**4/rho)


from wind_GR import *
A_pac,B_pac,C_pac = A(T),B(T),C(Lstar,T,R,rho,u)
A_new,B_new,C_new = A_e(rho,T),B_e(rho,T),C_e(Lstar,T,R,rho,u)


fig,ax=plt.subplots(1,1)
ax.semilogx(rho,A_pac,label='Paczynski')
ax.semilogx(rho,A_new,label='Electrons')
ax.set_xlabel(r'log $\rho$',fontsize=14)
ax.legend()
ax.set_title('A')
plt.tight_layout()
# fig.savefig('misc_plots/A.png')

fig,ax=plt.subplots(1,1)
ax.loglog(rho,B_pac,label='Paczynski')
ax.loglog(rho,B_new,label='Electrons')
ax.set_xlabel(r'log $\rho$',fontsize=14)
ax.legend()
ax.set_title('B')
plt.tight_layout()
# fig.savefig('misc_plots/B.png')

fig,ax=plt.subplots(1,1)
ax.loglog(rho,C_pac,label='Paczynski')
ax.loglog(rho,C_new,label='Electrons')
# ax.loglog(rho,C_new2,'k--',label='Electrons')
ax.set_xlabel(r'log $\rho$',fontsize=14)
ax.legend()
ax.set_title('C')
plt.tight_layout()
# fig.savefig('misc_plots/C.png')

fig,ax=plt.subplots(1,1)
ax.semilogx(rho,(A_pac-A_new)/A_new,label='A')
ax.semilogx(rho,(B_pac-B_new)/B_new,label='B')
ax.semilogx(rho,(C_pac-C_new)/C_new,label='C')
ax.set_xlabel(r'log $\rho$',fontsize=14)
ax.set_ylabel('Relative error',fontsize=14)
ax.legend()
plt.tight_layout()

plt.show()