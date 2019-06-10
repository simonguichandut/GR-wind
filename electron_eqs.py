import numpy as np 
import matplotlib.pyplot as plt 
from IO import * 

c = 2.99792458e10

R,u,cs,rho,T,P,phi,L,Lstar,E,tau,rs = read_from_file(17.5)
R,u,cs = R*1e5,u*1e5,cs*1e5

from wind_GR import *

A_pac,B_pac,C_pac = A(T) , cs2(T) , C(Lstar, T, R*1e5, rho, u)


# Modified from considering degenerate electron gas
def electrons(rho,T):  # From Paczynski (1983) semi-analytic formula : ApJ 267 315

    if comp == 'He': Ye = 0.5
    rY = rho*Ye
    pednr = 9.91e12 * (rho*Ye)**(5/3)     
    pedr = 1.231e15 * (rho*Ye)**(4/3)
    ped = 1/sqrt((1/pedr**2)+(1/pednr**2))
    pend = kB/mp*rY*T
    pe = sqrt(ped**2 + pend**2) # pressure
    
    f = 5/3*(ped/pednr)**2 + 4/3*(ped/pedr)**2
    Ue = pe/(f-1)               # energy density (erg cm-3)

    alpha1,alpha2 = (pend/pe)**2 , (ped/pe)**2

    return pe,Ue,f,pednr,pedr,ped,pend,alpha1,alpha2


pe,Ue,f,pednr,pedr,ped,pend,alpha1,alpha2 = electrons(rho,T)

A_new = 1 + 1.5*cs2(T)/c**2 + pe/(rho*c**2)*(f/(f-1) - alpha1)

B_new = cs2(T) + pe/rho*(alpha1+alpha2*f)

C_new = Tstar(Lstar, T, R, rho, u) * ((4-3*Beta(rho, T))/(1-Beta(rho, T)) + pe*alpha1/rho) * arad*T**4/(3*rho)



fig,ax=plt.subplots(1,1)
ax.semilogx(rho,A_pac)
ax.semilogx(rho,A_new)

fig,ax=plt.subplots(1,1)
ax.semilogx(rho,B_pac)
ax.semilogx(rho,B_new)

fig,ax=plt.subplots(1,1)
ax.semilogx(rho,C_pac)
ax.semilogx(rho,C_new)



plt.show()