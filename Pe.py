kB = 1.38e-16
mp = 1.67e-24
import numpy as np

# def andrewP(rho, T):
#   rY = rho*0.5
#   pednr = 9.91e-2*rY**(5/3)
#   pedr=1.231e1*rY**(4/3)
#   ped=1/sqrt((1/pedr**2)+(1/pednr**2)) 
#   pend=8.254e-7*rY*T                    # *1e8 in Andrew's eos.cc because T8 is T/1e8
#   return 1e14*sqrt(ped**(2) + pend**2)

# def simonP(rho, T):
#   rY = rho*0.5
#   pednr = 9.91e12*rY**(5/3)
#   pedr=1.231e15*rY**(4/3)
#   ped=1/sqrt((1/pedr**2)+(1/pednr**2)) 
#   pend=8.254e-7*rY*T
#   return sqrt(ped**2 + pend**2)

# print(andrewP(1e10,1e8))
# print(simonP(1e10,1e8))
# same


def electrons(rho,T):  # From Paczynski (1983) semi-analytic formula : ApJ 267 315

    Ye = 0.5
    rY = rho*Ye
    pednr = 9.91e12 * (rho*Ye)**(5/3)     
    pedr = 1.231e15 * (rho*Ye)**(4/3)
    ped = 1/np.sqrt((1/pedr**2)+(1/pednr**2))
    pend = kB/mp*rY*T
    pe = np.sqrt(ped**2 + pend**2) # pressure

    f = 5/3*(ped/pednr)**2 + 4/3*(ped/pedr)**2
    Ue = pe/(f-1)
    
    Ue2 = 3/2*pe

    return pe,Ue,pednr,pedr,ped,pend,Ue2


# Test for 1e9 K gas
rho = np.logspace(4,8,300)

pe,Ue,pednr,pedr,ped,pend,Ue2 = electrons(rho,1e9)

import matplotlib.pyplot as plt
fig,[ax1,ax2] = plt.subplots(1,2,figsize=(12,5))
fig.suptitle(r'T = 10$^{9}$ K')

ax1.loglog(rho,pe,'k',label='Total (Paczynski)')
ax1.loglog(rho,pend,'b',lw=0.6,label='non-degenerate')
ax1.loglog(rho,pednr,'r',lw=0.6,label='degenerate non-rel')
ax1.loglog(rho,pedr,'r--',lw=0.6,label='degenerate rel')

ax1.set_xlabel(r'$\rho$ (g cm$^{-3}$)')
ax1.set_ylabel(r'Pressure (dyne cm$^{-2}$)')
ax1.legend()

ax2.loglog(rho,Ue,'k',label='Pe/(f-1)')
ax2.loglog(rho,Ue2,'b',lw=0.6,label='1.5*Pe')

ax2.set_xlabel(r'$\rho$ (g cm$^{-3}$)')
ax2.set_ylabel(r'Ue (erg cm$^{-3}$)')
ax2.legend()

plt.show()