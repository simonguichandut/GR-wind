import sys
sys.path.append(".")

import numpy as np 
import matplotlib.pyplot as plt 
from IO import load_params,read_from_file


# # Parameters
# M, RNS, y_inner, tau_out, comp, mode, save, img = load_params()

logMdot=17.3

R,u,cs,rho,T,P,phi,L,Lstar,E,tau,rs = read_from_file(logMdot)


from wind_GR import A,B,C,A_e,B_e,C_e
A_pac,B_pac,C_pac = A(T),B(T),C(Lstar,T,R,rho,u)
A_new,B_new,C_new = A_e(rho,T),B_e(rho,T),C_e(Lstar,T,R,rho,u)


fig,ax=plt.subplots(1,1)
ax.semilogx(rho,A_pac,label='Paczynski')
ax.semilogx(rho,A_new,label='Electrons')
ax.set_xlabel(r'log $\rho$',fontsize=14)
ax.legend()
ax.set_title(r'A (log$\dot{M}$=%.1f)'%logMdot)
plt.tight_layout()
fig.savefig('analysis/misc_plots/A.png')

fig,ax=plt.subplots(1,1)
ax.loglog(rho,B_pac,label='Paczynski')
ax.loglog(rho,B_new,label='Electrons')
ax.set_xlabel(r'log $\rho$',fontsize=14)
ax.legend()
ax.set_title(r'B (log$\dot{M}$=%.1f)'%logMdot)
plt.tight_layout()
fig.savefig('analysis/misc_plots/B.png')

fig,ax=plt.subplots(1,1)
ax.loglog(rho,C_pac,label='Paczynski')
ax.loglog(rho,C_new,label='Electrons')
# ax.loglog(rho,C_new2,'k--',label='Electrons')
ax.set_xlabel(r'log $\rho$',fontsize=14)
ax.legend()
ax.set_title(r'C (log$\dot{M}$=%.1f)'%logMdot)
plt.tight_layout()
fig.savefig('analysis/misc_plots/C.png')

fig,ax=plt.subplots(1,1)
ax.semilogx(rho,(A_pac-A_new)/A_new,label='A')
ax.semilogx(rho,(B_pac-B_new)/B_new,label='B')
ax.semilogx(rho,(C_pac-C_new)/C_new,label='C')
ax.set_xlabel(r'log $\rho$',fontsize=14)
ax.set_ylabel('Relative error',fontsize=14)
ax.legend()
ax.set_title(r'log$\dot{M}$=%.1f'%logMdot)
plt.tight_layout()
fig.savefig('analysis/misc_plots/ABC_errors.png')

plt.show()