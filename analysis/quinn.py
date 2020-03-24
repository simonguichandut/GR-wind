''' Reproducing results from Quinn & Paczynski (1984) '''

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq,fsolve
from numpy import linspace, logspace, sqrt, log10, array, pi, gradient

# Constants
arad = 7.5657e-15
c = 2.99792458e10
sigmarad = 0.25*arad*c
kB = 1.380658e-16
arad = 7.5657e-15
c = 2.99792458e10
mp = 1.67e-24
kappa0 = 0.2

# Mass-dependent parameters
GM = 6.6726e-8*2e33*1.4
LEdd = 4*pi*c*GM / kappa0
# Composition
mu = 4/3    # ionized He


### Physics
def kappa(T):
    return kappa0/(1.0+(T/4.5e8)**0.86)
def taustar(r,rho,T):
    return kappa(T)*rho*r
def Pg(rho,T):
    return kB*rho*T/(mu*mp)
def cs2(T):
    return kB*T/(mu*mp)
def Pr(T):
    return arad*T**4/3
def P(rho,T):
    return Pg(rho,T) + Pr(T)
def U(rho,T):
    return 1.5*Pg(rho,T) + 3*Pr(T)





### Calculate variables and derivatives
    
def calculateVars_rho(r,T,rho):
    
    v = Mdot/(4*pi*r**2*rho)
    Lr = Edot - Mdot * (v**2/2 - GM/r + (U(rho,T)+P(rho,T))/rho)
    tau = taustar(r,rho,T)
    return v,Lr,tau


def dr_rho(r, y):
    
    T,rho = y[:2]
    v,Lr,tau = calculateVars_rho(r,T,rho)
    
    dT_dr = -3*kappa(T)*rho*Lr/(16*pi*r**2*c*arad*T**3) * (1+2/(3*tau))
    drho_dr = (2*rho*v**2/r - GM*rho/r**2 - Pg(rho,T)/T * dT_dr + kappa(T)*rho*Lr/(4*pi*r**2*c)) / (cs2(T) - v**2)
    
    return [dT_dr,drho_dr]


# Phi version
    
def uphi(phi, T, subsonic):
    
    if isinstance(phi, (list, tuple, np.ndarray)):
        u = []
        for i in range(len(phi)):
            u.append(uphi(phi[i], T[i], subsonic))
        u = np.array(u)

    else:
        if phi < 2.0:   
            u = sqrt(cs2(T))
        else:
            if subsonic:
                u = 0.5*phi*sqrt(cs2(T))*(1.0-sqrt(1.0-(2.0/phi)**2))
            else:
                u = 0.5*phi*sqrt(cs2(T))*(1.0+sqrt(1.0-(2.0/phi)**2))
                
    return u

def calculateVars_phi(r,T,phi,subsonic):
    
    v = uphi(phi,T,subsonic)
    rho = Mdot/(4*pi*r**2*v)
    Lr = Edot - Mdot * (v**2/2 - GM/r + (U(rho,T)+P(rho,T))/rho)
    tau = taustar(r,rho,T)
    return v,rho,Lr,tau
    
def dr_phi(r, y, subsonic):
    
    T,phi = y[:2]
    v,rho,Lr,tau = calculateVars_phi(r,T,phi,subsonic)
    
    dT_dr = -3*kappa(T)*rho*Lr/(16*pi*r**2*c*arad*T**3) * (1+2/(3*tau))
    dv_dr_numerator = -GM/r**2 + 2*cs2(T)/r - cs2(T)/T*dT_dr + kappa(T)*Lr/(4*pi*r**2*c)
    dcs_dr = 1/2*sqrt(cs2(T))/T*dT_dr
    dphi_dr = (1/v-v/cs2(T))*dcs_dr + 1/(v**2*sqrt(cs2(T)))*dv_dr_numerator
    if dphi_dr<0:
        print(dphi_dr)
    
    return [dT_dr,dphi_dr]

def dr_phi_wrapper_subsonic(r,y):   return dr_phi(r,y,subsonic=True)
def dr_phi_wrapper_supersonic(r,y): return dr_phi(r,y,subsonic=False)





    
### Calculate solution
    
# Fixed Parameters
global Mdot,Edot
Mdot,Edot = 3e18, 1.05*LEdd

# Adjustable parameters
rs,vinf = 10**6.8 , 10**8.5


## Integration from sonic point

# Find sonic point parameters
def numerator(r,T,rho):
    v,Lr,tau = calculateVars_rho(r,T,rho)
    dT_dr = -3*kappa(T)*rho*Lr/(16*pi*r**2*c*arad*T**3) * (1+2/(3*tau))
    return (2*rho*v**2/r - GM*rho/r**2 - Pg(rho,T)/T * dT_dr + kappa(T)*rho*Lr/(4*pi*r**2*c))

def Ts_error(Ts):
    vs = sqrt(cs2(Ts))
    rhos = Mdot/(4*pi*rs**2*vs)
    return numerator(rs,Ts,rhos)

Ts_trial = 10**7.3
Ts = fsolve(Ts_error,x0=Ts_trial,xtol=1e-10)[0]
#vs = sqrt(cs2(Ts))
#rhos = Mdot/(4*pi*rs**2*vs)

inic = [Ts,2.0]  # phi=2
result_out = solve_ivp(dr_phi_wrapper_supersonic, t_span=(rs,1e9), y0=inic, method='Radau', dense_output=True, rtol=1e-6)    
r1,(T1,phi1)=result_out.t,result_out.y
v1,rho1,Lr1,tau1 = calculateVars_phi(r1,T1,phi1,subsonic=False)



## Integration from large radius
Linf = Edot-Mdot*vinf**2/2

r0=1e12
Lr0 = Linf/(1+2*vinf/c)
v0 = sqrt(vinf**2 + 2*GM/r0 - kappa0*Lr0/(2*pi*r0*c))
rho0 = Mdot/(4*pi*r0**2*v0)
T0 = (Lr0/(4*pi*r0**2*c*arad))**(0.25)

inic = [T0,rho0]
result_in = solve_ivp(dr_rho, t_span=(r0,rs), y0=inic, method='Radau', dense_output=True, rtol=1e-6)  

r2,(T2,rho2)=result_in.t,result_in.y
v2,Lr2,tau2 = calculateVars_rho(r2,T2,rho2)


#%% Plots
import matplotlib.pyplot as plt

fig,((ax1,ax2,ax3),(ax4,ax5,ax6))=plt.subplots(2,3,figsize=(15,8))
ax1.set_ylabel(r'T (K)',fontsize=16)
ax2.set_ylabel(r'$\phi$',fontsize=16)
ax3.set_ylabel(r'v (cm s$^{-1}$)',fontsize=16)
ax4.set_ylabel(r'$\rho$',fontsize=16)
ax5.set_ylabel(r'$L/L_{E}$',fontsize=16)
ax6.set_ylabel(r'$\tau^*=\kappa\rho r$',fontsize=16)
for ax in (ax4,ax5,ax6): ax.set_xlabel('r (cm)',fontsize=16)

ax1.loglog(r1,T1,'k',label='From sonic point')
ax2.loglog(r1,phi1,'k')
ax3.loglog(r1,v1,'k')
ax4.loglog(r1,rho1,'k')
ax5.semilogx(r1,Lr1/LEdd,'k')
ax6.loglog(r1,tau1,'k')

ax1.loglog(r2,T2,label=r'From $\infty$')
#ax2.loglog(r1,phi1)
#ax3.loglog(r2,v2)
ax4.loglog(r2,rho2)
ax5.semilogx(r2,Lr2/LEdd)
ax6.loglog(r2,tau2)

ax1.legend()

plt.tight_layout()
plt.show()

    
