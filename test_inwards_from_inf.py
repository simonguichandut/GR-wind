## Trying to integrate in from a large but finite radius as in Quinn & Paczynski

from wind_GR import *
import IO
import physics
import sys
from numpy import array
import matplotlib.pyplot as plt


M, RNS, y_inner, tau_out, comp, EOS_type, FLD, mode, save, img = IO.load_params()
eos = physics.EOS(comp)
LEdd = 4*pi*c*GM/eos.kappa0


logMdot = 18.5
params=[1.025,7.2]
Mdot = 10**logMdot

if not FLD: raise
winds = MakeWind(params, logMdot, mode='wind',Verbose=0)
w1 = winds[0]

#%%
Edot = w1.Edot

# Variables at infinity
 
# vinf as a free parameter (add to Edot and Ts)
vinf = 10**8.42025          # 10**8.43 ball park best

gammainf = gamma(vinf)
Linf = Edot + Mdot*c**2*(1-gammainf)

## Variables at large radius (r0) as done in Quinn & Pac
r0 = 1e12

#L0 = Lcomoving(Linf,r0,vinf)
#v0 = sqrt(vinf**2 + 2*GM/r0 - eos.kappa0*L0/(2*pi*r0*c))
#
#if v0>vinf: print('does not make sense')
#    
#rho0 = Mdot/(4*pi*r0**2*v0*Y(r0,v0))
#T0 = (L0/(4*pi*r0**2*arad*c))**(0.25)
#
#
## L0 vs the first L that will be calculated
#print('L0: ', L0)
#Lstar_first = Edot - Mdot*eos.H(rho0,T0)*Y(r0,v0) + Mdot*c**2
#L0_first = Lcomoving(Lstar_first,r0,v0)
#print('L0: ', L0_first)


#%% Optimizing v0 and L0 with fsolve

def error(x):
    v0,Lstar0 = x
    L0 = Lcomoving(Lstar0,r0,v0)
    rho0 = Mdot/(4*pi*r0**2*v0*Y(r0,v0))
    T0 = (L0/(4*pi*r0**2*arad*c))**(0.25)
    
    E1 = Y(r0,v0) - gammainf + eos.kappa(rho0,T0)*L0/(4*pi*r0*c**3*Y(r0,v0))   # Integrated momentum equation ignoring Pg'
    E2 = (Edot + Mdot*c**2 - Lstar0 - Mdot*eos.H(rho0,T0)*Y(r0,v0))/LEdd
    
    return [E1,E2]
        

# Errorspace
x,y = np.linspace(0.99*vinf,1.01*vinf,100), np.linspace(0.95*Linf,1.05*Linf,100)
XX,YY = np.meshgrid(x,y)
E1,E2 = error([XX,YY])

#fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,8)) 
#im1 = ax1.contourf(XX/c,YY/LEdd,abs(E1),levels=40,cmap='Reds')
#im2 = ax2.contourf(XX/c,YY/LEdd,abs(E2),levels=40,cmap='Reds')
#fig.colorbar(im1,ax=ax1)
#fig.colorbar(im2,ax=ax2)
#ax1.set_xlabel('v/c')
#ax2.set_xlabel('v/c')
#ax1.set_ylabel(r'L$^*$/Ledd')
#for ax in ax1,ax2:
#    ax.axhline(Linf/LEdd,color='k')
#    ax.axvline(vinf/c,color='k')
#fig.suptitle('black lines = values at infinity. Solution should be in top left quadrant')
#plt.show()


from scipy.optimize import fsolve
x = fsolve(error,x0=[vinf,Linf])

v0,Lstar0=x
L0 = Lcomoving(Lstar0,r0,v0)
rho0 = Mdot/(4*pi*r0**2*v0*Y(r0,v0))
T0 = (L0/(4*pi*r0**2*arad*c))**(0.25)

#%%


def calculateVars_v(r, T, v, return_all=False):  

    mach = v/sqrt(B(T))
    phi = sqrt(A(T))*mach + 1/(sqrt(A(T))*mach)
    rho = Mdot/(4*pi*r**2*v*Y(r, v))
    Lstar = Edot-Mdot*eos.H(rho, T)*Y(r, v) + Mdot*c**2    
    L = Lcomoving(Lstar,r,v)
    if not return_all:
        return rho, L, Lstar
    else:
        LEdd_loc = Lcrit(r,rho,T)
        E = eos.internal_energy(rho, T)
        P = eos.pressure(rho, T)
        cs = sqrt(eos.cs2(T))
        tau = taustar(r,rho,T)
        lam = FLD_Lam(Lstar,r,v,T)
        return rho, phi, Lstar, L, LEdd_loc, E, P, cs, tau, lam
    
    
#%% Thin inwards integration

def dr_thin(r, y):

    T, v = y[:2]
    rho, L, Lstar = calculateVars_v(r, T, v=v)

    dT_dr = -L/(8*pi*r**3*arad*c*T**3)

    taus = taustar(r,rho,T)
    Cthin =  1/Y(r,v) * L/LEdd * eos.kappa(rho,T)/eos.kappa0 * GM/r * ( 1 + Y(r,v)*eos.Beta(rho,T) / (6*taus*(1-eos.Beta(rho,T))) ) 
    Athin = 1 + 2.5*eos.cs2(T)/c**2
    dlnv_dlnr = (GM/r/Swz(r) * (Athin - B(T)/c**2) - 2*B(T) - Cthin) / (B(T) - v**2*Athin)
    dv_dr = v/r * dlnv_dlnr
    
    return [dT_dr, dv_dr]

def inner_integration_thin():
    
    def causality(r,y):
        T, v = y[:2]
        rho, L, _ = calculateVars_v(r, T, v=v)
        flux = L/(4*pi*r**2)
        Er = arad*T**4
        return flux-c*Er
#    causality.terminal = True
   
#    rspan = (r0,0.001*r0)
    rspan = (r0,2*w1.rs)
    result = solve_ivp(dr_thin, rspan, (T0,v0), method='Radau', dense_output=True, events=(causality), rtol=1e-6)    
    return result


    
print('thin integration')
result = inner_integration_thin()
print(result.message,'\n')
#r,T,v = result.t[1:],result.y[0][1:],result.y[1][1:] # exclude first point which has alpha=1 and breaks lambda
r,(T,v)=result.t,result.y
rho, L, Lstar = calculateVars_v(r,T,v)

#rho, phi, Lstar, L, LEdd_loc, E, P, cs, tau, lam = calculateVars_v(r,T,v,return_all=True)
#
#flux = L/(4*pi*r**2)
#Er = arad*T**4
#alpha = flux/(c*Er)



#%% FLD inwards integration

def dr_fld(r,y):
    
    T, v = y[:2]
    rho, L, Lstar = calculateVars_v(r, T, v=v)

    Lam = FLD_Lam(Lstar,r,v,T)

    dlnT_dlnr = -Tstar(Lstar, T, r, rho, v) / (3*Lam) - 1/Swz(r) * GM/c**2/r
#    print('logr = %.3f  : dlnT_dlnr = %.3f \t lambda=%.3e'%(log10(r),dlnT_dlnr,Lam))
        
    dT_dr = T/r * dlnT_dlnr
    
    Cfld = Tstar(Lstar, T, r, rho, v) * ( 4 + eos.Beta(rho, T)/(3*Lam*(1-eos.Beta(rho,T))) ) * arad*T**4/(3*rho) 
    dlnv_dlnr = (GM/r/Swz(r) * (A(T) - B(T)/c**2) - 2*B(T) - Cfld) / (B(T) - v**2*A(T))
    dv_dr = v/r * dlnv_dlnr
    
    return [dT_dr, dv_dr]

def inner_integration_fld():
    
    def causality(r,y):
        T, v = y[:2]
        rho, L, _ = calculateVars_v(r, T, v=v)
        flux = L/(4*pi*r**2)
        Er = arad*T**4
#        print(flux/c/Er)
        return flux-c*Er
    causality.terminal = True
   
#    rspan = (r0,0.01*r0)
    rspan = (r0,2*w1.rs)
    
    # use normal dr which has fld
    result = solve_ivp(dr_fld, rspan, (T0,v0), method='RK45', dense_output=True, events=(causality), rtol=1e-6)    
    return result

print('fld integration')
result2= inner_integration_fld()
print(result2.message,'\n')
r2,(T2,v2)=result2.t,result2.y
rho2, phi2, Lstar2, L2, LEdd_loc2, E2, P2, cs2, tau2, lam2 = calculateVars_v(r2,T2,v2,return_all=True)

flux2 = L2/(4*pi*r2**2)
Er2 = arad*T2**4
alpha2 = flux2/(c*Er2)

#%% Plots

fig,((ax1,ax2,ax3),(ax4,ax5,ax6))=plt.subplots(2,3,figsize=(15,8))
ax1.set_ylabel(r'T (K)',fontsize=16)
ax2.set_ylabel(r'v (cm s$^{-1}$)',fontsize=16)
ax3.set_ylabel(r'$\lambda$',fontsize=16)
ax4.set_ylabel(r'$\rho$',fontsize=16)
ax5.set_ylabel(r'$L/L_{E}$',fontsize=16)
ax6.set_ylabel(r'$\tau^*=\kappa\rho r$',fontsize=16)
for ax in (ax4,ax5,ax6): ax.set_xlabel('r (cm)',fontsize=16)

ax1.loglog(w1.r,w1.T,'k',label='FLD from sonic point')
ax2.loglog(w1.r,w1.u,'k')
ax3.loglog(w1.r,w1.lam,'k')
ax4.loglog(w1.r,w1.rho,'k')
ax5.semilogx(w1.r,w1.L/LEdd,'k')
ax6.loglog(w1.r,w1.tau,'k')

#ax1.loglog(r,T,label=r'Optically thin $\infty$')
#ax2.loglog(r,v)
##ax3.loglog(r,lam)
#ax4.loglog(r,rho)
#ax5.semilogx(r,L/LEdd)
#ax6.loglog(r,tau)


ax1.loglog(r2,T2,label=r'FLD from $\infty$')
ax2.loglog(r2,v2)
ax3.loglog(r2,lam2)
ax4.loglog(r2,rho2)
ax5.semilogx(r2,L2/LEdd)
ax6.loglog(r2,tau2)

ax1.legend()

plt.tight_layout()
plt.show()





