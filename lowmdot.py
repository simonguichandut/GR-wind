# Trying to understand what doesn't work at low mdot, higher than some value of Edot

from wind_GR import *

# logMdot, params = 18, [1.015,6.95]              # works
# logMdot, params = 18, [1.05,6.95]              # doesn't work


# logMdot,params = 17.2, [1.015,7.1]               # works
# logMdot,params = 17.2, [1.02,7.1]               # works
logMdot,params = 17.2, [1.024,7.1]               # doesnt work
# logMdot,params = 17.2, [1.025,7.1]               # doesnt work
# logMdot,params = 17.2, [1.03,7.1]             # doesnt work


# global Mdot, Edot, rs, Ts, verbose
Mdot, Edot, Ts, verbose = 10**logMdot, params[0]*LEdd, 10**params[1], 1
setup_globals(params,logMdot,Verbose=1)

rs = rSonic(Ts)
print('For log10Ts = %.2f, located sonic point at log10r = %.2f' %
        (log10(Ts), log10(rs)))


# Quantities at sonic point
vs = sqrt(eos.cs2(Ts)/A(Ts))
rhos = Mdot/(4*pi*rs**2*Y(rs, vs)*vs)     # eq 1a
Lstars = Edot-Mdot*eos.H(rhos, Ts)*Y(rs, vs) + Mdot*c**2     # eq 1c, actually modified for the Mdot*c**2 (it's fine if we stay consistent)
Ls = Lcomoving(Lstars,rs,vs)
Thetas = Tstar(Lstars,Ts,rs,rhos,vs)

print('VALUES AT rs')
print('v: %.3e'%vs)
print('rho: %.3e'%rhos)
print('Lstar: %.3e'%Lstars)
print('L: %.3e'%Ls)
print('Thetas: %.3e'%Ls)


print('derivatives \n',dr_wrapper_subsonic(rs,[Ts,2]),'\n')

result_outer = outerIntegration(returnResult=True)
r_outer = linspace(rs,result_outer.t[-1],2000)
T_outer, phi_outer = result_outer.sol(r_outer)

import matplotlib.pyplot as plt
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,7))
fig.suptitle(r'$\dot{M}$=%.2f . $\dot{E}$=%.3f $L_e$ . log$T_s$=%.3f'%(logMdot,params[0],params[1]))
ax1.set_ylabel('T',fontsize=14)

ax1.plot(r_outer,T_outer,'b-')
ax2.plot(r_outer,phi_outer,'b-')
ax1.plot(result_outer.t,result_outer.y[0],'b.')
ax2.plot(result_outer.t,result_outer.y[1],'b.')
ax2.axhline(2,color='k')
ax2.plot([rs],[2],'ko',mfc=None)
ax1.set_xlim([0.95*rs,1.05*rs])
ax2.set_xlim([0.95*rs,1.05*rs])
ax2.set_ylim([1.97,2.05])




# Inner 1
r95 = 0.95*rs
r_inner1 = linspace(rs, r95, 1000)
result_inner1 = innerIntegration_r()
T_inner1,phi_inner1 = result_inner1.sol(r_inner1)
T95, phi95 = result_inner1.sol(r95)
_, rho95, _, _ = calculateVars_phi(r95, T95, phi=phi95, subsonic=True)

print('rho95 = %.3e\n\n'%rho95)

r,T,phi = result_inner1.t,result_inner1.y[0],result_inner1.y[1]
ax1.plot(r_inner1,T_inner1,'r-')
ax2.plot(r_inner1,phi_inner1,'r-')
ax1.plot(result_inner1.t,result_inner1.y[0],'r.')
ax2.plot(result_inner1.t,result_inner1.y[1],'r.')
ax2.set_ylabel(r'$\phi$',fontsize=14)
# ax3.loglog(r,u), ax3.set_ylabel('u',fontsize=14)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

import sys
if phi95<2:
    plt.show()
    sys.exit("phi goes below 2 in inner integration 1!!")






# result_inner2 = innerIntegration_rho(rho95, T95, returnResult=True)

inic = [T95, 0.95*rs]

def hit_Pinner(rho,y):              # Inner boundary condition
    T = y[0]
    P = eos.pressure_e(rho,T) if EOS_type == 'IGDE' else eos.pressure(rho,T)
    return P-P_inner
hit_Pinner.terminal = True

def hit_zerospeed(rho,y):           # Don't want u to change sign
    r = y[1]
    u = Mdot/sqrt((4*pi*r**2*rho)**2*Swz(r) + (Mdot/c)**2)
    return u
hit_zerospeed.terminal = True      

def hit_zeroT(rho,y):
    return y[0]

# Issue wiht solve_ivp in scipy 1.3.0 (fixed in yet to be released 1.4.0) https://github.com/scipy/scipy/pull/10802
# Will have a typeError when reaching NaNs, and won't return the result properly.
try:
    result = solve_ivp(drho, (rho95,rho95*10), inic, method='Radau',
                events = (hit_Pinner,hit_zerospeed,hit_zeroT), atol=1e-6, rtol=1e-6, dense_output=True)    # contains T(rho) and r(rho)
except:
    if verbose: print('Surface pressure never reached (NaNs before reaching p_inner)')
    sys.exit("")

# print(result.y)

rho,T,r = result.t,result.y[0],result.y[1]
u = Mdot/sqrt((4*pi*r**2*rho)**2*Swz(r) + (Mdot/c)**2)

import matplotlib.pyplot as plt
fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(17,7))
ax1.loglog(rho,T), ax1.set_ylabel('T',fontsize=14)
ax2.loglog(rho,r), ax2.set_ylabel('r',fontsize=14)
ax3.loglog(rho,u), ax3.set_ylabel('u',fontsize=14)
for ax in (ax1,ax2,ax3): ax.set_xlabel(r'$\rho$',fontsize=14)
plt.tight_layout()
plt.show()