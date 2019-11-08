''' Main code to calculate winds '''

from scipy.optimize import brentq
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import numpy as np
from numpy import linspace, sqrt, log10, array, pi
from IO import load_params
import os
from lsoda_remove import stdout_redirected
use_lsoda_remove = True

# --------------------------------------- Constants and parameters --------------------------------------------

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
    Z=2
    mu_I, mu_e, mu = 4, 2, 4/3
elif comp == 'Ni':
    Z = 28
    mu_I, mu_e = 56, 2
    mu = 1/(1/mu_I + 1/mu_e)

# Nickel 56
# A=56,Z=28	

GM = 6.6726e-8*2e33*M
LEdd = 4*pi*c*GM/kappa0

ZZ = (1-2*GM/(c**2*RNS*1e5))**(-1/2) # redshift
g = GM/(RNS*1e5)**2 * ZZ
P_inner = g*y_inner

rhomax = 1e6

# ----------------------------------------- General Relativity ------------------------------------------------

def gamma(v):
    return 1/sqrt(1-v**2/c**2)

def Swz(r):  # Schwartzchild metric term
    return (1-2*GM/c**2/r)

def Lcrit(r,rho,T): # local critical luminosity
    return LEdd*(kappa0/kappa(rho,T))*Swz(r)**(-1/2)

# -------------------------------------------- Microphysics ----------------------------------------------------

def kappa(rho,T):
    # return kappa0/(1.0+(T/4.5e8)**0.86)     
    return kappa0/(1.0+(T/4.5e8)**0.86) + 1e23*Z**2/(mu_e*mu_I)*rho*T**(-7/2)
    
def cs2(T):  # ideal gas sound speed  c_s^2  
    return kB*T/(mu*mp)

def pressure(rho, T): # ideal gas + radiation pressure (eq 2c)} 
    return rho*cs2(T) + arad*T**4/3.0 

def internal_energy(rho, T):  # ideal gas + radiation
    return 1.5*cs2(T)*rho + arad*T**4 

def Beta(rho, T):  # pressure ratio 
    Pg = rho*cs2(T)
    Pr = arad*T**4/3.0
    return Pg/(Pg+Pr)

# ----------------------------------------- Paczynski&Proczynski ----------------------------------------------

def Y(r, v):  # eq 2a
    return sqrt(Swz(r))*gamma(v)

def Tstar(Lstar, T, r, rho, v):  # eq 2b
    return Lstar/LEdd * kappa(rho,T)/kappa0 * GM/(4*r) * 3*rho/(arad*T**4) * (1+(v/c)**2)**(-1) * Y(r, v)**(-3)

def H(rho, T):  # eq 2c
    return c**2 + (internal_energy(rho, T)+pressure(rho, T))/rho

def A(T):  # eq 5a
    return 1 + 1.5*cs2(T)/c**2

def B(T):
    return cs2(T)

def C(Lstar, T, r, rho, v):  # eq 5c
    return Tstar(Lstar, T, r, rho, v) * (4-3*Beta(rho, T))/(1-Beta(rho, T)) * arad*T**4/(3*rho)


# ------------------------------------- Degenerate electron corrections ---------------------------------------
# We use these corrections when integrating to high densities (below sonic point, going towards the surface)

def electrons(rho,T):  # From Paczynski (1983) semi-analytic formula : ApJ 267 315

    rY = rho/mu_e # rho*Ye = rho/mu)e
    pednr = 9.91e12 * (rY)**(5/3)     
    pedr = 1.231e15 * (rY)**(4/3)
    ped = 1/sqrt((1/pedr**2)+(1/pednr**2))
    pend = kB/mp*rY*T
    pe = sqrt(ped**2 + pend**2) # pressure
    
    f = 5/3*(ped/pednr)**2 + 4/3*(ped/pedr)**2
    Ue = pe/(f-1)               # energy density (erg cm-3)

    alpha1,alpha2 = (pend/pe)**2 , (ped/pe)**2
    
    return pe,Ue,[alpha1,alpha2,f]

def cs2_I(T):  # ideal gas sound speed c_s^2 IONS ONLY
    return kB*T/(mu_I*mp)

def pressure_e(rho, T): # ideal gas + radiation pressure (eq 2c)}  PLUS(new)  electron pressure (non-degen + degen)
    pe,_,_ = electrons(rho,T)
    return rho*cs2_I(T) + arad*T**4/3.0 + pe

def internal_energy_e(rho, T):  # ideal gas + radiation + electrons 
    _,Ue,_ = electrons(rho,T)
    return 1.5*cs2_I(T)*rho + arad*T**4 + Ue

def Beta_I(rho, T):
    pg = rho*cs2_I(T)
    return pg/pressure_e(rho,T)

def Beta_e(rho, T):
    pe,_,_ = electrons(rho,T)
    return pe/pressure_e(rho,T)

def H_e(rho, T):  # eq 2c
    return c**2 + (internal_energy_e(rho, T)+pressure_e(rho, T))/rho

def A_e(rho,T):  
    pe,_,[alpha1,alpha2,f] = electrons(rho,T)
    return 1 + 1.5*cs2_I(T)/c**2 + pe/(rho*c**2)*(f/(f-1) - alpha1)

def B_e(rho,T): 
    pe,_,[alpha1,alpha2,f] = electrons(rho,T)
    return cs2_I(T) + pe/rho*(alpha1 + alpha2*f)

def C_e(Lstar, T, r, rho, v):  
    pe,_,[alpha1,alpha2,f] = electrons(rho,T)
    bi,be = Beta_I(rho, T), Beta_e(rho, T)
    return Tstar(Lstar, T, r, rho, v) * ((4-3*bi-(4-alpha1)*be)/(1-bi-be)) * arad*T**4/(3*rho)
# -------------------------------------------- u(phi) ------------------------------------------------------

def uphi(phi, T, inwards):
    ''' phi should never drop below 2, but numerically
    it is sometimes just < 2, so in that case return Mach number 1 (divided by sqrt(A))
    This is using the GR version of the Joss & Melia change of variable phi in sonic units,
    where the difference between the velocity and sound speed at the critical point
    (vs=sqrt(B)/sqrt(A)=0.999cs) is taken into account : phi = sqrt(A)*mach + 1/sqrt(A)/mach '''

    if phi < 2.0:   
        u = 1.0*sqrt(B(T)/sqrt(A(T)))
    else:
        if inwards:
            u = 0.5*phi*sqrt(B(T))*(1.0-sqrt(1.0-(2.0/phi)**2))/sqrt(A(T))
        else:
            u = 0.5*phi*sqrt(B(T))*(1.0+sqrt(1.0-(2.0/phi)**2))/sqrt(A(T))
    return u

# -------------------------------------------- Sonic point -------------------------------------------------

def numerator(r, T, v, verbose=False):  # numerator of eq (4a)

    rho = Mdot/(4*pi*r**2*Y(r, v)*v)     # eq 1a
    Lstar = Edot-Mdot*H(rho, T)*Y(r, v) + Mdot*c**2   # eq 1c, actually modified for the Mdot*c**2 (ok, just means the definition of Edot is changed)

    return gamma(v)**(-2) * (   GM/r/Swz(r) * (A(T)-B(T)/c**2) - C(Lstar, T, r, rho, v) - 2*B(T))

def rSonic(Ts):
    ''' Finds the sonic point radius corresponding to the sonic point temperature Ts (in GR it's exactly the "sonic" point, see Paczynski) '''

    rkeep1, rkeep2 = 0.0, 0.0
    npoints = 10
    vs = sqrt(cs2(Ts)/sqrt(A(Ts)))
    while rkeep1 == 0 or rkeep2 == 0:
        logr = linspace(6, 9, npoints)
        for r in 10**logr:
            foo = numerator(r, Ts, vs)
            if foo < 0.0:
                rkeep1 = r
            if foo > 0.0 and rkeep2 == 0:
                rkeep2 = r
        npoints += 10

    rs = brentq(numerator, rkeep1, rkeep2, args=(
        Ts, vs ), xtol=1e-10, maxiter=10000)
    return rs

# ---------------------------------------- Calculate vars and derivatives -----------------------------------

def calculateVars_phi(r, T, phi, inwards=False, return_all=False):  

    # At the end phi will be given as an array, so we need these lines to allow this
    if isinstance(phi, (list, tuple, np.ndarray)):
        u = []
        for i in range(len(phi)):
            u.append(uphi(phi[i], T[i], inwards))
        r, T, u = array(r), array(T), array(u)
    else:
        u = uphi(phi, T, inwards)

    rho = Mdot/(4*pi*r**2*u*Y(r, u))

    Lstar = Edot-Mdot*H(rho, T)*Y(r, u) + Mdot*c**2     # eq 1c, actually modified for the Mdot*c**2 (it's fine if we stay consistent)
    if not return_all:
        return u, rho, phi, Lstar
    else:
        L = Lstar/(1+u**2/c**2)/Y(r, u)**2
        LEdd_loc = Lcrit(r,rho,T)
        E = internal_energy(rho, T)
        P = pressure(rho, T)
        cs = sqrt(cs2(T))
        tau = rho*kappa(rho,T)*r
        return u, rho, phi, Lstar, L, LEdd_loc, E, P, cs, tau

def calculateVars_rho(r, T, rho, return_all=False): # Will consider degen electrons

    # At the end rho will be given as an array, so we need these lines to allow this
    if isinstance(rho, (list, tuple, np.ndarray)):
        r, T, rho = array(r), array(T), array(rho)  # to allow mathematical operations

    u = Mdot/sqrt((4*pi*r**2*rho)**2*Swz(r) + (Mdot/c)**2)
    mach = u/sqrt(B_e(rho,T))
    phi = sqrt(A_e(rho,T))*mach + 1/(sqrt(A_e(rho,T))*mach)

    Lstar = Edot-Mdot*H_e(rho, T)*Y(r, u) + Mdot*c**2     # eq 1c, actually modified for the Mdot*c**2 (need to verify this)
    if not return_all:
        return u, rho, phi, Lstar
    else:
        L = Lstar/(1+u**2/c**2)/Y(r, u)**2
        LEdd_loc = Lcrit(r,rho,T)
        E = internal_energy_e(rho, T)
        P = pressure_e(rho, T)
        cs = sqrt(cs2(T))
        tau = rho*kappa(rho,T)*r
        return u, rho, phi, Lstar, L, LEdd_loc, E, P, cs, tau

        

def dr(inic, r, inwards):
    ''' Calculates the derivatives of T and phi with r as the independent variable '''

    T, phi = inic[:2]
    u, rho, phi, Lstar = calculateVars_phi(r, T, phi=phi, inwards=inwards)

    ## OPTION 1
    # if r == rs or phi < 2:
    #     dlnT_dlnr = -1  # avoid divergence at sonic point
    # else:               # (eq. 4a&b)
    #     dlnv_dlnr = numerator(r, T, u) / (B(T) - u**2*A(T))
    #     dlnT_dlnr = -Tstar(Lstar, T, r, rho, u) - 1/Swz(r) * GM/c**2/r \
    #          - gamma(u)**2*(u/c)**2*dlnv_dlnr

    ## OPTION 2
    # if r == rs or phi < 2:
    #     dlnT_dlnr = -Tstar(Lstar, T, r, rho, u) - 1/Swz(r) * GM/c**2/r
    # else:               # (eq. 4a&b)
    #     dlnv_dlnr = numerator(r, T, u) / (B(T) - u**2*A(T))
    #     dlnT_dlnr = -Tstar(Lstar, T, r, rho, u) - 1/Swz(r) * GM/c**2/r \
    #          - gamma(u)**2*(u/c)**2*dlnv_dlnr


    # OPTION 3
    dlnT_dlnr = -Tstar(Lstar, T, r, rho, u) - 1/Swz(r) * GM/c**2/r


    # Note : Options 1&2 both run into numerical problems because dlnv_dlnr diverges not just at the
    # sonic point but also in its vicinity.  It therefore completely dominates the dlnT_dlnr term
    # when in reality it should be negligible because of the (u/c)**2 term.  Option 3 is the best to 
    # avoid numerical problems, and no significant loss in precision or accuracy is made by ignoring
    # the dlnv_dlnr term.

    dT_dr = T/r * dlnT_dlnr

    # cs, rootA = sqrt(B(T)), sqrt(A(T))
    # mach = u/cs
    # dphi_dr = dT_dr * (3/(4*rootA*T) * (cs/c)**2 * (mach-1/mach/A(T)) + cs/2/T * (-rootA*mach/cs + 1/rootA/u)) -\
    #     1/(rootA*cs*r*u) * numerator(r, T, u)
    
    mach = u/sqrt(B(T))
    dphi_dr = (A(T)*mach**2-1)*(3*B(T)-2*A(T)*c**2)/(4*mach*A(T)**(3/2)*c**2*r) * dlnT_dlnr - numerator(r, T, u)/(u*r*sqrt(A(T)*B(T)))
    
    # dphi_dr3 = dT_dr * (3/(4*sqrt(A(T))*T) * B(T)/c**2 * (mach-1/mach/A(T)) + sqrt(B(T))/2/T * (-sqrt(A(T)/B(T))*mach + 1/sqrt(A(T))/u)) -\
    #     1/(sqrt(A(T)*B(T))*r*u) * numerator(r, T, u)
    
    # print(dphi_dr/dphi_dr2)

    return [dT_dr, dphi_dr]


def drho(inic, rho):
    ''' Calculates the derivatives of T and r with rho as the independent variable 
        Considering degenerate electrons '''

    T, r = inic[:2]
    u, rho, phi, Lstar = calculateVars_rho(r, T, rho = rho)

    # Not using phi
    dlnT_dlnr = -Tstar(Lstar, T, r, rho, u) - 1/Swz(r) * GM/c**2/r
    dT_dr = T/r * dlnT_dlnr

    # eq 6 from Paczynski
    dlnr_dlnrho = (B_e(rho,T)-A_e(rho,T)*u**2) / ((2*u**2 - (GM/(r*Y(r, u)**2))) * A_e(rho,T) + C_e(Lstar, T, r, rho, u)) 
                                         
    dr_drho = r/rho * dlnr_dlnrho
    dT_drho = dT_dr * dr_drho

    return [dT_drho, dr_drho]

# -------------------------------------------- Integration ---------------------------------------------

def outerIntegration(Ts, returnResult=False):
    ''' Integrates out from the sonic point until the photosphere is reached '''

    if verbose:
        print('\n**** Running outerIntegration ****')

    inic = [Ts, 2.0]
    r_outer = 50*rs
    r = np.linspace(rs, r_outer, 5000)
    if use_lsoda_remove:
        with stdout_redirected():
            result,info = odeint(dr, inic, r, args=(False,), atol=1e-6,
                            rtol=1e-6,full_output=True)  # contains T(r) and phi(r)
    else:
        result,info = odeint(dr, inic, r, args=(False,), atol=1e-6,
                        rtol=1e-6,full_output=True)  # contains T(r) and phi(r)
        

    # odeint does not support stop conditions (should switch to scipy.integrate.ode, dopri5 integrator, with solout option)
    # For now, just cut the solution where the first NaN appears
    if True in np.isnan(result[:, 0]):
        firstnan = min([np.argwhere(np.isnan(result[:, i]) == True)[0][0]
                        for i in (0, 1) if True in np.isnan(result[:, i])])  # It works!
        result = result[:firstnan]

    flag = 1
    L1, L2, tau = 0, 0, []

    for counter, phi in enumerate(result[:, 1]):

        rr = r[counter]
        T = result[counter, 0]

        if T < 1e3:
            flag = 2
            if verbose:
                print('It has gotten too cold, exiting before NaNs appear')
                print('Last three values of T : %.1e -> %.1e -> %.1e' %
                      (result[counter-2, 0], result[counter-1, 0], T))
            break

        u, rho, phi, Lstar = calculateVars_phi(rr, T, phi=phi, inwards=False)
        L = Lstar/(1+(u/c)**2)/Y(rr, u)**2  # eq. 3 (comoving luminosity)
        tau.append(rho*kappa(rho,T)*rr)

        if flag and tau[-1] <= tau_out:
            flag = 0
            L1, L2 = L, 4.0*pi*rr**2*sigmarad*T**4
            photosphere = counter
            break

        # if tau increases, the solution is no longer physical
        if counter > 2 and tau[-1] > tau[-2]:
            flag = 2
            break

    if flag != 0:
        if verbose:
            print("tau = ", tau_out,
                  " never reached! Minimum tau reached :", tau[-2])
        if flag == 2:
            if verbose:
                print("Phi started to increase at logr = %.2f" % log10(rr))

    else:
        if verbose:
            print('Found photosphere at log10 r = %.2f' % log10(rr))

    if returnResult:
        if flag == 0:
            return r[:photosphere+1], result[:photosphere+1, :]
        else:
            return r[:counter+1], result[:counter+1, :]

    else:
        if L1 == 0 or L2 == 0:
            if flag == 2:     # we might want to decide what to do based on what happened to phi
                return +100
            else:
                return +200
        else:
            return (L2 - L1) / (L1 + L2)      # Boundary error #1


def innerIntegration_r(r, Ts):
    ''' Integrates in from the sonic point to 0.95rs, using r as the independent variable '''
    if verbose:
        print('\n**** Running innerIntegration R ****')
    inic = [Ts, 2.0]
    result,info = odeint(dr, inic, r, args=(True,), atol=1e-6,
                    rtol=1e-6,full_output=True)  # contains T(r) and phi(r)
    return result


def innerIntegration_rho(rho, r95, T95, returnResult=False):
    ''' Integrates in from 0.95rs, using rho as the independent variable, until rho=10^4 
        We want to match the location of p=p_inner to the NS radius '''

    if verbose:
        print('\n**** Running innerIntegration RHO ****')

    inic = [T95, r95]
    if use_lsoda_remove:
        with stdout_redirected(): 
            result,info = odeint(drho, inic, rho, atol=1e-6, 
                            rtol=1e-6,full_output=True)  # contains T(rho) and r(rho)
    else:
        result,info = odeint(drho, inic, rho, atol=1e-6, 
                        rtol=1e-6,full_output=True)  # contains T(rho) and r(rho)
    flag = 0

    # Removing NaNs
    nanflag=0
    if True in np.isnan(result):
        nanflag=1
        firstnan = min([np.argwhere(np.isnan(result[:, i]) == True)[0][0]
                        for i in (0, 1) if True in np.isnan(result[:, i])])
        result = result[:firstnan]
        rho = rho[:firstnan]
        if verbose:
            print('Inner integration : NaNs reached after rho = %.2e'%rho[firstnan-1])

    # Obtaining pressure
    T, r = result[:, 0], result[:, 1]
    _,_,_,_,_,_,_,P,_,_ = calculateVars_rho(r, T, rho=rho, return_all=True)
        

    # Checking that we reached surface pressure
    if P[-1]<P_inner:
        flag = 1
        if verbose:
            if nanflag: print('Surface pressure never reached (NaNs before reaching p_inner)')
            else:       print('Surface pressure never reached (max rho too small)')

    else: # linear interpolation to find the exact radius where P=P_inner
        x = np.argmin(np.abs(P_inner-P))
        a,b = (x,x+1) if P_inner-P[x] > 0 else (x-1,x)
        func = interp1d([P[a],P[b]],[r[a],r[b]])
        RNS_calc = func(P_inner)

        result = result[:b]

        # print(RNS_calc)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.loglog(r,P,'k.-')
        # plt.loglog([r[0],r[-1]],[P_inner,P_inner],'k--')
        # plt.loglog([r[a],r[b]],[P[a],P[b]],'ro')
        # plt.loglog([RNS_calc],[P_inner],'bo')
        # plt.show()


    if returnResult:
        return result
    else:
        if flag:
            return +100
        else:
            return (RNS_calc/1e5-RNS)/RNS       # Boundary error #2

# ------------------------------------------------- Wind ---------------------------------------------------


def MakeWind(params, logMdot, mode='rootsolve', Verbose=0):
    ''' Obtaining the wind solution for set of parameters Edot/LEdd and log10(Ts).
        The modes are rootsolve : not output, just obtaining the boundary errors, 
        and wind : obtain the full solutions.   '''

    global Mdot, Edot, rs, verbose
    Mdot, Edot, Ts, verbose = 10**logMdot, params[0]*LEdd, 10**params[1], Verbose

    # Start by finding the sonic point
    rs = rSonic(Ts)
    
    if verbose:
        print('\nFor log10Ts = %.2f, located sonic point at log10r = %.2f' %
              (log10(Ts), log10(rs)))

    if mode == 'rootsolve':

        # First error is given by the outer luminosity
        error1 = outerIntegration(Ts)

        if error1 == 100:        # don't bother integrating inwards
            return [100, 100]

        else:

            # First inner integration
            r95 = 0.95*rs
            r_inner1 = np.linspace(rs, r95, 1000)
            result_inner1 = innerIntegration_r(r_inner1, Ts)
            T95, phi95 = result_inner1[-1, :]
            _, rho95, _, _ = calculateVars_phi(r95, T95, phi=phi95, inwards=True)

            # Second inner integration
            rho_inner2 = np.logspace(log10(rho95), log10(rhomax), 2000)
            error2 = innerIntegration_rho(rho_inner2, r95, T95)

            return error1, error2

    elif mode == 'wind':  # Same thing but calculate variables and output all of the arrays

        # Outer integration
        r_outer, result_outer = outerIntegration(Ts, returnResult=True)
        T_outer, phi_outer = result_outer[:, 0], result_outer[:, 1]
        _, rho_outer, _, _ = calculateVars_phi(
            r_outer, T_outer, phi=phi_outer, inwards=False)

        # First inner integration
        r95 = 0.95*rs
        r_inner1 = np.linspace(rs, r95, 1000)
        result_inner1 = innerIntegration_r(r_inner1, Ts)
        T_inner1, phi_inner1 = result_inner1[:, 0], result_inner1[:, 1]
        T95, phi95 = result_inner1[-1, :]
        _, rho_inner1, _, _ = calculateVars_phi(
            r_inner1, T_inner1, phi=phi_inner1, inwards=True)
        rho95 = rho_inner1[-1]
        # r_inner1,T_inner1,rho_inner1 = r_inner1[1:],T_inner1[1:],rho_inner1[1:]  # remove first point(sonic) because duplicate with r_outer

        # Second inner integration 
        npoints = 2000
        rho_inner2 = np.logspace(log10(rho95), log10(rhomax), npoints)
        result_inner2 = innerIntegration_rho(
            rho_inner2, r95, T95, returnResult=True)
        T_inner2, r_inner2 = result_inner2[:, 0], result_inner2[:, 1]
        
        if len(result_inner2)<npoints:
            rho_inner2 = rho_inner2[:len(result_inner2)]
        # r_inner2,T_inner2,rho_inner2 = r_inner2[1:],T_inner2[1:],rho_inner2[1:]  # remove first point(sonic) because duplicate with r_outer

        # Attaching arrays for r,rho,T from surface to photosphere   (ignoring first point in both arrays because duplicates at r=rs and r=r95)
        r_inner = np.append(np.flip(r_inner2[1:], axis=0),
                            np.flip(r_inner1[1:], axis=0))
        T_inner = np.append(np.flip(T_inner2[1:], axis=0),
                            np.flip(T_inner1[1:], axis=0))
        rho_inner = np.append(np.flip(rho_inner2[1:], axis=0),
                              np.flip(rho_inner1[1:], axis=0))

        R = np.append(r_inner, r_outer)
        T = np.append(T_inner, T_outer)
        Rho = np.append(rho_inner, rho_outer)

        # Calculate the rest of the vars
        u, Rho, Phi, Lstar, L, LEdd_loc, E, P, cs, tau = calculateVars_rho(
            R, T, rho=Rho, return_all=True)

        return R, T, Rho, u, Phi, Lstar, L, LEdd_loc, E, P, cs, tau, rs, Edot, Ts






# ------------------------------------------------- Checking solutions and tests ---------------------------------------------------


import matplotlib.pyplot as plt
from matplotlib import gridspec
def plot_stuff(radius,T_points,phi_points,T_func,phi_func,dT_points,dphi_points,title):
    '''T_points and phi_points are the actual points from the solution (same for drho and dT)
       T_points and phi_func are some kind of fit of the date, like a spline, but they HAVE TO have a .derivative method()'''

    fig= plt.figure(figsize=(12,8))
    fig.suptitle(title,fontsize=15)

    gs = gridspec.GridSpec(3, 2, height_ratios=[3, 3, 2])
    ax = []
    for i in range(6): ax.append(plt.subplot(gs[i]))
    ax1,ax2,ax3,ax4,ax5,ax6 = ax

    ax1.set_ylabel(r'log $\rho$ (g cm$^{-3}$)',fontsize=14)
    ax2.set_ylabel(r'log T (K)',fontsize=14)
    ax3.set_ylabel(r'log |$d\rho/dr$|',fontsize=14)
    ax4.set_ylabel(r'log |$dT/dr$|',fontsize=14)
    ax5.set_ylabel('Rel. error (%)',fontsize=14)
    ax6.set_ylabel('Rel. error (%)',fontsize=14)
    ax5.set_xlabel(r'log $r$ (km)',fontsize=14)
    ax6.set_xlabel(r'log $r$ (km)',fontsize=14)
    ax5.set_ylim([-10,10])
    ax6.set_ylim([-10,10])

    x=radius/1e5
    ax1.plot(x,np.log10(T_points),'k.',label='Solution',ms=6,alpha=0.5)
    ax1.plot(x,np.log10(T_points(radius)),'b-',label='Fit')
    ax2.plot(x,np.log10(phi_points),'k.',label='Solution',ms=6,alpha=0.5)
    ax2.plot(x,np.log10(phi_func(radius)),'b-',label='Fit')
    ax3.plot(x,np.log10(np.abs(T_points.derivative()(radius))),'b-',label='Fit derivative')
    ax3.plot(x,np.log10(np.abs(dT_points)),'k.',label='Direct derivative',ms=6,alpha=0.5)
    ax4.plot(x,np.log10(np.abs(phi_func.derivative()(radius))),'b-',label='Fit derivative')
    ax4.plot(x,np.log10(np.abs(dphi_points)),'k.',label='Direct derivative',ms=6,alpha=0.5)
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()

    # Errors
    relerr_rho = (dT_points-T_points.derivative()(radius))/dT_points
    relerr_T = (dphi_points-phi_func.derivative()(radius))/dphi_points
    ax5.plot(x,relerr_rho*100,'k-',lw=1.5)
    ax6.plot(x,relerr_T*100,'k-',lw=1.5)

    plt.tight_layout(rect=(0,0,1,0.95))




from scipy.interpolate import InterpolatedUnivariateSpline as IUS
def check_solution(logMdot, sol):

    ''' checks solution vector's direct derivatives against analytic expressions '''
    
    global Mdot, Edot, rs, verbose
    R, T, Rho, u, Phi, Lstar, L, LEdd_loc, E, P, cs, tau, rs, Edot, Ts = sol

    Mdot, verbose = 10**logMdot, 0

    # Spline fit
    fT,fphi = IUS(R,T), IUS(R,T)
    
    # Analytical derivatives
    dT,dphi = [],[]
    for ri,Ti,phii  in zip(R,T,Phi):
        inwards = True if ri<rs else False
        z = dr([Ti,phii],ri,inwards=inwards)
        dT.append(z[0])
        dphi.append(z[1])

    plot_stuff(R,T,Phi,fT,fphi,dT,dphi,'Error')

    






# For testing when making modifications to this script

# use_lsoda_remove = 0
# from IO import load_roots
# x,z = load_roots()

# Just one solution
# err1,err2=MakeWind(z[20],x[20], Verbose=True)
# # err1,err2=MakeWind([1.02,7.1],18.9)
# print(err1,err2) 

# All solutions
# for logmdot,root in zip(x,z):
#    err1,err2=MakeWind(root,logmdot)
#    print('%.3e \t-\t %.3e\n'%(err1,err2))