''' 
Main code to calculate winds 
Version with flux-limited diffusion : transitions to optically thin
'''

import os
import sys
import numpy as np
from numpy import linspace, logspace, sqrt, log10, array, pi, gradient
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from collections import namedtuple
import IO
import physics

# --------------------------------------- Constants and parameters --------------------------------------------

# Constants
arad = 7.5657e-15
c = 2.99792458e10
sigmarad = 0.25*arad*c

# Parameters
M, RNS, y_inner, tau_out, comp, EOS_type, FLD, mode, save, img = IO.load_params()
if not FLD: sys.exit('This script is for FLD calculations')

# Generate EOS class and methods
eos = physics.EOS(comp)

# Mass-dependent parameters
GM = 6.6726e-8*2e33*M
LEdd = 4*pi*c*GM / eos.kappa0
ZZ = (1-2*GM/(c**2*RNS*1e5))**(-1/2) # redshift
g = GM/(RNS*1e5)**2 * ZZ
P_inner = g*y_inner

# Maximum density 
rhomax = 1e6

# ----------------------------------------- General Relativity ------------------------------------------------

def gamma(v):
    return 1/sqrt(1-v**2/c**2)

def Swz(r):  # Schwartzchild metric term
    return (1-2*GM/c**2/r)

def Lcrit(r,rho,T): # local critical luminosity
    return LEdd * (eos.kappa0/eos.kappa(rho,T)) *Swz(r)**(-1/2)

# --------------------------------------- Flux-limited diffusion ------------------------------------------------
# Modified version of Pomraning (1983) FLD prescription.  See Guichandut & Cumming (2020)

def FLD_Lam(Lstar,r,v,T):

    if isinstance(Lstar, (list,tuple,np.ndarray)): # for function to be able to take and return array
        Lam = []
        for lstari,ri,vi,Ti in zip(Lstar,r,v,T):
            Lam.append(FLD_Lam(lstari,ri,vi,Ti))
        return array(Lam)

    else:
        L = Lcomoving(Lstar,r,v)
        Flux = L/(4*pi*r**2)
        alpha = Flux/(c*arad*T**4)  # 0 opt thick, 1 opt thin

        if alpha>1:
#            raise Exception
#            print('causality warning : F>cE')
            alpha=1-1e-9

        Lam = 1/12 * ( (2-3*alpha) + sqrt(-15*alpha**2 + 12*alpha + 4) )  # 1/3 thick , 0 thin


        ## Quinn formula
        # YY = Y(r,v)
        # rho = Mdot/(4*pi*r**2*v*YY)
        # Lam = 1/(3 + 2*YY/taustar(r,rho,T))

        return Lam

# ----------------------------------------- Paczynski&Proczynski ------------------------------------------------

def Y(r, v):  # eq 2a
    return sqrt(Swz(r))*gamma(v)

def Lcomoving(Lstar,r,v):
    return Lstar/(1+v**2/c**2)/Y(r, v)**2

def taustar(r,rho,T):
    return rho*eos.kappa(rho,T)*r

def Tstar(Lstar, T, r, rho, v):  # eq 2b
    return Lstar/LEdd * eos.kappa(rho,T)/eos.kappa0 * GM/(4*r) * 3*rho/(arad*T**4) * (1+(v/c)**2)**(-1) * Y(r, v)**(-3)

def A(T):  # eq 5a
    return 1 + 1.5*eos.cs2(T)/c**2

def B(T):
    return eos.cs2(T)

def C(Lstar, T, r, rho, v):  # eq 5c, but modified because of FLD

    Lam = FLD_Lam(Lstar,r,v,T)
    L = Lcomoving(Lstar,r,v)

    return 1/Y(r,v) * L/LEdd * eos.kappa(rho,T)/eos.kappa0 * GM/r * (1 + eos.Beta(rho,T)/(12*Lam*(1-eos.Beta(rho,T))))


# ------------------------------------- Degenerate electron corrections ---------------------------------------
# We use these corrections when integrating to high densities (below sonic point, going towards the surface)

def A_e(rho,T):  
    pe,_,[alpha1,_,f] = eos.electrons(rho,T)
    return 1 + 1.5*eos.cs2_I(T)/c**2 + pe/(rho*c**2)*(f/(f-1) - alpha1)

def B_e(rho,T): 
    pe,_,[alpha1,alpha2,f] = eos.electrons(rho,T)
    return eos.cs2_I(T) + pe/rho*(alpha1 + alpha2*f)

def C_e(Lstar, T, r, rho, v):  
    _,_,[alpha1,_,_] = eos.electrons(rho,T)
    bi,be = eos.Beta_I(rho, T), eos.Beta_e(rho, T)

    return 1/Y(r,v) * Lcomoving(Lstar,r,v)/LEdd * eos.kappa(rho,T)/eos.kappa0 * GM/r * (1 + (eos.Beta_I(rho,T) + alpha1*eos.Beta_e(rho,T))/(12*Lam*(1-eos.Beta_I(rho,T)-eos.Beta_e(rho,T))))

# -------------------------------------------- u(phi) ------------------------------------------------------

def uphi(phi, T, subsonic):
    ''' phi should never drop below 2, but numerically
    it is sometimes just < 2, so in that case return Mach number 1 (divided by sqrt(A))
    This is using the GR version of the Joss & Melia change of variable phi in sonic units,
    where the difference between the velocity and sound speed at the critical point
    (vs=sqrt(B)/sqrt(A)=0.999cs) is taken into account : phi = sqrt(A)*mach + 1/sqrt(A)/mach '''

    if phi < 2.0:   
        u = 1.0*sqrt(B(T)/sqrt(A(T)))
    else:
        if subsonic:
            u = 0.5*phi*sqrt(B(T))*(1.0-sqrt(1.0-(2.0/phi)**2))/sqrt(A(T))
        else:
            u = 0.5*phi*sqrt(B(T))*(1.0+sqrt(1.0-(2.0/phi)**2))/sqrt(A(T))
    return u

# -------------------------------------------- Sonic point -------------------------------------------------

def numerator(r, T, v):  # numerator of eq (4a)
    
    rho = Mdot/(4*pi*r**2*Y(r, v)*v)     # eq 1a
    Lstar = Edot-Mdot*eos.H(rho, T)*Y(r, v) + Mdot*c**2   # eq 1c, actually modified for the Mdot*c**2 (ok, just means the definition of Edot is changed)

    return gamma(v)**(-2) * (   GM/r/Swz(r) * (A(T)-B(T)/c**2) - C(Lstar, T, r, rho, v) - 2*B(T))

def rSonic(Ts):

    rkeep1, rkeep2 = 0.0, 0.0
    npoints = 50
    vs = sqrt(eos.cs2(Ts)/A(Ts))
    while rkeep1 == 0 or rkeep2 == 0:
        logr = linspace(6, 9, npoints)
        for r in 10**logr:
            try:
                foo = numerator(r, Ts, vs)
            except Exception:
                # print('Broke causality (F>cE) when trying sonic point at r=%.3e'%r)
                pass
            else:
                if foo < 0.0:
                    rkeep1 = r
                if foo > 0.0 and rkeep2 == 0:
                    rkeep2 = r
        
        npoints += 10
    # print('sonic: rkeep1 = %.3e \t rkeep2 = %.3e'%(rkeep1,rkeep2))
    global rs
    rs = brentq(numerator, rkeep1, rkeep2, args=(Ts, vs), maxiter=100000)
    return rs

# ---------------------------------------- Calculate vars and derivatives -----------------------------------

def calculateVars_phi(r, T, phi, subsonic=False, return_all=False):  

    # At the end phi will be given as an array, so we need these lines to allow this
    if isinstance(phi, (list, tuple, np.ndarray)):
        u = []
        for i in range(len(phi)):
            u.append(uphi(phi[i], T[i], subsonic))
        r, T, u = array(r), array(T), array(u)
    else:
        u = uphi(phi, T, subsonic)

    rho = Mdot/(4*pi*r**2*u*Y(r, u))

    Lstar = Edot-Mdot*eos.H(rho, T)*Y(r, u) + Mdot*c**2     # eq 1c, actually modified for the Mdot*c**2 (it's fine if we stay consistent)
    if not return_all:
        return u, rho, phi, Lstar
    else:
        L = Lcomoving(Lstar,r,u)
        LEdd_loc = Lcrit(r,rho,T)
        E = eos.internal_energy(rho, T)
        P = eos.pressure(rho, T)
        cs = sqrt(eos.cs2(T))
        tau = taustar(r,rho,T)
        lam = FLD_Lam(Lstar,r,u,T)
        return u, rho, phi, Lstar, L, LEdd_loc, E, P, cs, tau, lam

def calculateVars_rho(r, T, rho, return_all=False): # Will consider degenerate electrons if EOS_type is set to 'IGDE'

    # At the end rho will be given as an array, so we need these lines to allow this
    if isinstance(rho, (list, tuple, np.ndarray)):
        r, T, rho = array(r), array(T), array(rho)  # to allow mathematical operations

    u = Mdot/sqrt((4*pi*r**2*rho)**2*Swz(r) + (Mdot/c)**2)

    if EOS_type == 'IGDE':
        Lstar = Edot-Mdot*eos.H_e(rho, T)*Y(r, u) + Mdot*c**2   
    else:
        Lstar = Edot-Mdot*eos.H(rho, T)*Y(r, u) + Mdot*c**2    

    if not return_all:
        return u, rho, Lstar
    else:

        if EOS_type == 'IGDE':
            mach = u/sqrt(B_e(rho,T))
            phi = sqrt(A_e(rho,T))*mach + 1/(sqrt(A_e(rho,T))*mach)
            E = eos.internal_energy_e(rho, T)
            P = eos.pressure_e(rho, T) 
        else:
            mach = u/sqrt(B(T))
            phi = sqrt(A(T))*mach + 1/(sqrt(A(T))*mach)
            E = eos.internal_energy(rho, T)
            P = eos.pressure(rho, T)    

        L = Lcomoving(Lstar,r,u)
        LEdd_loc = Lcrit(r,rho,T)
        cs = sqrt(eos.cs2(T))
        tau = taustar(r,rho,T)
        lam = FLD_Lam(Lstar,r,u,T)
        return u, rho, phi, Lstar, L, LEdd_loc, E, P, cs, tau, lam

        

def dr(r, y, subsonic):
    ''' Calculates the derivatives of T and phi with r as the independent variable '''

    T, phi = y[:2]
    u, rho, phi, Lstar = calculateVars_phi(r, T, phi=phi, subsonic=subsonic)

    Lam = FLD_Lam(Lstar,r,u,T)
    dlnT_dlnr = -Tstar(Lstar, T, r, rho, u) / (3*Lam) - 1/Swz(r) * GM/c**2/r  # remove small dv_dr term which has numerical problems near sonic point
    dT_dr = T/r * dlnT_dlnr

    mach = u/sqrt(B(T))
    dphi_dr = (A(T)*mach**2-1)*(3*B(T)-2*A(T)*c**2)/(4*mach*A(T)**(3/2)*c**2*r) * dlnT_dlnr - numerator(r, T, u)/(u*r*sqrt(A(T)*B(T)))
    
    return [dT_dr, dphi_dr]

def dr_wrapper_supersonic(r,y): return dr(r,y,subsonic=False)
def dr_wrapper_subsonic(r,y):   return dr(r,y,subsonic=True)

def drho(rho, y):
    ''' Calculates the derivatives of T and r with rho as the independent variable 
        Considering degenerate electrons '''

    T, r = y[:2]
    u, rho, Lstar = calculateVars_rho(r, T, rho = rho)

    # Not using phi
    dlnT_dlnr = -Tstar(Lstar, T, r, rho, u) - 1/Swz(r) * GM/c**2/r
    dT_dr = T/r * dlnT_dlnr

    # eq 6 from Paczynski
    if EOS_type == 'IGDE':
        dlnr_dlnrho = (B_e(rho,T)-A_e(rho,T)*u**2) / ((2*u**2 - (GM/(r*Y(r, u)**2))) * A_e(rho,T) + C_e(Lstar, T, r, rho, u)) 
    else:
        dlnr_dlnrho = (B(T)-A(T)*u**2) / ((2*u**2 - (GM/(r*Y(r, u)**2))) * A(T) + C(Lstar, T, r, rho, u)) 

    dr_drho = r/rho * dlnr_dlnrho
    dT_drho = dT_dr * dr_drho

    return [dT_drho, dr_drho]

# -------------------------------------------- Integration ---------------------------------------------

def outerIntegration(returnResult=False):
    ''' Integrates out from the sonic point until the photosphere is reached '''

    if verbose:
        print('**** Running outerIntegration ****')

    inic = [Ts, 2.0]
    rmax = 50*rs

    # Stopping events
    def hit_mach1(r,y): 
        if r>5*rs:
            return y[1]-2  # mach 1 
        else: 
            return 1
    hit_mach1.terminal = True # stop integrating at this point
   
    def hit_1e8(r,y):
        return uphi(y[1],y[0],subsonic=False)-1e8
    hit_1e8.direction = -1
    hit_1e8.terminal = True
    
    def dv_dr_zero(r,y):
        if r>5*rs:
            return numerator(r,y[0],uphi(y[1],y[0],subsonic=False))
        else:
            return -1
    dv_dr_zero.direction = +1
    dv_dr_zero.terminal = True
        
    # Go
    rmax=1e12
    sol = solve_ivp(dr_wrapper_supersonic, (rs,rmax), inic, method='Radau', 
            events=(dv_dr_zero,hit_mach1,hit_1e8), atol=1e-6, rtol=1e-10, dense_output=True, max_step=1e5)
    
    if verbose: 
            print('FLD outer integration : ',result.message)
            
    return sol


def innerIntegration_r():
    ''' Integrates in from the sonic point to 95% of the sonic point, using r as the independent variable '''
    
    if verbose:
        print('**** Running innerIntegration R ****')

    inic = [Ts, 2.0]

    result = solve_ivp(dr_wrapper_subsonic, (rs,0.95*rs), inic, method='RK45',
                    atol=1e-6, rtol=1e-6, dense_output=True)    # contains T(r) and phi(r)

    if verbose: print(result.message)

    return result


def innerIntegration_rho(rho95, T95, returnResult=False):
    ''' Integrates in from 0.95rs, using rho as the independent variable, until rho=rhomax
        We want to match the location of p=p_inner to the NS radius '''

    if verbose:
        print('**** Running innerIntegration RHO ****')

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

    # Issue wiht solve_ivp in scipy 1.3.0 (fixed in yet to be released 1.4.0) https://github.com/scipy/scipy/pull/10802
    # Will have a typeError when reaching NaNs, and won't return the result properly.
    
    try:
        result = solve_ivp(drho, (rho95,rhomax), inic, method='Radau',
                    events = (hit_Pinner,hit_zerospeed), atol=1e-6, rtol=1e-6, dense_output=True)    # contains T(rho) and r(rho)
    except:
        if verbose: print('Surface pressure never reached (NaNs before reaching p_inner)')
        return +200


    if verbose: print(result.message)

    if result.status == 1 :         # A termination event occured

        if len(result.t_events[0]) == 1:  # The correct termination event occured (P_inner)
            
            rbase = result.y[1][-1]
            if verbose: print('Found base at r = %.2f km\n' % (rbase/1e5))

            if returnResult:
                return result
            else:
                return (rbase/1e5 - RNS)/RNS    # Boundary error #2

        else:
            flag_u0 = 1
            p = hit_Pinner(result.t[-1],result.y[-1]) + P_inner
            col = p/g
            if verbose: print('Zero velocity before pressure condition reached.\
                                Last pressure : %.3e (y = %.3e)\n'%(p,col))
        
    else:
        if verbose: print('Pressure condition nor zero velocity reached. Something else went wrong\n')

    if returnResult:
        return result
    else:
        if flag_u0:
            return +100
        else:
            return +300


# ------------------------------------------------- Wind ---------------------------------------------------

# A named tuple allows us to access arrays by their variable name, while also being able to tuple unpack to get everything
Wind = namedtuple('Wind',['r','T','rho','u','phi','Lstar','L','LEdd_loc','E','P','cs','tau','lam','rs','Edot','Ts'])   

def setup_globals(params,logMdot,Verbose,return_them=False):
    global Mdot, Edot, Ts, verbose
    Mdot, Edot, Ts, verbose = 10**logMdot, params[0]*LEdd, 10**params[1], Verbose
    if return_them:
        return Mdot, Edot, Ts, verbose

def MakeWind(params, logMdot, mode='rootsolve', Verbose=0, IgnoreErrors = False):
    ''' Obtaining the wind solution for set of parameters Edot/LEdd and log10(Ts).
        The modes are rootsolve : not output, just obtaining the boundary errors, 
        and wind : obtain the full solutions.   '''

    setup_globals(params,logMdot,Verbose)

    if verbose: print('\nMaking a wind for logMdot = %.2f, logTs = %.5f, Edot/Ledd = %.5f'%(logMdot,log10(Ts),Edot/LEdd))

    # Start by finding the sonic point
    rs = rSonic(Ts)
    
    if verbose:
        print('For log10Ts = %.2f, located sonic point at log10r = %.2f' %
              (log10(Ts), log10(rs)))

    if mode == 'rootsolve':

        sys.exit("Rootsolving is done in RootFinding_FLD.py")

    elif mode == 'wind':  # Same thing but calculate variables and output all of the arrays

        if FLD:  # FLD only returns the results from outer integration. This is meant to be temporary

            res = outerIntegration(returnResult=True)
            
            r = res.t
            T,phi = res.sol(r)
            u, rho, phi, Lstar, L, LEdd_loc, E, P, cs, tau, lam = calculateVars_phi(r,T,phi,return_all=True)
            wind1 = Wind(r,T,rho,u,phi,Lstar,L,LEdd_loc,E,P,cs,tau,lam,rs,Edot,Ts)
            
            return wind1


#            if len(res)>1:
#                res2 = res[1]
#                r = res2.t
#                T,phi = res2.sol(r)
#                u, rho, phi, Lstar, L, LEdd_loc, E, P, cs, tau, lam = calculateVars_phi(r,T,phi,return_all=True,subsonic=True)
#                wind2 = Wind(r,T,rho,u,phi,Lstar,L,LEdd_loc,E,P,cs,tau,lam,rs,Edot,Ts)
#
#                return [wind1,wind2]
#
#            else:
#                return [wind1]


        else:   

            # Outer integration
            result_outer = outerIntegration(returnResult=True)
            # r_outer = linspace(rs,result_outer.t[-1],2000)
            r_outer = linspace(1.01*rs,result_outer.t[-1],2000)   # ignore data in 1% around rs
            T_outer, phi_outer = result_outer.sol(r_outer)

            # re-add sonic point values
            r_outer, T_outer, phi_outer = np.insert(r_outer, 0, rs), np.insert(T_outer, 0, Ts), np.insert(phi_outer, 0, 2.0)

            _, rho_outer, _, _ = calculateVars_phi(r_outer, T_outer, phi=phi_outer, subsonic=False)

            # First inner integration
            r95 = 0.95*rs
            # r_inner1 = linspace(rs, r95, 500)
            r_inner1 = linspace(0.99*rs, r95, 30)      # ignore data in 1% around rs
            result_inner1 = innerIntegration_r()
            T95, phi95 = result_inner1.sol(r95)
            T_inner1, phi_inner1 = result_inner1.sol(r_inner1)

            _, rho_inner1, _, _ = calculateVars_phi(r_inner1, T_inner1, phi=phi_inner1, subsonic=True)
            rho95 = rho_inner1[-1]

            # Second inner integration 
            result_inner2 = innerIntegration_rho(rho95, T95, returnResult=True)
            rho_inner2 = logspace(log10(rho95) , log10(result_inner2.t[-1]), 2000)
            T_inner2, r_inner2 = result_inner2.sol(rho_inner2)
            

            # Attaching arrays for r,rho,T from surface to photosphere   (ignoring first point in inner2 because duplicate values at r=r95)
            r_inner = np.append(np.flip(r_inner2[1:], axis=0),
                                np.flip(r_inner1, axis=0))
            T_inner = np.append(np.flip(T_inner2[1:], axis=0),
                                np.flip(T_inner1, axis=0))
            rho_inner = np.append(np.flip(rho_inner2[1:], axis=0),
                                np.flip(rho_inner1, axis=0))

            R = np.append(r_inner, r_outer)
            T = np.append(T_inner, T_outer)
            Rho = np.append(rho_inner, rho_outer)

            # Calculate the rest of the vars
            u, Rho, Phi, Lstar, L, LEdd_loc, E, P, cs, tau, lam = calculateVars_rho(R, T, rho=Rho, return_all=True)

            return Wind(R, T, Rho, u, Phi, Lstar, L, LEdd_loc, E, P, cs, tau, lam, rs, Edot, Ts)




# # For testing when making modifications to this script

# x,z = IO.load_roots()

## All solutions
# verbose=0
# for logmdot,root in zip(x,z):
#    err1,err2=MakeWind(root,logmdot,Verbose=verbose)
#    print('%.2f \t\t %.3e \t-\t %.3e\n'%(logmdot,err1,err2))

# # # Just one solution
# # err1,err2=MakeWind(z[20],x[20], Verbose=True)
# # err1,err2=MakeWind([1.02,7.1],18.9)
# # print(err1,err2)