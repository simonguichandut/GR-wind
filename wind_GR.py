''' Main code to calculate winds '''

import os
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

# Minimum lambda before transitionning to optically thin
lambda_min = 2.9e-1

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
            raise Exception
            print('causality')
            alpha=1-1e-9

        # # Set maximum alpha for numerical stability
        # alphamax = 1 - 1e-7
        # if alpha>alphamax:
        #     alpha=alphamax


        Lam = 1/12 * ( (2-3*alpha) + sqrt(-15*alpha**2 + 12*alpha + 4) )  # 1/3 thick , 0 thin
        # Lam = (1-alpha)/3
        # Lam = 1/3*(1-alpha**4)

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

def C(Lstar, T, r, rho, v):  # eq 5c

    if FLD:
        Lam = FLD_Lam(Lstar,r,v,T)

        # if Lam > lambda_min:
        #     return Tstar(Lstar, T, r, rho, v) * (4 - eos.Beta(rho, T) * (4 - 1/(3*Lam)) )/(1-eos.Beta(rho, T)) * arad*T**4/(3*rho) 

        # else:  # Safely optically thin. To avoid numerical problems, just write optically thin expression
        #     tau = taustar(r,rho,T)
        #     return Tstar(Lstar, T, r, rho, v) * ( 1 + Y(r,v)*eos.Beta(rho,T) / (6*tau*(1-eos.Beta(rho,T))) ) * arad*T**4/(3*rho) 

        # return Tstar(Lstar, T, r, rho, v) * (4 - eos.Beta(rho, T) * (4 - 1/(3*Lam)) )/(1-eos.Beta(rho, T)) * arad*T**4/(3*rho) 
        return Tstar(Lstar, T, r, rho, v) * ( 4 + eos.Beta(rho, T)/(3*Lam*(1-eos.Beta(rho,T))) ) * arad*T**4/(3*rho) 


    else:
        return Tstar(Lstar, T, r, rho, v) * (4-3*eos.Beta(rho, T))/(1-eos.Beta(rho, T)) * arad*T**4/(3*rho)     


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

    return Tstar(Lstar, T, r, rho, v) * ((4-3*bi-(4-alpha1)*be)/(1-bi-be)) * arad*T**4/(3*rho)

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
    if FLD:
        Lam = FLD_Lam(Lstar,r,u,T)

        # if Lam > lambda_min:
        #     dlnT_dlnr = -Tstar(Lstar, T, r, rho, u) / (3*Lam) - 1/Swz(r) * GM/c**2/r
        #     print('logr = %.3f  : dlnT_dlnr = %.3f \t lambda=%.3e'%(log10(r),dlnT_dlnr,Lam))

        # else: # Safely optically thin. To avoid numerical problems, just write optically thin expression
        #     dlnT_dlnr = -Lcomoving(Lstar,r,u)/(8*pi*r**2*arad*c*T**4)
        #     print('logr = %.3f  : dlnT_dlnr = %.3f \t lambda=%.3e \t THIN'%(log10(r),dlnT_dlnr,Lam))

        
        # Or just use Fld equation the whole time
        dlnT_dlnr = -Tstar(Lstar, T, r, rho, u) / (3*Lam) - 1/Swz(r) * GM/c**2/r
        print('logr = %.3f  : dlnT_dlnr = %.3f \t lambda=%.3e'%(log10(r),dlnT_dlnr,Lam))
        

    else:
        dlnT_dlnr = -Tstar(Lstar, T, r, rho, u) - 1/Swz(r) * GM/c**2/r


    # Note : Options 1&2 both run into numerical problems because dlnv_dlnr diverges not just at the
    # sonic point but also in its vicinity.  It therefore completely dominates the dlnT_dlnr term
    # when in reality it should be negligible because of the (u/c)**2 term.  Option 3 is the best to 
    # avoid numerical problems, and no significant loss in precision or accuracy is made by ignoring
    # the dlnv_dlnr term.

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


    def hit_mach1(r,y): 
        if r>5*rs:
            return y[1]-2  # mach 1 
        else: 
            return 1
    hit_mach1.terminal = True # stop integrating at this point
   

    if FLD:    # for FLD : stop integrating when mach=1 (phi=2).  Then go a bit further in subsonic region
        
        rmax=1e9
        result1 = solve_ivp(dr_wrapper_supersonic, (rs,rmax), inic, method='Radau', 
                events=hit_mach1, atol=1e-6, rtol=1e-10, dense_output=True, max_step=1e5)

        if verbose: print('First FLD integration message:', result1.message)

        if result1.status == 1 and len(result1.t_events[0]==1):

            rs2 = result1.t[-1] # second sonic point
            inic = result1.sol(rs2)
            assert(inic[1]==2)

            result2 = solve_ivp(dr_wrapper_subsonic, (rs2,rs2+1e6), inic, method='Radau',
                    atol=1e-6, rtol=1e-10, dense_output=True)

            if verbose: print('Second FLD integration message:',result2.message)

            if returnResult:
                return [result1, result2]
        
        else:
            return [result1]


    else:    # Not FLD : stop integrating when tau*=3

        def hit_tau_out(r,y):
            T,phi = y
            u = uphi(phi, T, subsonic=False)
            rho = Mdot/(4*pi*r**2*u*Y(r, u))
            return taustar(r,rho,T) - tau_out
        hit_tau_out.terminal = True


        # Sonic point might be optically thin. Before we integrate, check if we start already at tau<3
        taus = hit_tau_out(rs,inic) + tau_out
        if taus < tau_out:
            print('Sonic point is optically thin! Tau = %.3f'%taus)
            return +400

        # Now go
        result = solve_ivp(dr_wrapper_supersonic, (rs,rmax), inic, method='RK45',
                    events=(hit_tau_out,hit_mach1), atol=1e-6, rtol=1e-6, dense_output=True)

        if verbose: print(result.message)

        if result.status == 1:              # A termination event occured
        
            if len(result.t_events[0]) == 1:    # The correct termination event occured (tau_out)

                rphot,Tphot,phiphot = result.t[-1],result.y[0][-1],result.y[1][-1]
                u, _, _, Lstar = calculateVars_phi(rphot, Tphot, phi=phiphot, subsonic=False)
                L1 = Lcomoving(Lstar,rphot,u)
                L2 = 4.0*pi*rphot**2*sigmarad*Tphot**4

                if verbose: print('Found photosphere at log10 r = %.2f' % log10(rphot))

                if returnResult:
                    return result
                else:
                    return (L2 - L1) / (L1 + L2)      # Boundary error #1
                
                # The rest won't run if we return here
                
            else: 
                flag_mach1 = 1
                if verbose: print('Hit mach 1 before reaching a photosphere at log10 r = %.2f' % log10(result.t[-1]))

  
        #### Further analysis if we did not manage to reach a photosphere
        flag_tauincrease = 0

        tau = []
        for ti,yi in zip(result.t,result.y.transpose()):
            tau.append( hit_tau_out(ti,yi) + tau_out)

        if verbose: print("tau = ", tau_out, " never reached! Minimum tau reached :", min(tau))

        # check if tau started to increase anywhere
        grad_tau = gradient(tau)

        if True in (grad_tau>0):
            flag_tauincrease = 1
            i = np.argwhere(grad_tau>0)[0][0]
            if verbose: print("Tau started to increase at logr = %.2f" % log10(result.t[i]))


        if returnResult:
            return result
        else:
            if flag_tauincrease:
                return +200
            else:
                if flag_mach1:
                    return +100
                else:
                    return +300   # a weird case that probably won't happen : reaching r_outer while never having a tau increase nor reaching phi=2


def innerIntegration_r():
    ''' Integrates in from the sonic point to 95% of the sonic point, using r as the independent variable '''
    
    if verbose:
        print('**** Running innerIntegration R ****')

    inic = [Ts, 2.0]
    # result,_ = odeint(dr, inic, r, args=(True,), atol=1e-6,
    #                 rtol=1e-6,full_output=True)  # contains T(r) and phi(r)

    result = solve_ivp(dr_wrapper_subsonic, (rs,0.95*rs), inic, method='RK45',
                    atol=1e-6, rtol=1e-6, dense_output=True)    # contains T(r) and phi(r)



    # # Trying stepping off sonic point
    # dT,_ = dr_wrapper_subsonic(rs,[Ts,2.0]) # dphi is numerically not zero, which causes problems
    # dr = rs/1000
    # inic = [Ts-dr*dT, 2.0]
    # result = solve_ivp(dr_wrapper_subsonic, (rs-dr,0.95*rs), inic, method='RK45',
    #                 atol=1e-6, rtol=1e-6, dense_output=True)    # contains T(r) and phi(r)


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

def setup_globals(params,logMdot,Verbose):
    global Mdot, Edot, Ts, verbose
    Mdot, Edot, Ts, verbose = 10**logMdot, params[0]*LEdd, 10**params[1], Verbose

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

        if FLD:
            raise TypeError("rootsolving not yet setup for FLD")

        # First error is given by the outer luminosity
        error1 = outerIntegration()

        if error1 in (100,200,300,400) and IgnoreErrors is False:   # don't bother integrating inwards (unless required to)
            return [error1, error1]

        else:

            # First inner integration
            r95 = 0.95*rs
            r_inner1 = linspace(rs, r95, 1000)
            result_inner1 = innerIntegration_r()
            T95, phi95 = result_inner1.sol(r95)
            _, rho95, _, _ = calculateVars_phi(r95, T95, phi=phi95, subsonic=True)

            # Second inner integration
            rho_inner2 = logspace(log10(rho95), log10(rhomax), 2000)
            # error2 = innerIntegration_rho(rho_inner2, T95)
            error2 = innerIntegration_rho(rho95, T95)


            return error1, error2

    elif mode == 'wind':  # Same thing but calculate variables and output all of the arrays

        if FLD:  # FLD only returns the results from outer integration. This is meant to be temporary

            res = outerIntegration(returnResult=True)
            
            res1 = res[0]
            r = res1.t
            T,phi = res1.sol(r)
            u, rho, phi, Lstar, L, LEdd_loc, E, P, cs, tau, lam = calculateVars_phi(r,T,phi,return_all=True)
            wind1 = Wind(r,T,rho,u,phi,Lstar,L,LEdd_loc,E,P,cs,tau,lam,rs,Edot,Ts)

            if len(res)>1:
                res2 = res[1]
                r = res2.t
                T,phi = res2.sol(r)
                u, rho, phi, Lstar, L, LEdd_loc, E, P, cs, tau, lam = calculateVars_phi(r,T,phi,return_all=True,subsonic=True)
                wind2 = Wind(r,T,rho,u,phi,Lstar,L,LEdd_loc,E,P,cs,tau,lam,rs,Edot,Ts)

                return [wind1,wind2]

            else:
                return [wind1]


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