''' Main code to calculate winds '''

import sys
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from collections import namedtuple
import IO
import physics

# ---------------------- Constants and parameters --------------------------

# Constants
arad = 7.5657e-15
c = 2.99792458e10
sigmarad = 0.25*arad*c

# Parameters
params = IO.load_params()
if params['FLD'] == True: 
    sys.exit('This script is for pure optically thick, not FLD')

# Generate EOS class and methods
eos = physics.EOS(params['comp'])

# Mass-dependent parameters
M,RNS,y_inner = params['M'],params['R'],params['y_inner']
GM = 6.6726e-8*2e33*M
LEdd = 4*np.pi*c*GM / eos.kappa0
ZZ = (1-2*GM/(c**2*RNS*1e5))**(-1/2) # redshift
g = GM/(RNS*1e5)**2 * ZZ
P_inner = g*y_inner

# Maximum density 
rhomax = 1e6

# ------------------------ General Relativity -----------------------------

def gamma(v):
    return 1/np.sqrt(1-v**2/c**2)

def Swz(r):  # Schwartzchild metric term
    return (1-2*GM/c**2/r)

def Lcrit(r,rho,T): # local critical luminosity
    return LEdd * (eos.kappa0/eos.kappa(rho,T)) *Swz(r)**(-1/2)

# ------------------------ Paczynski&Proczynski ----------------------------

def Y(r, v):  # eq 2a
    return np.sqrt(Swz(r))*gamma(v)

def Lcomoving(Lstar,r,v):
    return Lstar/(1+v**2/c**2)/Y(r, v)**2

def taustar(r,rho,T):
    return rho*eos.kappa(rho,T)*r

def Tstar(Lstar, T, r, rho, v):  # eq 2b
    return Lstar/LEdd * eos.kappa(rho,T)/eos.kappa0 * GM/(4*r) *\
            3*rho/(arad*T**4) * (1+(v/c)**2)**(-1) * Y(r, v)**(-3)

def A(T):  # eq 5a
    return 1 + 1.5*eos.cs2(T)/c**2

def B(T):
    return eos.cs2(T)

def C(Lstar, T, r, rho, v):  # eq 5c
    b = eos.Beta(rho,T)
    return Tstar(Lstar, T, r, rho, v) * (4-3*b)/(1-b) * arad*T**4/(3*rho)     


# --------------------- Degenerate electron corrections --------------------
# We use these corrections when integrating to high densities 
# (below sonic point, going towards the surface)

def A_e(rho,T):  
    pe,_,[alpha1,_,f] = eos.electrons(rho,T)
    return 1 + 1.5*eos.cs2_I(T)/c**2 + pe/(rho*c**2)*(f/(f-1) - alpha1)

def B_e(rho,T): 
    pe,_,[alpha1,alpha2,f] = eos.electrons(rho,T)
    return eos.cs2_I(T) + pe/rho*(alpha1 + alpha2*f)

def C_e(Lstar, T, r, rho, v):  
    _,_,[alpha1,_,_] = eos.electrons(rho,T)
    bi,be = eos.Beta_I(rho, T), eos.Beta_e(rho, T)

    return Tstar(Lstar, T, r, rho, v) * \
            ((4-3*bi-(4-alpha1)*be)/(1-bi-be)) * arad*T**4/(3*rho)

# --------------------------------- u(phi) -----------------------------------

def uphi(phi, T, subsonic):
    ''' phi should never drop below 2, but numerically
    it is sometimes just < 2, so in that case return Mach number 1 (divided by 
    sqrt(A)). This is using the GR version of the Joss & Melia change of var
    phi in sonic units, where the difference between the velocity and sound 
    speed at the critical point (vs=sqrt(B)/sqrt(A)=0.999cs) is taken into 
    account : phi = sqrt(A)*mach + 1/sqrt(A)/mach '''

    if phi < 2.0:   
        u = 1.0*np.sqrt(B(T)/np.sqrt(A(T)))
    else:
        if subsonic:
            u = 0.5*phi*np.sqrt(B(T))*\
                (1.0 - np.sqrt(1.0-(2.0/phi)**2)) / np.sqrt(A(T))
        else:
            u = 0.5*phi*np.sqrt(B(T))*\
                (1.0 + np.sqrt(1.0-(2.0/phi)**2)) / np.sqrt(A(T))
    return u

# -------------------------------- Sonic point -------------------------------

def numerator(r, T, v):  # numerator of eq (4a)
    
    rho = Mdot/(4*np.pi*r**2*Y(r, v)*v)     # eq 1a
    Lstar = Edot-Mdot*eos.H(rho, T)*Y(r, v) + Mdot*c**2   
    # eq 1c, but Edot now means Edot + Mc^2

    return gamma(v)**(-2) *\
           (GM/r/Swz(r) * (A(T)-B(T)/c**2) - C(Lstar, T, r, rho, v) - 2*B(T))

def rSonic(Ts):

    rkeep1, rkeep2 = 0.0, 0.0
    npoints = 50
    vs = np.sqrt(eos.cs2(Ts)/A(Ts))
    while rkeep1 == 0 or rkeep2 == 0:
        logr = np.linspace(6, 9, npoints)
        for r in 10**logr:
            try:
                foo = numerator(r, Ts, vs)
            except Exception:
                #print('Broke causality (F>cE) when trying sonic pt at r=%.3e'%r)
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

# -------------------- Calculate vars and derivatives ----------------------

def calculateVars_phi(r, T, phi, subsonic=False, return_all=False):  

    # At the end phi will be given as an array
    if isinstance(phi, (list, tuple, np.ndarray)):
        u = []
        for i in range(len(phi)):
            u.append(uphi(phi[i], T[i], subsonic))
        r, T, u = np.array(r), np.array(T), np.array(u)
    else:
        u = uphi(phi, T, subsonic)

    rho = Mdot/(4*np.pi*r**2*u*Y(r, u))

    Lstar = Edot-Mdot*eos.H(rho, T)*Y(r, u) + Mdot*c**2     
    # eq 1c, modified for the redefinition of Edot
    if not return_all:
        return u, rho, phi, Lstar
    else:
        L = Lcomoving(Lstar,r,u)
        P = eos.pressure(rho, T)
        cs = np.sqrt(eos.cs2(T))
        taus = taustar(r,rho,T)
        return u, rho, phi, Lstar, L, P, cs, taus

def calculateVars_rho(r, T, rho, return_all=False): 
    # Will consider degenerate electrons if EOS_type is set to 'IGDE'

    if isinstance(rho, (list, tuple, np.ndarray)):
        r, T, rho = np.array(r), np.array(T), np.array(rho)  

    u = Mdot/np.sqrt((4*np.pi*r**2*rho)**2*Swz(r) + (Mdot/c)**2)

    if params['EOS_type'] == 'IGDE':
        Lstar = Edot-Mdot*eos.H_e(rho, T)*Y(r, u) + Mdot*c**2   
    else:
        Lstar = Edot-Mdot*eos.H(rho, T)*Y(r, u) + Mdot*c**2    

    if not return_all:
        return u, rho, Lstar
    else:

        if params['EOS_type'] == 'IGDE':
            mach = u/np.sqrt(B_e(rho,T))
            phi = np.sqrt(A_e(rho,T))*mach + 1/(np.sqrt(A_e(rho,T))*mach)
            P = eos.pressure_e(rho, T) 
        else:
            mach = u/np.sqrt(B(T))
            phi = np.sqrt(A(T))*mach + 1/(np.sqrt(A(T))*mach)
            P = eos.pressure(rho, T)    

        L = Lcomoving(Lstar,r,u)
        cs = np.sqrt(eos.cs2(T))
        taus = taustar(r,rho,T)
        return u, rho, phi, Lstar, L, P, cs, taus

        

def dr(r, y, subsonic):
    ''' Calculates the derivatives of T and phi by r '''

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
    dlnT_dlnr = -Tstar(Lstar, T, r, rho, u) - 1/Swz(r) * GM/c**2/r


    # Note : Options 1&2 both run into numerical problems because dlnv_dlnr 
    # diverges not just at the sonic point but also in its vicinity.  It 
    # therefore completely dominates the dlnT_dlnr term when in reality it 
    # should be negligible because of the (u/c)**2 term.  Option 3 is the best 
    # to avoid numerical problems, and no significant loss in precision or 
    # accuracy is made by ignoring the dlnv_dlnr term.

    dT_dr = T/r * dlnT_dlnr

    mach = u/np.sqrt(B(T))
    dphi_dr = (A(T)*mach**2 - 1) *\
        ( 3*B(T) - 2*A(T)*c**2) / (4*mach*A(T)**(3/2)*c**2*r) * dlnT_dlnr \
        - numerator(r, T, u) / (u*r*np.sqrt(A(T)*B(T)) )
    
    return [dT_dr, dphi_dr]

def dr_wrapper_supersonic(r,y): return dr(r,y,subsonic=False)
def dr_wrapper_subsonic(r,y):   return dr(r,y,subsonic=True)

def drho(rho, y):
    ''' Calculates the derivatives of T and r by rho
        Considering degenerate electrons '''

    T, r = y[:2]
    u, rho, Lstar = calculateVars_rho(r, T, rho = rho)

    # Not using phi
    dlnT_dlnr = -Tstar(Lstar, T, r, rho, u) - 1/Swz(r) * GM/c**2/r
    dT_dr = T/r * dlnT_dlnr

    # eq 6 from Paczynski
    if params['EOS_type'] == 'IGDE':
        dlnr_dlnrho = (B_e(rho,T) - A_e(rho,T)*u**2) / \
                        ( (2*u**2 - (GM/(r*Y(r, u)**2))) *\
                        A_e(rho,T) + C_e(Lstar, T, r, rho, u) ) 
    else:
        dlnr_dlnrho = (B(T) - A(T)*u**2) / \
            ((2*u**2 - (GM/(r*Y(r, u)**2))) * A(T) + C(Lstar, T, r, rho, u)) 

    dr_drho = r/rho * dlnr_dlnrho
    dT_drho = dT_dr * dr_drho

    return [dT_drho, dr_drho]

# --------------------------- Integration -----------------------------------

def outerIntegration(returnResult=False):
    ''' Integrates out from the sonic point until the photosphere is reached '''

    if verbose:
        print('**** Running outerIntegration ****')

    inic = [Ts, 2.0]
    rmax = 50*rs

    tausmin = params['tau_out'] # definition of photosphere as taustar

    def hit_mach1(r,y): 
        if r>5*rs:
            return y[1]-2  # mach 1 
        else: 
            return 1
    hit_mach1.terminal = True # stop integrating at this point
   
    def hit_taustarmin(r,y):
        T,phi = y
        u = uphi(phi, T, subsonic=False)
        rho = Mdot/(4*np.pi*r**2*u*Y(r, u))
        return taustar(r,rho,T) - tausmin
    hit_taustarmin.terminal = True


    # Sonic point might be optically thin. 
    # Before we integrate, check if we start already at tau<3
    taus = hit_taustarmin(rs,inic) + tausmin
    if taus < tausmin:
        print('Sonic point is optically thin! Taustar = %.3f'%taus)
        return +400

    # Now go
    result = solve_ivp(dr_wrapper_supersonic, (rs,rmax), inic, method='RK45',
                        events=(hit_taustarmin,hit_mach1), dense_output=True,
                        atol=1e-6, rtol=1e-6)

    if verbose: print(result.message)

    if result.status == 1:   # A termination event occured
    
        if len(result.t_events[0]) == 1: # The correct termination event 

            rph,Tph,phiph = result.t[-1],result.y[0][-1],result.y[1][-1]
            u,_,_,Lstar = calculateVars_phi(rph, Tph, phi=phiph, subsonic=False)
            L1 = Lcomoving(Lstar,rph,u)
            L2 = 4.0*np.pi*rph**2*sigmarad*Tph**4

            if verbose: 
                print('Found photosphere at log10 r = %.2f' % np.log10(rph))

            if returnResult:
                return result
            else:
                return (L2 - L1) / (L1 + L2)      # Boundary error #1
            
            # The rest won't run if we return here
            
        else: 
            flag_mach1 = 1
            if verbose: 
                print('Hit mach 1 before reaching a photosphere at \
                        log10 r = %.2f' % np.log10(result.t[-1]))


    #### Further analysis if we did not manage to reach a photosphere
    flag_tauincrease = 0

    taus = []
    for ti,yi in zip(result.t,result.y.transpose()):
        taus.append( hit_taustarmin(ti,yi) + tausmin)

    if verbose: 
        print("tau=", tausmin, " never reached! Minimum reached :", min(taus))

    # check if tau started to increase anywhere
    grad_taus = np.gradient(taus)

    if True in (grad_taus>0):
        flag_tauincrease = 1
        i = np.argwhere(grad_taus>0)[0][0]
        if verbose: 
            print("Taustar started to increase at logr = %.2f" 
                    % np.log10(result.t[i]))


    if returnResult:
        return result
    else:
        if flag_tauincrease:
            return +200
        else:
            if flag_mach1:
                return +100
            else:
                return +300   
                # a weird case unlikely to happen : reaching r_outer 
                # while never having a tau increase nor reaching phi=2


def innerIntegration_r():
    ''' Integrates in from the sonic point to 95% of the sonic point
        using r as the independent variable '''
    
    if verbose:
        print('**** Running innerIntegration R ****')

    inic = [Ts, 2.0]
    # result,_ = odeint(dr, inic, r, args=(True,), atol=1e-6,
    #                 rtol=1e-6,full_output=True)  # contains T(r) and phi(r)

    result = solve_ivp(dr_wrapper_subsonic, (rs,0.95*rs), inic,
                        method='RK45', dense_output=True,
                        atol=1e-6, rtol=1e-6)    

    # # Trying stepping off sonic point
    # dT,_ = dr_wrapper_subsonic(rs,[Ts,2.0]) 
    # dphi is numerically not zero, which causes problems
    # dr = rs/1000
    # inic = [Ts-dr*dT, 2.0]
    # result = solve_ivp(dr_wrapper_subsonic, (rs-dr,0.95*rs), inic, 
    # method='RK45', atol=1e-6, rtol=1e-6, dense_output=True)    


    if verbose: print(result.message)

    return result


def innerIntegration_rho(rho95, T95, returnResult=False):
    ''' Integrates in from 0.95rs, using rho as the independent variable, 
        until rho=rhomax. Want to match the location of p=p_inner to the RNS '''

    if verbose:
        print('**** Running innerIntegration RHO ****')

    inic = [T95, 0.95*rs]

    def hit_Pinner(rho,y):              # Inner boundary condition
        T = y[0]
        if params['EOS_type'] == 'IGDE':
            P = eos.pressure_e(rho,T) 
        else:
            P = eos.pressure(rho,T)
        return P-P_inner
    hit_Pinner.terminal = True

    def hit_zerospeed(rho,y):           # Don't want u to change sign
        r = y[1]
        u = Mdot/np.sqrt((4*np.pi*r**2*rho)**2*Swz(r) + (Mdot/c)**2)
        return u
    hit_zerospeed.terminal = True        

    # Issue with solve_ivp in scipy 1.3.0 (fixed in yet to be released 1.4.0) 
    # https://github.com/scipy/scipy/pull/10802. # Will have a typeError when 
    # reaching NaNs, and won't return the result properly.
    
    try:
        result = solve_ivp(drho, (rho95,rhomax), inic, method='Radau',
                    events = (hit_Pinner,hit_zerospeed), dense_output=True,
                    atol=1e-6, rtol=1e-6)    
    except:
        if verbose: 
            print('Surface pressure never reached (NaNs before reaching p_inner)')
        return +200


    if verbose: print(result.message)

    flag_u0 = 0
    if result.status == 1 : # A termination event occured

        if len(result.t_events[0]) == 1:  # The correct termination event
            
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
        if verbose: 
            print('Pressure condition nor zero velocity reached.')

    if returnResult:
        return result
    else:
        if flag_u0:
            return +100
        else:
            return +300


# -------------------------------- Wind --------------------------------------

# A named tuple allows us to access arrays by their variable name, 
# while also being able to tuple unpack to get everything
Wind = namedtuple('Wind',
                ['rs','r','T','rho','u','phi','Lstar','L','P','cs','taus'])  

def setup_globals(params,logMdot,Verbose,return_them=False):
    global Mdot, Edot, Ts, verbose
    Mdot, Edot, Ts, verbose = 10**logMdot, params[0]*LEdd, 10**params[1],Verbose
    if return_them:
        return Mdot, Edot, Ts, verbose

def MakeWind(params, logMdot, mode='rootsolve', Verbose=0, IgnoreErrors=False):
    ''' Obtaining the wind solution for set of parameters Edot/LEdd and logTs.
        The modes are rootsolve : no output, just obtaining the boundary errors, 
        and wind : obtain the full solutions. '''

    setup_globals(params,logMdot,Verbose)

    if verbose: 
        print('\nMaking a wind for logMdot=%.2f, logTs=%.5f, Edot/Ledd=%.5f'
                %(logMdot,np.log10(Ts),Edot/LEdd))

    # Start by finding the sonic point
    rs = rSonic(Ts)
    
    if verbose:
        print('For log10Ts = %.2f, located sonic point at log10r = %.2f' %
              (np.log10(Ts), np.log10(rs)))

    if mode == 'rootsolve':

        # First error is given by the outer luminosity
        error1 = outerIntegration()

        if error1 in (100,200,300,400) and IgnoreErrors is False:   
            # don't bother integrating inwards (unless required to)
            return [error1, error1]

        else:

            # First inner integration
            r95 = 0.95*rs
            r_inner1 = np.linspace(rs, r95, 1000)
            result_inner1 = innerIntegration_r()
            T95, phi95 = result_inner1.sol(r95)
            _, rho95, _, _ = calculateVars_phi(r95,T95,phi=phi95,subsonic=True)

            # Second inner integration
            rho_inner2 = np.logspace(np.log10(rho95), np.log10(rhomax), 2000)
            # error2 = innerIntegration_rho(rho_inner2, T95)
            error2 = innerIntegration_rho(rho95, T95)


            return error1, error2

    elif mode == 'wind':  
        # Same thing but calculate variables and output all of the arrays

        # Outer integration
        result_outer = outerIntegration(returnResult=True)
        # r_outer = np.linspace(rs,result_outer.t[-1],2000)
        # ignore data in 1% around rs
        r_outer = np.linspace(1.01*rs,result_outer.t[-1],2000)   
        T_outer, phi_outer = result_outer.sol(r_outer)

        # re-add sonic point values
        r_outer   = np.insert(r_outer, 0, rs)
        T_outer   = np.insert(T_outer, 0, Ts)
        phi_outer = np.insert(phi_outer, 0, 2.0)

        _,rho_outer,_,_ = calculateVars_phi(r_outer, T_outer, phi=phi_outer, subsonic=False)

        # First inner integration
        r95 = 0.95*rs
        # r_inner1 = np.linspace(rs, r95, 500)
        # ignore data in 1% around rs
        r_inner1 = np.linspace(0.99*rs, r95, 30)      
        result_inner1 = innerIntegration_r()
        T95, phi95 = result_inner1.sol(r95)
        T_inner1, phi_inner1 = result_inner1.sol(r_inner1)

        _,rho_inner1,_,_ = calculateVars_phi(r_inner1, T_inner1, phi=phi_inner1, 
                                            subsonic=True)
        rho95 = rho_inner1[-1]

        # Second inner integration 
        result_inner2 = innerIntegration_rho(rho95, T95, returnResult=True)
        rho_inner2 = np.logspace(np.log10(rho95) , np.log10(result_inner2.t[-1]), 2000)
        T_inner2, r_inner2 = result_inner2.sol(rho_inner2)
        

        # Attaching arrays for r,rho,T from surface to photosphere   
        # (ignoring first point in inner2 because duplicate values at r=r95)
        r_inner = np.append(np.flip(r_inner2[1:], axis=0),
                            np.flip(r_inner1, axis=0))
        T_inner = np.append(np.flip(T_inner2[1:], axis=0),
                            np.flip(T_inner1, axis=0))
        rho_inner = np.append(np.flip(rho_inner2[1:], axis=0),
                            np.flip(rho_inner1, axis=0))

        r = np.append(r_inner, r_outer)
        T = np.append(T_inner, T_outer)
        rho = np.append(rho_inner, rho_outer)

        # Calculate the rest of the vars
        u, rho, phi, Lstar, L, P, cs, taus = calculateVars_rho(r, T, rho=rho, return_all=True)

        return Wind(rs, r, T, rho, u, phi, Lstar, L, P, cs, taus)




# # For testing when making modifications to this script

# x,z = IO.load_roots()

# All solutions
# verbose=0
# for logmdot,root in zip(x,z):
#    err1,err2=MakeWind(root,logmdot,Verbose=verbose)
#    print('%.2f \t\t %.3e \t-\t %.3e\n'%(logmdot,err1,err2))

# # # Just one solution
# # err1,err2=MakeWind(z[20],x[20], Verbose=True)
# # err1,err2=MakeWind([1.02,7.1],18.9)
# # print(err1,err2)
