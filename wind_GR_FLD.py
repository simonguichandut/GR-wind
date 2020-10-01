''' 
Main code to calculate winds 
Version with flux-limited diffusion : transitions to optically thin
'''

import sys
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import solve_ivp,quad
from collections import namedtuple
from scipy.interpolate import InterpolatedUnivariateSpline as IUS 
import IO
import physics

# ----------------------- Constants and parameters ---------------------------

# Constants
arad = 7.5657e-15
c = 2.99792458e10
sigmarad = 0.25*arad*c

# Parameters
params = IO.load_params()
if params['FLD'] == False: 
    sys.exit('This script is for FLD calculations')

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

# ------------------------- General Relativity -------------------------------

def gamma(v):
    return 1/np.sqrt(1-v**2/c**2)

def Swz(r):  # Schwartzchild metric term
    return (1-2*GM/c**2/r)

def Lcrit(r,rho,T): # local critical luminosity
    return LEdd * (eos.kappa0/eos.kappa(rho,T)) *Swz(r)**(-1/2)

# ----------------------- Flux-limited diffusion -----------------------------
# Modified version of Pomraning (1983) FLD prescription.  
# See Guichandut & Cumming (2020)

def FLD_Lam(Lstar,r,v,T):

    if isinstance(Lstar, (list,tuple,np.ndarray)): 
        # for function to be able to take and return array
        Lam = []
        for lstari,ri,vi,Ti in zip(Lstar,r,v,T):
            Lam.append(FLD_Lam(lstari,ri,vi,Ti))
        return np.array(Lam)

    else:
        L = Lcomoving(Lstar,r,v)
        Flux = L/(4*np.pi*r**2)
        alpha = Flux/(c*arad*T**4)  # 0 opt thick, 1 opt thin

        if alpha>1:
#            raise Exception
#            print('causality warning : F>cE')
            alpha=1-1e-9

        Lam = 1/12 * ( (2-3*alpha) + np.sqrt(-15*alpha**2 + 12*alpha + 4) )  
        # 1/3 thick , 0 thin

        ## Quinn formula
        # YY = Y(r,v)
        # rho = Mdot/(4*np.pi*r**2*v*YY)
        # Lam = 1/(3 + 2*YY/taustar(r,rho,T))

        return Lam

# -------------------------- Paczynski&Proczynski ----------------------------

def Y(r, v):  # eq 2a
    return np.sqrt(Swz(r))*gamma(v)

def Lcomoving(Lstar,r,v):
    return Lstar/(1+v**2/c**2)/Y(r, v)**2

def taustar(r,rho,T):
    return rho*eos.kappa(rho,T)*r

def Tstar(Lstar, T, r, rho, v):  # eq 2b
    return Lstar/LEdd * eos.kappa(rho,T)/eos.kappa0 * GM/(4*r) * \
            3*rho/(arad*T**4) * (1+(v/c)**2)**(-1) * Y(r, v)**(-3)

def A(T):  # eq 5a
    return 1 + 1.5*eos.cs2(T)/c**2

def B(T):
    return eos.cs2(T)

def C(Lstar, T, r, rho, v):  # eq 5c, but modified because of FLD

    Lam = FLD_Lam(Lstar,r,v,T)
    L = Lcomoving(Lstar,r,v)

    return 1/Y(r,v) * L/LEdd * eos.kappa(rho,T)/eos.kappa0 * GM/r * \
            (1 + eos.Beta(rho,T)/(12*Lam*(1-eos.Beta(rho,T))))


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

    Lam = FLD_Lam(Lstar,r,v,T)
    L = Lcomoving(Lstar,r,v)

    return 1/Y(r,v) * L/LEdd * eos.kappa(rho,T)/eos.kappa0 * GM/r * \
            (1 + (bi + alpha1*be)/(12*Lam*(1-bi-be)))

# ------------------------------- u(phi) ------------------------------------

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
            u = 0.5*phi*np.sqrt(B(T))*(1.0-np.sqrt(1.0-(2.0/phi)**2))/np.sqrt(A(T))
        else:
            u = 0.5*phi*np.sqrt(B(T))*(1.0+np.sqrt(1.0-(2.0/phi)**2))/np.sqrt(A(T))
    return u

# ----------------------------- Sonic point ---------------------------------

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

    # First check if sonic point would be below 2 x gravitational radius
    # NS radius should never be under 2rg anyway
    rg = 2*GM/c**2
    if numerator(2*rg, Ts, vs) > 0.0:
        raise Exception("Error: sonic point is below gravitational radius")
    
    # Check if it's below RNS
    if numerator(RNS*1e5, Ts, vs) > 0.0:
        if verbose: print("Warning : sonic point below RNS")

    while rkeep1 == 0 or rkeep2 == 0:
        logr = np.linspace(np.log10(2*rg), 9, npoints)
        for r in 10**logr:
            try:
                foo = numerator(r, Ts, vs)
            except Exception:
                print('Broke causality (F>cE) when trying sonic pt at r=%.3e'%r)
                pass
            else:
                if foo < 0.0:
                    rkeep1 = r
                if foo > 0.0 and rkeep2 == 0:
                    rkeep2 = r
        
        npoints += 10
    # print('sonic: rkeep1 = %.3e \t rkeep2 = %.3e'%(rkeep1,rkeep2))
    # global rs
    rs = brentq(numerator, rkeep1, rkeep2, args=(Ts, vs), maxiter=100000)
    return rs

# -------------------- Calculate vars and derivatives -------------------------

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
        lam = FLD_Lam(Lstar,r,u,T)
        return u, rho, phi, Lstar, L, P, cs, taus, lam


def calculateVars_rho(r, T, rho, return_all=False): 
    # Will consider degenerate electrons if EOS_type is set to 'IGDE'

    # At the end rho will be given as an array
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
        lam = FLD_Lam(Lstar,r,u,T)
        return u, rho, phi, Lstar, L, P, cs, taus, lam
   

def dr(r, y, subsonic):
    ''' Calculates the derivatives of T and phi by r '''

    T, phi = y[:2]
    u, rho, phi, Lstar = calculateVars_phi(r, T, phi=phi, subsonic=subsonic)

    Lam = FLD_Lam(Lstar,r,u,T)
    dlnT_dlnr = -Tstar(Lstar, T, r, rho, u) / (3*Lam) - 1/Swz(r) * GM/c**2/r  
    # remove small dv_dr term which has numerical problems near sonic point
    dT_dr = T/r * dlnT_dlnr

    mach = u/np.sqrt(B(T))
    dphi_dr = (A(T)*mach**2 - 1) *\
        ( 3*B(T) - 2*A(T)*c**2) / (4*mach*A(T)**(3/2)*c**2*r) * dlnT_dlnr \
        - numerator(r, T, u) / (u*r*np.sqrt(A(T)*B(T)) )
    
    return [dT_dr, dphi_dr]

def dr_wrapper_supersonic(r,y): return dr(r,y,subsonic=False)
def dr_wrapper_subsonic(r,y):   return dr(r,y,subsonic=True)

def drho(rho, y):
    ''' Calculates the derivatives of T and r with by rho
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

# ------------------------------ Integration ---------------------------------

def outerIntegration(r0,T0,phi0,rmax=1e10):
    ''' Integrates out from r0 to rmax tracking T and phi '''

    if verbose:
        print('**** Running outerIntegration ****')

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
    # triggers when going from negative to positive 
    # (numerator is negative in supersonic region)
    dv_dr_zero.terminal = True
        
    # Go
    inic = [T0,phi0]
    sol = solve_ivp(dr_wrapper_supersonic, (r0,rmax), inic, method='Radau', 
                    events=(dv_dr_zero,hit_mach1,hit_1e8), dense_output=True, 
                    atol=1e-6, rtol=1e-10, max_step=1e5)
    
    if verbose: 
        print('FLD outer integration : ',sol.message, ('rmax = %.3e'%sol.t[-1]))
    return sol

def innerIntegration_r():
    ''' Integrates in from the sonic point to 95% of the sonic point, 
        using r as the independent variable '''
    
    if verbose:
        print('**** Running innerIntegration R ****')

    inic = [Ts, 2.0]

    sol = solve_ivp(dr_wrapper_subsonic, (rs,0.95*rs), inic, method='RK45',
                    atol=1e-6, rtol=1e-6, dense_output=True)   

    if verbose: print(sol.message)

    return sol


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
        sol = solve_ivp(drho, (rho95,rhomax), inic, method='Radau',
                        events = (hit_Pinner,hit_zerospeed),  dense_output=True,
                        atol=1e-6, rtol=1e-6,)    
    except:
        if verbose: 
            print('Surface pressure never reached (NaNs before reaching p_inner)')
        return +200


    if verbose: print(sol.message)

    if sol.status == 1 :         # A termination event occured

        if len(sol.t_events[0]) == 1:  # The correct termination event 
            
            rbase = sol.y[1][-1]
            if verbose: print('Found base at r = %.2f km\n' % (rbase/1e5))

            if returnResult:
                return sol
            else:
                return (rbase/1e5 - RNS)/RNS    # Boundary error #2

        else:
            flag_u0 = 1
            p = hit_Pinner(sol.t[-1],sol.y[-1]) + P_inner
            col = p/g
            if verbose: print('Zero velocity before pressure condition reached.\
                                Last pressure : %.3e (y = %.3e)\n'%(p,col))
        
    else:
        if verbose: 
            print('Pressure condition nor zero velocity reached. \
                    Something else went wrong\n')

    if returnResult:
        return sol
    else:
        if flag_u0:
            return +100
        else:
            return +300


# -------------------------------- Wind --------------------------------------

# A named tuple allows us to access arrays by their variable name, 
# while also being able to tuple unpack to get everything
Wind = namedtuple('Wind',
            ['rs','r','T','rho','u','phi','Lstar','L','P','cs','taus','lam'])  

def setup_globals(params,logMdot,Verbose=False,return_them=False):
    global Mdot, Edot, Ts, rs, verbose
    Mdot, Edot, Ts, verbose = 10**logMdot, params[0]*LEdd, 10**params[1],Verbose
    rs = rSonic(Ts)
    if return_them:
        return Mdot, Edot, Ts, rs, verbose


def OuterBisection(rend=1e9,tol=1e-5):

    """ Makes a full outer solution for the wind by integrating until 
    divergence, interpolating values by bissection and restarting prior to 
    divergence point, over and over until reaching rend."""

    # get the solution from the root's Ts (rs and Ts already set as global)
    if verbose: print('Calculating solution from Ts root')
    rsa,Tsa = rs,Ts
    sola = outerIntegration(r0=rsa,T0=Tsa,phi0=2.0)

    # find other solution that diverges in different direction
    if sola.status == 0: 
        sys.exit('reached end of integration interval with root!')

    elif sola.status == +1:
        direction = +1  # reached dv/dr=0,other solution needs to have higher Ts
    elif sola.status == -1:
        direction = -1 # diverged, other solution needs to have smaller Ts

    if verbose: print('Finding second solution')
    step = 1e-6
    Tsb,rsb,solb = Tsa,rsa,sola
    i=0
    while solb.status == sola.status:

        if i>0: 
            Tsa,rsa,sola = Tsb,rsb,solb  
            # might as well update solution a 
            # (since this process gets closer to the right Ts)

        logTsb = np.log10(Tsb) + direction*step
        Tsb = 10**logTsb
        rsb = rSonic(Tsb)
        solb = outerIntegration(r0=rsb,T0=Tsb,phi0=2.0)
        i+=1
        if i==20:
            print('Not able to find a solution that diverges in opposite \
                    direction after changing Ts by 20 tolerances.  \
                    Problem in the TsEdot interpolation')
            raise Exception('Improve root')
            # break

    # if sola was the high Ts one, switch sola and solb
    if direction == -1:
        (rsa,Tsa,sola),(rsb,Tsb,solb) = (rsb,Tsb,solb),(rsa,Tsa,sola)
            
    if verbose:
        print('Two initial solutions. sonic point values:')
        print('logTs - sola:%.6f \t solb:%.6f'%(np.log10(Tsa),np.log10(Tsb)))
        print('logrs - sola:%.6f \t solb:%.6f'%(np.log10(rsa),np.log10(rsb)))


    def check_convergence(sola,solb,rcheck):
        """ checks if two solutions are converged (similar T, phi) at some r """
        Ta,phia = sola.sol(rcheck)
        Tb,phib = solb.sol(rcheck)
        if abs(Ta-Tb)/Ta < tol and abs(phia-phib)/phia < tol:
            return True,Ta,Tb,phia,phib
        else:
            return False,Ta,Tb,phia,phib


    # Start by finding the first point of divergence
    Npts = 1000
    R = np.logspace(np.log10(rsa),np.log10(rend),Npts)   
    # choose colder (larger) rs (rsa) as starting point because 
    # sola(rsb) doesnt exist

    for i,ri in enumerate(R):
        conv = check_convergence(sola,solb,ri)
        if conv[0] is False:
            i0=i            # i0 is index of first point of divergence
            break
        else:
            Ta,Tb,phia,phib = conv[1:]

    if i0==0:
        print('Diverging at rs!')
        print(conv)
        print('rs=%.5e \t rsa=%.5e \t rsb=%.5e'%(rs,rsa,rsb))
        

    # Construct initial arrays
    T,Phi = sola.sol(R[:i0])
    def update_arrays(T,Phi,sol,R,j0,jf):
        # Add new values from T and Phi using ODE solution object. 
        # Radius points to add are R[j0] and R[jf]
        Tnew,Phinew = sol(R[j0:jf+1])  # +1 so R[jf] is included
        T,Phi = np.concatenate((T,Tnew)), np.concatenate((Phi,Phinew))
        return T,Phi

    # Begin bisection
    if verbose:
        print('\nBeginning bisection')
        print('rconv (km) \t Step # \t Iter \t m')  
    a,b = 0,1
    step,count = 0,0
    i = i0
    rconv = R[i-1]  # converged at that radius
    rcheck = R[i]   # checking if converged at that radius
    do_bisect = True
    while rconv<rend:  
        # probably can be replaced by while True if the break conditions are ok

        if do_bisect: # Calculate a new solution from interpolated values
            
            m = (a+b)/2
            Tm,phim = Ta + m*(Tb-Ta) , phia + m*(phib-phia)
            solm = outerIntegration(r0=rconv,T0=Tm,phi0=phim) 
            # go further than rmax to give it the chance to diverge either way

            if solm.status == 0: # Reached rend - done
                T,Phi = update_arrays(T,Phi,solm.sol,R,i0,Npts)  
                #jf=Npts so that it includes the last point of R
                print('reached end of integration interval  without \
                    necessarily converging.. perhaps wrong')
                return R,T,Phi

            elif solm.status == 1:
                a,sola = m,solm
            elif solm.status == -1:
                b,solb = m,solm

        else:
            i += 1
            rconv = R[i-1]
            rcheck = R[i] 

        conv = check_convergence(sola,solb,rcheck)
        if conv[0] is True:

            # Exit here if reached rend
            if rcheck==rend or i==Npts:  # both should be the same
                T,Phi = update_arrays(T,Phi,solm.sol,R,i0,i)
                return R,T,Phi

            Ta,Tb,phia,phib = conv[1:]
            a,b = 0,1 # reset bissection parameters
            step += 1 # update step counter
            count = 0 # reset iteration counter

            # Converged, so on next iteration just look further
            do_bisect = False 
        
        else:
            count+=1
            do_bisect = True

            # Before computing new solution, add converged results to array 
            # (but only if we made at least 1 step progress)
            if i-1>i0:
                T,Phi = update_arrays(T,Phi,solm.sol,R,i0,i-1)  # i-1 is where we converged last
                i0=i # next time we append

        # Exit if stuck at one step
        nitermax=1000
        if count==nitermax:
            sys.exit("Could not integrate out to rend! Exiting after being \
                        stuck at the same step for %d iterations"%nitermax)

        # End of step
        if verbose: print('%.4e \t %d \t\t %d \t\t %f'%(rconv,step,count,m))

    return R,T,Phi


def MakeWind(params, logMdot, mode='rootsolve', Verbose=0, IgnoreErrors = False):
    ''' Obtaining the wind solution for set of parameters Edot/LEdd and logTs'''

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
        sys.exit("Rootsolving is done in RootFinding_FLD.py")

    elif mode == 'wind': 

        # Outer integration
        r_outer,T_outer, phi_outer = OuterBisection()
        _,rho_outer,_,_ = calculateVars_phi(r_outer, T_outer, phi=phi_outer, 
                            subsonic=False)

        # Sonic point changes slightly in the bisection process
        Tsnew,rsnew = T_outer[0], r_outer[0]
        print('Change in sonic point (caused by error in Edot-Ts relation interpolation)')
        print('root:\t Ts = %.5e \t rs = %.5e'%(Ts,rs))
        print('new:\t  Ts = %.5e \t rs = %.5e'%(Tsnew,rsnew))
        print('Judge if this is a problem or not')
        rs = rsnew

        # First inner integration
        r95 = 0.95*rs
        # r_inner1 = np.linspace(rs, r95, 500)
        r_inner1 = np.linspace(0.99*rs, r95, 30) # ignore data in 1% around rs
        result_inner1 = innerIntegration_r()
        T95, _ = result_inner1.sol(r95)
        T_inner1, phi_inner1 = result_inner1.sol(r_inner1)

        _,rho_inner1,_,_ = calculateVars_phi(r_inner1, T_inner1, phi=phi_inner1, 
                                subsonic=True)
        rho95 = rho_inner1[-1]

        # Second inner integration 
        result_inner2 = innerIntegration_rho(rho95, T95, returnResult=True)
        rho_inner2 = np.logspace(np.log10(rho95) , np.log10(result_inner2.t[-1]), 2000)
        T_inner2, r_inner2 = result_inner2.sol(rho_inner2)
        

        # Attaching arrays for r,rho,T from surface to photosphere  
        #  (ignoring first point in inner2 because duplicate values at r=r95)
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
        u, rho, phi, Lstar, L, P, cs, taus, lam = calculateVars_rho(r, T, rho=rho, return_all=True)

        return Wind(rs, r, T, rho, u, phi, Lstar, L, P, cs, taus ,lam)





# # For testing when making modifications to this script

# x,z = IO.load_roots()
# W = MakeWind(z[3],x[3], mode='wind', Verbose=True)