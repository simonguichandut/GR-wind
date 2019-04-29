from scipy.optimize import brentq
from scipy.integrate import odeint
import numpy as np
from numpy import linspace,sqrt,log10,array
from math import pi

# --------------------------------------- Constants and parameters --------------------------------------------

tauOuter = 3.0 # Flexible outer photosphere optical depth, 3<tau<5
eps = 1e-6

mu=4.0/3.0 # pure He

kB=1.380658e-16
arad=7.5657e-15
c=2.99792458e10
mp=1.67e-24
kappa0=0.2 
sigmarad=0.25*arad*c
GM=6.6726e-8*2e33*1.4
LEdd=4*pi*c*GM/kappa0

# -------------------------------------------- Microphysics ----------------------------------------------------

def kappa(T):
	return kappa0/(1.0+(2.2e-9*T)**0.86) 
#    return kappa0/(1+(2.187e-9*T)**0.976) # From Juri Poutanen's paper

def cs2(T): # ideal gas sound speed  c_s^2
	return kB*T/(mu*mp)
	
def pressure(rho,T): # ideal gas + radiation pressure (eq 2c)
	return rho*cs2(T) + arad*T**4/3.0

def internal_energy(rho,T): # ideal gas + radiation
    return 1.5*cs2(T)*rho + arad*T**4
		
def	Beta(rho,T):  # pressure ratio
	Pg = rho*cs2(T)
	Pr = arad*T**4/3.0
	return Pg/(Pg+Pr)

# ----------------------------------------- General Relativity ------------------------------------------------

def gamma(v):
    return 1/sqrt(1-v**2/c**2)

def Swz(r): # Schwartzchild metric term
    return (1-2*GM/c**2/r)

# ----------------------------------------- Paczynski&Proczynski ----------------------------------------------
    
def Y(r,v): # eq 2a
    return sqrt(Swz(r))*gamma(v)

def Tstar(Lstar,T,r,rho,v): # eq 2b
    return Lstar/LEdd * kappa(T)/kappa0 * GM/(4*r) * 3*rho/(arad*T**4) * (1+(v/c)**2)**(-1) * Y(r,v)**(-3)

def H(rho,T): # eq 2c
    return c**2 + (internal_energy(rho,T)+pressure(rho,T))/rho

def A(T): # eq 5a
    return 1 + 1.5*cs2(T)/c**2
    
def C(Lstar,T,r,rho,v): # eq 5c
    return Tstar(Lstar,T,r,rho,v) * (4-3*Beta(rho,T))/(1-Beta(rho,T)) * arad*T**4/(3*rho)
    
# -------------------------------------------- u(phi) ------------------------------------------------------

def uphi(phi,T,inwards):

	''' phi should never drop below 2, but numerically
	it is sometimes just < 2, so in that case return Mach number 1 (divided by sqrt(A))
    This is using the GR version of the Joss & Melia change of variable phi in sonic units,
    where the difference between the velocity and sound speed at the critical point
    (vs=cs/sqrt(A)=0.999cs) is taken into account : phi = sqrt(A)*mach + 1/sqrt(A)/mach '''
    
	if phi < 2.0:
            u = 1.0*sqrt(cs2(T))/sqrt(A(T))
	else:
            if inwards:
                u = 0.5*phi*sqrt(cs2(T))*(1.0-sqrt(1.0-(2.0/phi)**2))/sqrt(A(T))
            else:
                u = 0.5*phi*sqrt(cs2(T))*(1.0+sqrt(1.0-(2.0/phi)**2))/sqrt(A(T))
	return u

# -------------------------------------------- Sonic point -------------------------------------------------

def numerator(r,T,v,verbose=False): # numerator of eq (4a) 
    
    rho = Mdot/(4*pi*r**2*Y(r,v)*v)     # eq 1a
    Lstar = Edot-Mdot*H(rho,T)*Y(r,v) + Mdot*c**2   # eq 1c
    
    return gamma(v)**(-2) * ( GM/r/Swz(r) * (A(T)-cs2(T)/c**2) - C(Lstar,T,r,rho,v) - 2*cs2(T) )   

def rSonic(Ts):
    
    ''' Finds the sonic point radius corresponding to the sonic point temperature Ts '''
    
    rkeep1,rkeep2 = 0.0,0.0
    npoints=10
    vs = sqrt(cs2(Ts)/sqrt(A(Ts)))
    while rkeep1 == 0 or rkeep2 == 0:
        logr = linspace(6,9,npoints)
        for r in 10**logr:
            foo = numerator(r,Ts,vs)
            if foo < 0.0 : rkeep1 = r
            if foo > 0.0 and rkeep2 == 0 : rkeep2 = r
        npoints += 10
        
    rs=brentq(numerator,rkeep1,rkeep2,args=(Ts,vs),xtol=1e-10,maxiter=10000)
    return rs

# ---------------------------------------- Calculate vars and derivatives -----------------------------------

def calculateVars(r,T,phi=0,rho=-1,inwards=False,return_all=False):
    
    # At the end rho will be given as an array, so we need these lines to allow this
    if isinstance(rho,(list,tuple,np.ndarray)): 
        rho_given = 1
        r,T,rho = array(r),array(T),array(rho) # to allow mathematical operations
    else:
        if rho<0: 
            rho_given = 0
        else:
            rho_given = 1
    
    if not rho_given: # if density is not given, phi is
        if isinstance(phi,(list,tuple,np.ndarray)):
            u=[]
            for i in range(len(phi)):
                u.append(uphi(phi[i],T[i],inwards))
            r,T,u = array(r),array(T),array(u)
        else:
            u = uphi(phi,T,inwards)
            
        rho = Mdot/(4*pi*r**2*u*Y(r,u))
    else:      # if density is given, phi is not
        u = Mdot/sqrt((4*pi*r**2*rho)**2*Swz(r) + (Mdot/c)**2)
        mach = u/sqrt(cs2(T))
        phi = sqrt(A(T))*mach + 1/(sqrt(A(T))*mach)
        
    Lstar = Edot-Mdot*H(rho,T)*Y(r,u) + Mdot*c**2
    if not return_all:        
        return u,rho,phi,Lstar
    else:
        L = Lstar/(1+u**2/c**2)/Y(r,u)**2
        LEdd_loc = LEdd*(kappa0/kappa(T))*Swz(r)**(-1/2)
        E = internal_energy(rho,T)
        P = pressure(rho,T)
        cs = sqrt(cs2(T))
        tau = rho*kappa(T)*r
        return u,rho,phi,Lstar,L,LEdd_loc,E,P,cs,tau


def dr(inic,r,inwards):
    
    ''' Calculates the derivatives of T and phi with r as the independnt variable '''

    T,phi = inic[:2]
    u,rho,phi,Lstar = calculateVars(r,T,phi=phi,inwards=inwards)
    mach = u/sqrt(cs2(T))
    
    if r==rs or phi<2:
        dlnT_dlnr = -1  # avoid divergence at sonic point
    else:               # (eq. 4a&b)
        dlnv_dlnr = numerator(r,T,u) / (cs2(T) - u**2*A(T)) 
        dlnT_dlnr = -Tstar(Lstar,T,r,rho,u) - 1/Swz(r)*GM/c**2/r - gamma(u)**2*(u/c)**2*dlnv_dlnr
        
    cs,rootA = sqrt(cs2(T)) , sqrt(A(T))
    mach = u/cs
    
    dT_dr = T/r * dlnT_dlnr
    dphi_dr = dT_dr * ( 3/(4*rootA*T) * (cs/c)**2 *(mach-1/mach/A(T)) + cs/2/T *(-rootA*mach/cs + 1/rootA/u )) -\
              1/(rootA*cs*r*u) * numerator(r,T,u)
        
    return [dT_dr, dphi_dr]


def drho(inic,rho): 
    
    ''' Calculates the derivatives of T and r with rho as the independnt variable '''
    
    T,r = inic[:2]
    u,rho,phi,Lstar = calculateVars(r,T,rho=rho)
    
    dT_dr,_ = dr([T,phi],r,inwards=True)
    dlnr_dlnrho = (cs2(T)-A(r)*u**2) / ((2*u**2 - (GM/(r*Y(r,u)**2)))*A(r) + C(Lstar,T,r,rho,u)) # eq 6
    
    dr_drho = r/rho * dlnr_dlnrho
    dT_drho = dT_dr * dr_drho

    return [dT_drho, dr_drho]


# -------------------------------------------- Integration ---------------------------------------------
  
def outerIntegration(Ts,returnResult=False):
    
    ''' Integrates out from the sonic point until the photosphere is reached '''
    
    if verbose: print('\n**** Running outerIntegration ****')
    
    inic = [Ts, 2.0]
    r_outer = 50*rs
    r = np.linspace(rs, r_outer, 5000)
    result = odeint(dr, inic, r, args=(False,), atol=eps, rtol=eps) # contains T(r) and phi(r)
    
    # odeint does not support stop conditions (should switch to scipy.integrate.ode, dopri5 integrator, with solout option)
    # For now, just cut the solution where the first NaN appears
    if True in np.isnan(result[:,0]):
        firstnan = min([np.argwhere(np.isnan(result[:,i])==True)[0][0] for i in (0,1) if True in np.isnan(result[:,i])]) # It works!
        result = result[:firstnan]

    flag = 1
    L1,L2,tau = 0,0,[]
    
    for counter, phi in enumerate(result[:,1]):
        
        rr = r[counter]
        T = result[counter, 0]
        
        if T<1e3:
            flag = 2
            if verbose:
                print('It has gotten too cold, exiting before NaNs appear')
                print('Last three values of T : %.1e -> %.1e -> %.1e'%(result[counter-2,0],result[counter-1,0],T))
            break
        
        u,rho,phi,Lstar = calculateVars(rr,T,phi=phi,inwards=False)
        L = Lstar/(1+(u/c)**2)/Y(rr,u)**2 # eq. 3 (comoving luminosity)
        tau.append(rho*kappa(T)*rr)
        
        if flag and tau[-1] <= tauOuter:
            flag = 0
            L1,L2 = L , 4.0*pi*rr**2*sigmarad*T**4
            photosphere = counter
            break
        
        if counter>2 and tau[-1]>tau[-2]:  # if tau increases, the solution is no longer physical
            flag = 2
            break

    if flag != 0: 
        if verbose:print("tau = ", tauOuter, " never reached! Minimum tau reached :", tau[-2])
        if flag == 2:
            if verbose:print("Phi started to increase at logr = %.2f"%log10(rr))
            
    else: 
        if verbose:print('Found photosphere at log10 r = %.2f'%log10(rr))
        
    if returnResult:
        if flag == 0:
            return r[:photosphere+1],result[:photosphere+1 , :]
        else:
            return r[:counter+1],result[:counter+1 , :]
    
    else:
        if L1 == 0 or L2 == 0:
            if flag == 2:     # we might want to decide what to do based on what happened to phi
                return +100
            else:
                return +200
        else:
            return (L2 - L1) / (L1 + L2)      # Boundary error #1
    
    
def innerIntegration_phi(r,Ts):
    
    ''' Integrates in from the sonic point to 0.95rs, using phi as the independent variable '''
    if verbose: print('\n**** Running innerIntegration PHI ****')
    inic = [Ts, 2.0]
    result = odeint(dr, inic, r, args=(True,), atol=eps, rtol=eps) # contains T(r) and phi(r)
    return result

def innerIntegration_rho(rho,r95,T95,returnResult=False):
    
    ''' Integrates in from 0.95rs, using rho as the independent variable, until rho=10^4 being the surface '''
    if verbose: print('\n**** Running innerIntegration RHO ****')
    
    inic = [T95,r95]
    result = odeint(drho, inic, rho, atol=eps, rtol=eps) # contains T(rho) and r(rho)
    flag = 0
    
    if True in np.isnan(result):
        flag = 1
        firstnan = min([np.argwhere(np.isnan(result[:,i])==True)[0][0] for i in (0,1) if True in np.isnan(result[:,i])]) 
        result = result[:firstnan]
        if verbose : print('Surface density never reached!')
    
    if returnResult:
        return result
    else:
        if flag:
            return +100
        else:
            RNS = result[-1,1]
            return (RNS-1e6)/1e6       # Boundary error #2
    
# ------------------------------------------------- Wind ---------------------------------------------------

def MakeWind(params,logMdot,mode='rootsolve',Verbose=0):
    
    ''' Obtaining the wind solution for set of parameters Edot/LEdd and log10(Ts).
        The modes are rootsolve : not output, just obtaining the boundary errors, 
        and wind : obtain the full solutions.   '''
    
    global Mdot,Edot,rs,verbose
    Mdot,Edot,Ts,verbose = 10**logMdot , params[0]*LEdd , 10**params[1] , Verbose
    
    # Start by finding the sonic point
    rs = rSonic(Ts)
    if verbose: print('\nFor log10Ts = %.2f, located sonic point at log10r = %.2f'%(log10(Ts),log10(rs)))
    
    if mode == 'rootsolve':
                
        # First error is given by the outer luminosity
        error1 = outerIntegration(Ts)
        
        if error1==100:        # don't bother integrating inwards
            return [100,100]
        
        else:
        
            # First inner integration
            r95 = 0.95*rs
            r_inner1 = np.linspace(rs,r95,1000)
            result_inner1 = innerIntegration_phi(r_inner1,Ts)
            T95,phi95 = result_inner1[-1,:]
            _,rho95,_,_ = calculateVars(r95,T95,phi=phi95,inwards=True)
            
            # Second inner integration
            rho_inner2 = np.logspace(log10(rho95),4,2000)
            error2 = innerIntegration_rho(rho_inner2,r95,T95)
            
            return error1,error2
    
    elif mode == 'wind': # Same thing but calculate variables and output all of the arrays
        
        # Outer integration
        r_outer,result_outer = outerIntegration(Ts,returnResult=True)
        T_outer,phi_outer = result_outer[:,0] , result_outer[:,1]   
        _,rho_outer,_,_ = calculateVars(r_outer,T_outer,phi=phi_outer,inwards=False)                              
        
        # First inner integration
        r95 = 0.95*rs
        r_inner1 = np.linspace(rs,r95,1000)
        result_inner1 = innerIntegration_phi(r_inner1,Ts)
        T_inner1,phi_inner1 = result_inner1[:,0] , result_inner1[:,1]
        T95,phi95 = result_inner1[-1,:]
        _,rho_inner1,_,_ = calculateVars(r_inner1,T_inner1,phi=phi_inner1,inwards=True)
        rho95 = rho_inner1[-1]
        
        # Second inner integration
        rho_inner2 = np.logspace(log10(rho95),4,2000)
        result_inner2 = innerIntegration_rho(rho_inner2,r95,T95,returnResult=True)
        T_inner2, r_inner2 = result_inner2[:,0] , result_inner2[:,1]
        
        # Attaching arrays for r,rho,T from surface to photosphere
        r_inner = np.append(np.flip(r_inner2,axis=0),np.flip(r_inner1,axis=0))
        T_inner = np.append(np.flip(T_inner2,axis=0),np.flip(T_inner1,axis=0))
        rho_inner = np.append(np.flip(rho_inner2,axis=0),np.flip(rho_inner1,axis=0))
        
        R   = np.append(r_inner,r_outer)
        T   = np.append(T_inner,T_outer)
        Rho = np.append(rho_inner,rho_outer)
        
        # Calculate the rest of the vars
        u,Rho,Phi,Lstar,L,LEdd_loc,E,P,cs,tau = calculateVars(R,T,rho=Rho,return_all=True)
        
        return R,T,Rho,u,Phi,Lstar,L,LEdd_loc,E,P,cs,tau
        