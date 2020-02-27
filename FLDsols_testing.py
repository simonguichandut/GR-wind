'''
Finding solutions to FLD winds 

Method #2

For every Mdot, choose (Edot,Ts,vinf), integrate out from sonic point and in from infinity.  Only keep parameter sets for which the solutions 
cross in T and v.  Error is crossing radius for T - crossing radius for v.

This is a test file
04/02/2020
'''


import wind_GR as W
from wind_GR import Y,gamma,Swz,Lcomoving,Lcrit,taustar,FLD_Lam,A,B,C,Tstar
import IO
import physics
import numpy as np
from numpy import linspace, logspace, sqrt, log10, array, pi, gradient
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from scipy.interpolate import InterpolatedUnivariateSpline as IUS  
from scipy.optimize import brentq
import sys

arad = 7.5657e-15
c = 2.99792458e10
sigmarad = 0.25*arad*c
M, RNS, y_inner, tau_out, comp, EOS_type, FLD, mode, save, img = IO.load_params()
eos = physics.EOS(comp)
GM = 6.6726e-8*2e33*M
LEdd = 4*pi*c*GM/eos.kappa0




#%% Second solution setup

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
    
#    print('logr = %.3f \t dlnT/dlnr = %.3f \t dlnv/dlnr = %.3f'%(log10(r),dlnT_dlnr,dlnv_dlnr))
    
    return [dT_dr, dv_dr]


def innerIntegration_inf():
    
    if verbose:
        print('**** Running innerIntegration from inf ****')
    
    gammainf = gamma(vinf)
    Linf = Edot + Mdot*c**2*(1-gammainf)
    r0 = 1e12
   
    # Need to rootfind for the values of v0 and Lstar0
    
    def calc_initial_vars(x):
        v0,Lstar0 = x
        L0 = Lcomoving(Lstar0,r0,v0)
        rho0 = Mdot/(4*pi*r0**2*v0*Y(r0,v0))
        T0 = (L0/(4*pi*r0**2*arad*c))**(0.25)
        return v0,Lstar0,L0,rho0,T0
    
    def error(x):
        
        v0,Lstar0,L0,rho0,T0 = calc_initial_vars(x)
        
        E1 = Y(r0,v0) - gammainf + eos.kappa(rho0,T0)*L0/(4*pi*r0*c**3*Y(r0,v0))   # Integrated momentum equation ignoring Pg'
        E2 = (Edot + Mdot*c**2 - Lstar0 - Mdot*eos.H(rho0,T0)*Y(r0,v0))/LEdd
        return [E1,E2]
    
    x = fsolve(error,x0=[vinf,Linf])
    v0,Lstar0,L0,rho0,T0 = calc_initial_vars(x)
    
    
    
    # setup solver
    def causality(r,y):
        T, v = y[:2]
        rho, L, _ = calculateVars_v(r, T, v=v)
        flux = L/(4*pi*r**2)
        Er = arad*T**4
        return flux-c*Er
#    causality.terminal = True
    
#    def divergence(r,y):
#        dT_dr,dv_dr = dr_fld(r,y)
#        return np.sign(dT_dr*dv_dr)
#    divergence.direction = +1  # if both gradients have the same sign, the integration will diverge
##    divergence.terminal = True
    
    def divergence(r,y):
        return y[1]-vinf
#    divergence.direction = -1  # velocity gradient becomes negative (v goes up as we go in, we're diverging for sure)
    divergence.terminal = True
    
    def hit_mach1(r,y): 
        cs = sqrt(eos.cs2(y[0]))
        return y[1]-cs
    hit_mach1.terminal = True # stop integrating at this point
       
    
   
    rspan = (r0,2*rs)
    
    # use normal dr which has fld

    result = solve_ivp(dr_fld, rspan, (T0,v0), method='RK45', dense_output=True, events=(causality,divergence,hit_mach1), rtol=1e-6)    

    if verbose:
        if result.status==1:
            if len(result.t_events[1])==1:
                print('FLD inner integration : Velocity diverged')
            elif len(result.t_events[2])==1:
                print('FLD inner integration : reached mach 1')
        else:
            print('FLD inner integration : ',result.message)
        
    return result

    
#    try:
#        result = solve_ivp(dr_fld, rspan, (T0,v0), method='RK45', dense_output=True, events=(causality), rtol=1e-6)    
#    
#        if verbose:
#            print(result.message)
#            
#        return result
#    
#    except:
#        pass
##        print('didnt work')
  
    

#%% Evaluate crossings

from scipy.optimize import fminbound

def FindCrossing(x1,y1,x2,y2): # Finds if there is a crossing between two lines and if so returns an interpolated location
    
    # need increasing values for spline fit
    if x1[-1]<x1[0]: x1,y1 = np.flip(x1[:]),np.flip(y1[:])
    if x2[-1]<x2[0]: x2,y2 = np.flip(x2[:]),np.flip(y2[:])
    
    
    # Ensure there is overlap between the two domains
    if x2[0]>x1[0]:
        if x1[-1]<x2[0]:
#            print ('no overlap')
            return (False,)
    if x1[0]>x2[0]:
        if x2[-1]<x1[0]:
#            print ('no overlap')
            return (False,)
    
    
    
    x = [max((x1[0],x2[0])),min((x1[-1],x2[-1]))] # new domain containing region of overlap
    
    
    f1,f2 = IUS(x1,y1),IUS(x2,y2) # spline fits

    if (f2(x[0])-f1(x[0]))*(f2(x[-1])-f1(x[-1])) > 0:
        diff = lambda xi : abs(f2(xi)-f1(xi))
#        print('no crossing')
    
        # if no crossing, take "crossing" as where the curves get closest to each other
        minimum = fminbound(diff,x[0],x[1],full_output=True,disp=False)
#        print(minimum)
    
        return (False,minimum)


    else:
        diff = lambda xi : f2(xi)-f1(xi)
        xcross = brentq(diff,x[0],x[-1])
        return (True,xcross)
        
    
#rcross_T = FindCrossing(wind1.r,wind1.T,np.flip(wind2.r),np.flip(wind2.T))
#rcross_u = FindCrossing(wind1.r,wind1.u,np.flip(wind2.r),np.flip(wind2.u))
    
#
#fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,7))
#ax1.loglog(wind1.r,wind1.T)
#ax1.loglog(wind2.r,wind2.T)
#ax2.loglog(wind1.r,wind1.u)
#ax2.loglog(wind2.r,wind2.u)
#ax1.axvline(rcross_T,color='k',lw=0.7)
#ax2.axvline(rcross_u,color='k',lw=0.7)
#plt.show()
#    
#crossing_error = rcross_T-rcross_u



def MakeWindFLD(params, logMdot, Verbose=0): # this would go into the full MakeWind later
    
    global Mdot,Edot,Ts,verbose,rs,vinf
    Mdot, Edot, Ts, verbose = W.setup_globals(params[:2],logMdot,Verbose,return_them=True)
    vinf = 10**params[2]


    # First solution : use old outerIntegration
    rs = W.rSonic(Ts)
    res1 = W.outerIntegration(returnResult=True)
    r1 = res1.t
    T1,phi1 = res1.sol(r1)
    u1, _,_,_ = W.calculateVars_phi(r1,T1, phi1, subsonic=False)
    
    result2= innerIntegration_inf()
    if len(result2.t_events[0])>0: # broke causality
#        return +500
        if verbose: print('F>cE at : ',result2.t_events[0])
    
    r2,(T2,u2)=result2.t,result2.y

    Tcross = FindCrossing(r1,T1,r2,T2)
    ucross = FindCrossing(r1,u1,r2,u2)
    
    if (not Tcross[0]) or (not ucross[0]):  # no crossing
        return +1000
    else:
        return ucross[1] - Tcross[1]



# Parameters
logMdot = 18.5
params = [1.025,7.2,8.42025]           # Edot/LEdd, logTs, logvinf

#params = [1.025,7.2,8.4]
#params = [1.025,6.9,8]

#E = MakeWindFLD(params, logMdot, Verbose=1)
    

#MakeWindFLD([1.025,7.2,8.42], logMdot, Verbose=1)

#%% Errorspace
import pickle
def get_map(logMDOT,Edot):

#    n=10
#    Tsvals = np.linspace(7.1,7.3,n)
#    vinfvals = np.linspace(8.2,8.7,n)
    
#    n=10
#    Tsvals = np.linspace(7.15,7.25,n)
#    vinfvals = np.linspace(8.3,8.5,n)
    
    n=10
    Tsvals = np.linspace(7.16,7.22,n)
    vinfvals = np.linspace(8.4,8.45,n)
    
    Errors = [[] for i in range(n)] 
    for i,Ts in zip(np.arange(n),Tsvals):
        print('\n\n LOG TS = %.3f \n\n'%Ts)
        for vinf in vinfvals:
            try:
                z = MakeWindFLD([Edot,Ts,vinf],logMDOT,Verbose=0)
                
#                if z!=1000: z /= rs
                
                print('log(vinf) = %.3f : %.3e'%(vinf,z))
            except:
#                pass
                z=1000
#                 print("\nFatal error in integration, skipping...\n")
            

            Errors[i].append(z)
            

    filename = 'analysis/errorspaces_FLD/save'+str(logMDOT)+'_'+str(Edot)+'.p'
    with open(filename,'wb') as f:
        pickle.dump([Tsvals,vinfvals,Errors],f)
        
    return (Tsvals,vinfvals,Errors)
    
#
#    print('\n\nFINISHED \nMap file saved to ',filename)    
    

import matplotlib.colors as colors

def plot_map(logMDOT,Edot):

    filename = 'analysis/errorspaces_FLD/save'+str(logMDOT)+'_'+str(Edot/LEdd)+'.p'
    [Tsvals,vinfvals,E]=pickle.load(open(filename,'rb'))

    fig,ax = plt.subplots(1,1)    
    fig.subplots_adjust(wspace=0.25)
    ax.patch.set_color('.25')      
    cmap = plt.get_cmap('Blues') 
    
    for i in range(len(E)):
        for j in range(len(E[0])):
            if E[i][j]==1000:
                E[i][j]=1           # ignore these values (1 will be out of range)
    levs=np.linspace(5,9,10)
    im = ax.contourf(vinfvals,Tsvals,np.log10(np.abs(E)),levs,cmap=cmap)
    
#    levs=np.linspace(0,20,20)
#    im = ax.contourf(vinfvals,Tsvals,E,levs,cmap=cmap)    
    
    fig.colorbar(im,ax=ax)
    ax.set_title(r'Error in rmatch')

    ax.set_xlabel(r'log $v_\infty$')
    ax.set_ylabel(r'log $T_s$ (K)')

    
    return Tsvals,vinfvals,E,fig,ax
    

        
#map1 = get_map(18.5,1.025)
#Tsvals,vinfvals,E,fig,ax = plot_map(18.5,1.025*LEdd)

   
            
#%% Manual 
            
#            
#
#logMdot = 18.5
#
##params = [1.025,7.2,8.42]             
##params =  [1.025,7,8]
#
##params = [1.025,7.12,8.405] # a good one : log10(E)=6.737
#
##params = [1.025,7.1595,8.4137]    # best one so far : log10(E)=6.3072
#
##params = [1.025,7.1595,8.414]   # this one is interesting to look at and maybe useful in future! shows the upper limit case of vinf i think
#
#
##params = [1.025,7.1595,8.41374983]
#
#params = [1.025,7.175,8.41374983]    # wow!! that is the one
#
##params = [1.025,7.175,8.413]  
#


#params = [1.025,7.175,8.38]  
#
#
    
#global Mdot,Edot,Ts,verbose,rs
#Mdot, Edot, Ts, verbose = W.setup_globals(params[:2],logMdot,Verbose=1,return_them=True)
##vinf = 10**params[2]
##
##
#rs = W.rSonic(Ts)
#res1 = W.outerIntegration(returnResult=True)
#r1 = res1.t
#T1,phi1 = res1.sol(r1)
#u1, _,_,_ = W.calculateVars_phi(r1,T1, phi1, subsonic=False)
#
#

#
#
#gammainf = gamma(vinf)
#Linf = Edot + Mdot*c**2*(1-gammainf)
#r0 = 1e12
#   
## Need to rootfind for the values of v0 and Lstar0
#
#def calc_initial_vars(x):
#    v0,Lstar0 = x
#    L0 = Lcomoving(Lstar0,r0,v0)
#    rho0 = Mdot/(4*pi*r0**2*v0*Y(r0,v0))
#    T0 = (L0/(4*pi*r0**2*arad*c))**(0.25)
#    return v0,Lstar0,L0,rho0,T0
#
#def error(x):
#    
#    v0,Lstar0,L0,rho0,T0 = calc_initial_vars(x)
#    
#    E1 = Y(r0,v0) - gammainf + eos.kappa(rho0,T0)*L0/(4*pi*r0*c**3*Y(r0,v0))   # Integrated momentum equation ignoring Pg'
#    E2 = (Edot + Mdot*c**2 - Lstar0 - Mdot*eos.H(rho0,T0)*Y(r0,v0))/LEdd
#    return [E1,E2]
#
#x = fsolve(error,x0=[vinf,Linf])
#v0,Lstar0,L0,rho0,T0 = calc_initial_vars(x)
#
#
#def causality(r,y):
#    T, v = y[:2]
#    rho, L, _ = calculateVars_v(r, T, v=v)
#    flux = L/(4*pi*r**2)
#    Er = arad*T**4
#    return flux-c*Er
##causality.terminal = True
#
#    
##def divergence(r,y):
##    _,dv_dr = dr_fld(r,y)
##    return dv_dr
##divergence.direction = -1  # if both gradients have the same sign, the integration will diverge
##divergence.terminal = True
#
#def divergence(r,y):
#    return y[1]-vinf
#divergence.terminal = True
#
#
#def hit_mach1(r,y): 
#    cs = sqrt(eos.cs2(y[0]))
#    return y[1]-cs
#hit_mach1.terminal = True # stop integrating at this point
#   
#
#
#rspan = (r0,rs)
#result2 = solve_ivp(dr_fld, rspan, (T0,v0), method='RK45', dense_output=True, events=(causality,divergence,hit_mach1), rtol=1e-6)    
#r2,(T2,u2)=result2.t,result2.y
#
#
#
#Tcross = FindCrossing(r1,T1,r2,T2)
#ucross = FindCrossing(r1,u1,r2,u2)    
#if (not Tcross[0]) or (not ucross[0]):  # no crossing
#    print('no crossing, error in radius of closest approach:',log10(abs(ucross[1][0]-Tcross[1][0])))
#    print('Magnitude of differences:',log10(Tcross[1][1]),log10(ucross[1][1]))
#else:
#    print(log10(ucross[1] - Tcross[1]))
#
#
#fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,7))
#ax1.loglog(r1,T1,label='from sonic point')
#ax2.loglog(r1,u1)
#ax1.loglog(r2,T2,label='from infinity')
#ax2.loglog(r2,u2) 
#ax1.set_xlabel('r (cm)')
#ax2.set_xlabel('r (cm)')
#ax1.set_ylabel('T (K)')
#ax2.set_ylabel('u (cm/s)')
#ax1.legend()
#plt.tight_layout()
#plt.show()



#%% 

def MakeWind_frominf(params):

    global Mdot,Edot,Ts,verbose,rs
    Mdot, Edot, Ts, verbose = W.setup_globals(params[:2],logMdot,Verbose=1,return_them=True)
    vinf = 10**params[2]
    
    gammainf = gamma(vinf)
    Linf = Edot + Mdot*c**2*(1-gammainf)
    r0 = 1e12
    
    def calc_initial_vars(x):
        v0,Lstar0 = x
        L0 = Lcomoving(Lstar0,r0,v0)
        rho0 = Mdot/(4*pi*r0**2*v0*Y(r0,v0))
        T0 = (L0/(4*pi*r0**2*arad*c))**(0.25)
        return v0,Lstar0,L0,rho0,T0
    
    def error(x):
        
        v0,Lstar0,L0,rho0,T0 = calc_initial_vars(x)
        
        E1 = Y(r0,v0) - gammainf + eos.kappa(rho0,T0)*L0/(4*pi*r0*c**3*Y(r0,v0))   # Integrated momentum equation ignoring Pg'
        E2 = (Edot + Mdot*c**2 - Lstar0 - Mdot*eos.H(rho0,T0)*Y(r0,v0))/LEdd
        return [E1,E2]
    
    x = fsolve(error,x0=[vinf,Linf])
    v0,Lstar0,L0,rho0,T0 = calc_initial_vars(x)
            
    
    def divergence(r,y):
        return y[1]-vinf
    divergence.terminal = True
    
    
    def hit_mach1(r,y): 
        cs = sqrt(eos.cs2(y[0]))
        return y[1]-cs
    hit_mach1.terminal = True # stop integrating at this point
    
    rspan = (r0,rs)
    result = solve_ivp(dr_fld, rspan, (T0,v0), method='RK45', dense_output=True, events=(divergence,hit_mach1), rtol=1e-6)    
    r,(T,u)=result.t,result.y
    print(result.message,result.t_events)
    
    return r,T,u
           

#r2,T2,u2 = MakeWind_frominf([1.025,7.175,8.38] )
#
#fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,7))
#ax1.loglog(r1,T1,label='from sonic point')
#ax2.loglog(r1,u1)
#ax1.loglog(r2,T2,label='from infinity')
#ax2.loglog(r2,u2) 



#%% Now iterate on the values of vinf 

#logvinf = 8.41
#update = 0.0005
#
#fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,7))
#ax1.loglog(r1,T1,'k-')
#ax2.loglog(r1,u1,'k-')
#ax1.set_xlabel('r (cm)')
#ax2.set_xlabel('r (cm)')
#ax1.set_ylabel('T (K)')
#ax2.set_ylabel('u (cm/s)')
#plt.close()
#
#i=1
#while update>1e-4:
#    
#    r2,T2,u2 = MakeWind_frominf([1.025,7.175,logvinf])
#    
#    fig.suptitle((r'log $v_\infty$ = %.4f'%np.round(logvinf,4)))
#    l1=ax1.loglog(r2,T2,'r-')
#    l2=ax2.loglog(r2,u2,'r-') 
#    
#    plt.pause(0.01)
#    fig.savefig('./FLD_rootsolve/vinf_demo2/'+str(i)+'.png')
#    l1.pop(0).remove()
#    l2.pop(0).remove()
#    
#    logvinf+=update
#    i+=1
#    
#    
    

#%%
#    
#logvinfs = linspace(8,9,10)
#v0=[]
#for v in logvinfs:
#    gammainf = gamma(10**v)
#    Linf = Edot + Mdot*c**2*(1-gammainf)
#    v0.append(fsolve(error,x0=[10**v,Linf])[0])
#
#plt.plot(logvinfs,log10(v0))
#    
    
    



#%% Can I go to infinity by only changing Ts? (at every Edot)
plt.close("all")

def loop_on_Ts(logMdot,Edot_LEdd):
    
    
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,7))
    ax1.set_xlabel('r (cm)')
    ax2.set_xlabel('r (cm)')
    ax1.set_ylabel('T (K)')
    ax2.set_ylabel('u (cm/s)')
    ax1.set_xlim([1e7,1e11])
    ax2.set_xlim([1e7,1e11])
    ax2.set_ylim([2e7,3e8])

#    logTsvals = np.linspace(7.0,7.2,10)
    logTsvals = np.linspace(7.17,7.18,10)
    
#    logTsvals = np.linspace(7.17673811,7.17673814,10)

        
    for logTs in logTsvals:
    
        global Mdot,Edot,Ts,verbose,rs
        Mdot, Edot, Ts, verbose = W.setup_globals([Edot_LEdd,logTs],logMdot,Verbose=1,return_them=True)
    
        rs = W.rSonic(Ts)
        print('\n\n logTs = %.8f \t ; \t logrs = %.8f'%(logTs,log10(rs)))
        
        res1 = W.outerIntegration(returnResult=True)
        print(res1.t_events)
        r1 = res1.t
        T1,phi1 = res1.sol(r1)
        u1, _,_,_ = W.calculateVars_phi(r1,T1, phi1, subsonic=False)
        
        
        if logTs!=logTsvals[0]:
            l1.set_color("black")
            l2.set_color("black")
        
        
        l1,=ax1.loglog(r1,T1,'r-',lw=0.5)
        l2,=ax2.loglog(r1,u1,'r-',lw=0.5)
        plt.pause(0.0001)

        
#    
        


#loop_on_Ts(18.5,1.025)




def loop_on_Ts_movie(logMdot,Edot_LEdd,save=1):
    
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,7))
    ax1.set_xlabel('r (cm)')
    ax2.set_xlabel('r (cm)')
    ax1.set_ylabel('T (K)')
    ax2.set_ylabel('u (cm/s)')
    ax1.set_xlim([1e6,1e11])
    ax2.set_xlim([1e6,1e11])
    ax1.set_ylim([1e6,1e8])
    ax2.set_ylim([2e7,3e8])
    if save:
        plt.close()
    else:
        plt.show()
    
    a,b = 6.8,7.8
    logTsvals = np.linspace(a,b,11)
    
    i=1
    refinement_count = 0 
    while refinement_count<10:
        
        for logTs in logTsvals:
        
            global Mdot,Edot,Ts,verbose,rs
            Mdot, Edot, Ts, verbose = W.setup_globals([Edot_LEdd,logTs],logMdot,Verbose=1,return_them=True)
        
            rs = W.rSonic(Ts)
            print('logTs = %.8f \t ; \t logrs = %.8f'%(logTs,log10(rs)))
            
            res1 = W.outerIntegration(returnResult=True)
            print(res1.message,'\n\n')
            r1 = res1.t
            T1,phi1 = res1.sol(r1)
            u1, _,_,_ = W.calculateVars_phi(r1,T1, phi1, subsonic=False)
            
            
            if not(refinement_count==0 and logTs==logTsvals[0]):
                l1.set_color("black")
                l2.set_color("black")
        
            l1,=ax1.loglog(r1,T1,'r-',lw=0.5)
            l2,=ax2.loglog(r1,u1,'r-',lw=0.5)
            fig.suptitle('Pass #%d - logTs=%.10f'%(refinement_count+1,logTs))
                         
            if save:
                fig.savefig(('./FLD_rootsolve/Ts_demo/%06d.png'%i))
            else:
                plt.pause(0.01)
            
            if res1.status==1:
                if logTs > a:
                    a = logTs
            else:
                b = logTs
                break
                    
            i+=1
                    
                    
        # update brackets
        logTsvals = np.linspace(a,b,10)
        refinement_count+=1
    
        

#loop_on_Ts_movie(18.5,1.025)


#loop_on_Ts_movie(17.5,1.015,save=0)     # 1.025 Edot does not work! would require Ts so large that rs would be in the NS :)




#%% Bissection method to integrate out to infinity
# Would need infinite precision on Ts to actually shoot to inf.  Instead stop iterating Ts until two values are very close
# Then bissection from the point at which they diverge


def Shoot_out(rspan,T0,phi0):  # so that can integrate from anywhere past the sonic point
    
    inic = [T0,phi0]

    def hit_1e8(r,y):
        return W.uphi(y[1],y[0],subsonic=False)-1e8
    hit_1e8.direction = -1
    hit_1e8.terminal = True
    
    def dv_dr_zero(r,y):
        if r>5*rs:
            return W.numerator(r,y[0],W.uphi(y[1],y[0],subsonic=False))
        else:
            return -1
    dv_dr_zero.direction = +1
    dv_dr_zero.terminal = True
        
    sol = solve_ivp(W.dr_wrapper_supersonic, rspan, inic, method='Radau', 
            events=(dv_dr_zero,hit_1e8), atol=1e-6, rtol=1e-10, dense_output=True, max_step=1e5)
    
    # if verbose: 
    #         print('FLD outer integration : ',sol.message)
            
    return sol

        
def Make_outersolution_bissection(logMdot,Edot_LEdd,logTsA,logTsB,tol=1e-5,rmax=1e11):

    # Assuming already found logTsA and logTsB such that shoot(TsA):M->1  , shoot(TsB) diverges and 
    # |logTsA-logTsB|<tol

    global Mdot,Edot,Ts,verbose,rs


    Mdot, Edot, TsA, verbose = W.setup_globals([Edot_LEdd,logTsA],logMdot,Verbose=1,return_them=True)
    rs = W.rSonic(TsA)    
    sola = W.outerIntegration(returnResult=True)

    Mdot, Edot, TsB, verbose = W.setup_globals([Edot_LEdd,logTsB],logMdot,Verbose=1,return_them=True)
    _ = W.rSonic(TsB)    
    solb = W.outerIntegration(returnResult=True)

    count=1
    def check_convergence(sola,solb,rcheck_prev):
        d = 1e9/(count+1)   # 5000km steps (count starts at 1), reduced by iteration count if stuck
        rcheck = rcheck_prev + d
        Ta,phia = sola.sol(rcheck)
        Tb,phib = solb.sol(rcheck)
        
        if abs(Ta-Tb)/Ta < tol and abs(phia-phib)/phia < tol:
            return True,rcheck,Ta,Tb,phia,phib
        else:
            return False,
        

    # Start by finding the first point of divergence
    R = linspace(rs,rmax,1000)
    for ri in R:
        conv = check_convergence(sola,solb,ri)
        if conv[0] is False:
            rfirst = ri
            break
        else:
            rcheck,Ta,Tb,phia,phib = conv[1:]

    # # test shoot_out func
    # print('\nTesting shoot_out func. first should have termination event, second should diverge')
    # _=Shoot_out((rcheck,rmax),Ta,phia)
    # _=Shoot_out((rcheck,rmax),Tb,phib)


    print('\n\n**Starting bissection**\n')
    print('Radius (km) \t Step # \t Iter count \t m')  

    # Begin bissection
    a,b = 0,1
    step_count,count = 0,1
    while rcheck<rmax:

        m = (a+b)/2
        Tm,phim = Ta + m*(Tb-Ta) , phia + m*(phib-phia)

        rspan = (rcheck,rmax)
        solm = Shoot_out(rspan,Tm,phim)

        if solm.status == 1:  # hit mach 1
            a,sola = m,solm
        else:
            b,solb = m,solm


        conv = check_convergence(sola,solb,rcheck)
        if conv[0] is True:
            rcheck,Ta,Tb,phia,phib = conv[1:]
            a,b = 0,1
            step_count += 1  # update step counter
            count = 1      # reset iteration counter


        # End of step 
        print('%.4e \t %d \t\t %d \t\t %f'%(rcheck,step_count,count,m))
        count+=1

        # Exit if stuck at a step
        nitermax=1000
        if count==nitermax:
            sys.exit("Could not integrate out to infinity! Exiting after being stuck at the same step for %d iterations"%nitermax)


# Make_outersolution_bissection(18.5,1.025,7.1767381327,7.1767381332)



#%%  Obtain the Ts-Edot relation that allows a solution to go to infty

def get_TsEdotrel(logMdot,tol=1e-6):

    # find the value of Ts that allow a solution to go to inf (to tol precision), for each value of Edot

    print('\nLOGMDOT = %.1f\n'%logMdot)

    Edotvals = linspace(1.01,1.05,10)
    Tsvals = []
    
    for Edot_LEdd in Edotvals:

        print('\nFinding Ts for Edot/LEdd = %.3f'%Edot_LEdd)

        a,b = 6.3,7.8
        logTsvals = np.linspace(a,b,15)

        while abs(b-a)>tol:

            print('%.6f\t%.6f'%(a,b))
        
            for logTs in logTsvals:
            
                global Mdot,Edot,Ts,verbose,rs
                Mdot, Edot, Ts, verbose = W.setup_globals([Edot_LEdd,logTs],logMdot,Verbose=0,return_them=True)
            
                rs = W.rSonic(Ts)
                res = W.outerIntegration(returnResult=True)

                if res.status==1:
                    a = logTs
                else:
                    b = logTs
                    break

            logTsvals = np.linspace(a,b,5)


        # Take final sonic point temperature to be bottom value (the one that leads to Mach 1.  We know the real value is in between a and a+tol)
        Tsvals.append(a)

    return Edotvals,Tsvals

        
# Edotvals,Tsvals = get_TsEdotrel(17.5)
# print(Edotvals)
# print('\n\n')
# print(Tsvals)



# Plot over old optically thick errorspaces

def plot_TsEdotrels():
    # import relations generated from function above
    with open('./TsEdotrel.txt','r') as f:

        logMdots,Edot,Ts = [[] for i in range(3)]
        i=-1
        for line in f:
            l = line.split()
            if len(l)==0:
                pass
            
            else:
                if l[0]=='logMdot':
                    logMdots.append(eval(l[2]))
                    Edot.append([])
                    Ts.append([])
                    i+=1
                elif l[0]=='Edot/LEdd':
                    pass
                else:
                    Edot[i].append(eval(l[0]))
                    Ts[i].append(eval(l[1]))

    _,ax = plt.subplots(1,1)
    ax.set_xlabel(r'$\dot{E}/L_{E}$',fontsize=15)
    ax.set_ylabel(r'log$T_s$',fontsize=15)
    for i,M in enumerate(logMdots):
        ax.plot(Edot[i],Ts[i],'--',label=str(M))
    ax.legend(title=r'log$\dot{M}$')
    plt.tight_layout()


    # import errormaps 
    import pickle

    fig,axes = plt.subplots(1,len(logMdots),figsize=(18,8))    
    fig.subplots_adjust(wspace=0.25)
    cmap = plt.get_cmap('YlOrRd') # diverging colormap (white at bottom)
    levs=np.logspace(-3,1,50)
    fig.suptitle(r'Error : $\vert r_{b}-R_{NS}\vert$/$R_{NS}$',fontsize=15)
    axes[0].set_ylabel(r'log $T_s$ (K)')
    for i,ax in enumerate(axes):
        ax.patch.set_color('.25')
        ax.set_xlabel(r'$\dot{E}/L_{Edd}$')

        filename = './analysis/errorspaces/save'+str(logMdots[i])+'.p'
        [Edotvals,Tsvals,Errors]=pickle.load(open(filename,'rb'))

        im = ax.contourf(Edotvals,Tsvals,np.abs(array(Errors[1])),levs,norm=colors.LogNorm(),cmap=cmap)
        xlims,ylims = ax.get_xlim(),ax.get_ylim()
        ax.plot(Edot[i],Ts[i],'k--',label=str(M))
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_title(r'log$\dot{M}$=%.1f'%logMdots[i])
        
        fig.colorbar(im,ax=ax,ticks=[1e-3, 1e-2, 1e-1, 1e0, 1e1])
 



    plt.show()
            

plot_TsEdotrels()








