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
#        print('no crossing')
        return (False,)

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
        return Tcross[1] - ucross[1]



# Parameters
#logMdot = 18.5
#params = [1.025,7.2,8.42025]           # Edot/LEdd, logTs, logvinf

#params = [1.025,7.2,8.4]
#params = [1.025,6.9,8]

#E = MakeWindFLD(params, logMdot, Verbose=1)
    

#MakeWindFLD([1.025,7.2,8.42], logMdot, Verbose=1)

#%% Errorspace
import pickle
def get_map(logMDOT,Edot):

    n=10
    Errors = [[] for i in range(n)] 
    Tsvals = np.linspace(7.1,7.3,n)
    vinfvals = np.linspace(8.2,8.7,n)
    
    for i,Ts in zip(np.arange(n),Tsvals):
        print('\n\n LOG TS = %.3f \n\n'%Ts)
        for vinf in vinfvals:
            try:
                z = MakeWindFLD([Edot,Ts,vinf],logMDOT,Verbose=0)
                print('log(vinf) = %.3f : %.3e'%(vinf,z))
            except:
#                pass
                z=1000
#                 print("\nFatal error in integration, skipping...\n")
            

            Errors[i].append(z)
            

    filename = 'analysis/errorspaces_FLD/save'+str(logMDOT)+'_'+str(Edot/LEdd)+'.p'
    with open(filename,'wb') as f:
        pickle.dump([Tsvals,vinfvals,Errors],f)
        
    return (Tsvals,vinfvals,Errors)
    
#
#    print('\n\nFINISHED \nMap file saved to ',filename)    
            
            
map1 = get_map(18.5,1.025)


#%% PLOT MAP
#
#
#def plot_map(map):
#
#    filename = 'analysis/errorspaces/save'+str(logMDOT)+'.p'
#    [Edotvals,Tsvals,Errors]=pickle.load(open(filename,'rb'))
#
#    fig,(ax1,ax2) = plt.subplots(1,2,figsize=set_size('mnras1col'))    
#    fig.subplots_adjust(wspace=0.25)
#    ax1.patch.set_color('.25')
#    ax2.patch.set_color('.25')
#
#    # Option 1 
##     cmap = plt.get_cmap('RdBu') # diverging colormap (white at center)
##     levels=np.arange(-1,1,0.01)
##     im1 = ax1.contourf(Edotvals,Tsvals,array(Errors[0]),levels=levels,cmap=cmap)
##     im2 = ax2.contourf(Edotvals,Tsvals,array(Errors[1]),levels=levels,cmap=cmap)
##     fig.colorbar(im1,ax=ax1)
##     fig.colorbar(im2,ax=ax2)
##     ax1.set_title(r'Error #1 : ($L_{phot}-4\pi r^2\sigma T^4$)',fontsize=15)
##     ax2.set_title(r'Error #2 : ($R_{base}-R_{NS}$)',fontsize=15)    
#
#    # Option 2
#    cmap = plt.get_cmap('YlOrRd') # diverging colormap (white at bottom)
#    levs=np.logspace(-3,1,50)
#    im1 = ax1.contourf(Edotvals,Tsvals,np.abs(array(Errors[0])),levs,norm=colors.LogNorm(),cmap=cmap)
#    im2 = ax2.contourf(Edotvals,Tsvals,np.abs(array(Errors[1])),levs,norm=colors.LogNorm(),cmap=cmap)
#    fig.colorbar(im1,ax=ax1,ticks=[1e-3, 1e-2, 1e-1, 1e0, 1e1])
#    fig.colorbar(im2,ax=ax2,ticks=[1e-3, 1e-2, 1e-1, 1e0, 1e1])
#    ax1.set_title(r'Error 1 : $\vert L_{ph}-4\pi r^2\sigma T^4\vert$/$L_{ph}$')
#    ax2.set_title(r'Error 2 : $\vert r_{b}-R_{NS}\vert$/$R_{NS}$')    
#
#
#    ax1.set_xlabel(r'$\dot{E}/L_{Edd}$')
#    ax2.set_xlabel(r'$\dot{E}/L_{Edd}$')
#    ax1.set_ylabel(r'log $T_s$ (K)')
#
#                
#    ax1.set_xlim([Edotvals[0],Edotvals[-1]])
#    ax1.set_ylim([Tsvals[0],Tsvals[-1]])  
#    ax2.set_xlim([Edotvals[0],Edotvals[-1]])
#    ax2.set_ylim([Tsvals[0],Tsvals[-1]])     
#    
#    
#    # Add scatters for errors
#    cases_left = (r'Mach $\rightarrow 1$',r'$\tau^*_{min}>3$','?','tau^*(r_c)<3')
#    cases_right = ('u=0','Diverging (NaN)','?')
#    color_left = ('m','c','w')
#    color_right = ('r','g','k')
#    legendleft,legendright = False,False
#    
#    for E,caseL,caseR,cl,cr in zip((100,200,300,400),cases_left,cases_right,color_left,color_right):
#    
#        # Left plot (error on outer int)
#        Epointsleft,Tspointsleft = [],[]
#        for i in range(len(Edotvals)):
#            for j in range(len(Tsvals)):
#                if Errors[0][j][i]==E:
#                    Epointsleft.append(Edotvals[i])
#                    Tspointsleft.append(Tsvals[j])
#        
#        if len(Epointsleft)>0:
#            ax1.scatter(Epointsleft,Tspointsleft,s=3,label=caseL,color=cl)
#            legendleft=True
#            
#        
#        # Right plot (error on inner int)
#        Epointsright,Tspointsright = [],[]
#        for i in range(len(Edotvals)):
#            for j in range(len(Tsvals)):
#                if Errors[1][j][i]==E:
#                    Epointsright.append(Edotvals[i])
#                    Tspointsright.append(Tsvals[j])
#        
#        if len(Epointsright)>0:
#            ax2.scatter(Epointsright,Tspointsright,s=3,label=caseR,color=cr)
#            legendright=True
#    
#    
#    if legendleft: ax1.legend(loc=4)
#    if legendright: ax2.legend(loc=4)
#        
#    
#    # plt.tight_layout()
#    
#    filename = str(logMDOT).replace('.','_') # dont want points in filenames
#    fig.savefig('analysis/errorspaces/'+filename+'.'+img)    
#
#            
#            
            
#%% Manual 
            
#            
#
#logMdot = 18.5
#
##params = [1.025,7.2,8.42]             
#params =  [1.025,7,8]
#
#
#global Mdot,Edot,Ts,verbose,rs,vinf
#Mdot, Edot, Ts, verbose = W.setup_globals(params[:2],logMdot,Verbose=1,return_them=True)
#vinf = 10**params[2]
#
#
#rs = W.rSonic(Ts)
#res1 = W.outerIntegration(returnResult=True)
#r1 = res1.t
#T1,phi1 = res1.sol(r1)
#u1, _,_,_ = W.calculateVars_phi(r1,T1, phi1, subsonic=False)
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
##    divergence.direction = -1  # velocity gradient becomes negative (v goes up as we go in, we're diverging for sure)
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
#   
#rspan = (r0,2*rs)
#result2 = solve_ivp(dr_fld, rspan, (T0,v0), method='RK45', dense_output=True, events=(causality,divergence,hit_mach1), rtol=1e-6)    
#r2,(T2,u2)=result2.t,result2.y
#
#
#fig,(ax1,ax2)=plt.subplots(1,2)
#ax1.loglog(r1,T1)
#ax2.loglog(r1,u1)
#ax1.loglog(r2,T2)
#ax2.loglog(r2,u2)
