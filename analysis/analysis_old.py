''' Analysis of wind solutions (not part of main code) ''' 

import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import log10, pi, array, linspace, logspace, where, exp

from wind_GR import *
from IO import *

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
# warnings.filterwarnings("ignore", category=ODEintWarning) 

# Parameters
M, RNS, y_inner, tau_out, comp, mode, save, img = load_params()

# Constants
c = 2.99292458e10
h = 6.62e-27
kappa0 = 0.2
GM = 6.6726e-8*2e33*M
LEdd = 4*pi*c*GM/kappa0
ZZ = (1-2*GM/(c**2*RNS*1e5))**(-1/2) # redshift
g = GM/(RNS*1e5)**2 * ZZ

# Import Winds
clean_rootfile()
logMDOTS,roots = load_roots()

# Extracted terms
Lb = [] # base luminosity
Tb = [] # base temperature
Lbs = [] # base luminosity at infinity
rson = [] # sonic radius

# fig,ax = plt.subplots(1,1)
# ax.set_xlabel('r')
# ax.set_ylabel('dlnv_dlnr')

# fig2,ax2 = plt.subplots(1,1)
# ax2.set_xlabel('r')
# ax2.set_ylabel('lnT gradient terms (abs value)')

i = 0
for logMdot, root in zip(logMDOTS, roots):

    print(logMdot)
    global Mdot, verbose
    Mdot, verbose = 10**logMdot, 0

    R, T, Rho, u, Phi, Lstar, L, LEdd_loc, E, P, cs, tau, rs, Edot, Ts = MakeWind(
        root, logMdot, mode='wind')

    Lb.append(L[0])
    Lbs.append(Lstar[0])
    rson.append(rs)
    Tb.append(T[0])

    # dlnT_dlnr = []
    # dlnv_dlnr = []
    # term1 = []
    # term2 = []
    # for i in range(len(R)):
    #     r,temp,vel,rho = R[i],T[i],u[i],Rho[i]

    #     if r==rs:
    #         dlnv_dlnr.append(0)
    #     else:
    #         dlnv_dlnr.append( numerator(r,temp,vel)/ (cs2(temp) - vel**2*A(temp)) )

    #     term1.append( -Tstar(Lstar[i], temp, r, rho , vel) - 1/Swz(r) * GM/c**2/r )
    #     term2.append( -gamma(vel)**2 * (vel/c)**2 * dlnv_dlnr[-1] )
    #     dlnT_dlnr.append( term1[-1] + term2[-1] )

    # ax.semilogx(R,dlnv_dlnr,label=('%.1f' % (log10(Mdot))))
    # ax.semilogx([rs,rs],[-10,10],'k--',lw=0.3)

    # if logMdot==18.5:
    #     ax2.loglog(R,np.abs(dlnT_dlnr),'k-',lw=1.5,label='dlnT_dlnr')
    #     ax2.loglog(R,np.abs(term1),'r--',lw=0.5,label='term 1')
    #     ax2.loglog(R,np.abs(term2),'b--',lw=0.5,label='term 2')

# ax.legend(title=r'log $\dot{M}$', loc=1)
# ax.set_ylim([-10,10])

# ax2.legend()

###### Other Plots #######

# Base Luminosity
Lb, Lbs, Tb = array(Lb), array(Lbs), array(Tb)
Fb = Lb/(4*pi*(RNS*1e5)**2) # non-redshifted base flux
Fbs = Lbs/(4*pi*(RNS*1e5)**2) # redshifted base flux
Mdots = 10**array(logMDOTS)

# fig1, (ax11, ax12) = plt.subplots(1, 2, figsize=(15, 8))
# ax11.plot(logMDOTS, Lb/LEdd, 'ro-', label=r'Local Luminosity')
# ax11.plot(logMDOTS, Lbs/LEdd, 'bo-', label=r'Luminosity at $\infty$')
# ax11.plot([logMDOTS[0], logMDOTS[-1]], [1, 1], 'k--')
# ax11.set_xlabel(r'log $\dot{M}$', fontsize=14)
# ax11.set_ylabel(r'$L_b/L_{Edd}$', fontsize=14)
# ax11.legend()
# ax12.plot(logMDOTS, Lb/Lbs, 'ko-')
# ax12.set_xlabel(r'log $\dot{M}$', fontsize=14)
# ax12.set_ylabel(r'$Local/Infinity$', fontsize=14)
# ax12.set_ylim([1, 2])

# L_error, Lstar_error = (Lb-LEdd)/(GM*Mdots/1e6), (Lbs-LEdd)/(GM*Mdots/1e6)
# fig2, ax2 = plt.subplots(1, 1)
# ax2.plot(logMDOTS, L_error, 'ro-', label=r'Comoving Luminosity')
# ax2.plot(logMDOTS, Lstar_error, 'bo-', label=r'Luminosity at $\infty$')
# ax2.plot([logMDOTS[0], logMDOTS[-1]], [1, 1], 'k--')
# ax2.set_xlabel(r'log $\dot{M}$', fontsize=14)
# ax2.set_ylabel(r'$(L_b-L_{Edd})/(GM\dot{M}/R)$', fontsize=14)
# ax2.legend()



# Base flux and luminosity
# fig3,ax31 = plt.subplots(1,1)
# ax31.plot(logMDOTS,Tb/1e9,'ko-')
# ax31.set_xlabel(r'log $\dot{M}$ (g s$^{-1}$)', fontsize=14)
# ax31.set_ylabel(r'T$_b$ (10$^{9}$ K) ', fontsize=14)

# ax32 = ax31.twinx()
# ax32.plot(logMDOTS,Fb/1e25,'bo-')
# ax32.set_ylabel(r'F$_b$ (10$^{35}$ erg s$^{-1}$ cm$^{-2}$)', fontsize=14)



# fig4,ax4 = plt.subplots(1,1)
# ax4.plot(Fb/1e25,Tb/1e9,'ko',markerfacecolor='w')
# ax4.set_xlabel(r'F$_b$ (10$^{35}$ erg s$^{-1}$ cm$^{-2}$)', fontsize=14)
# ax4.set_ylabel(r'T$_b$ (10$^{9}$ K) ', fontsize=14)
# ax4.text(Fb[0]/1e25+0.1,Tb[0]/1e9-0.01,(r'log $\dot{M}$ = '+str(logMDOTS[0])),fontweight='bold',color='r')
# for i in range(1,7):
#     ax4.text(Fb[i]/1e25+0.1,Tb[i]/1e9-0.01,str(logMDOTS[i]),fontweight='bold',color='r')

# Fit 
#func = interp1d(Fb,Tb,kind='cubic')
#x = linspace(Fb[0]-0.1,Fb[-1]+0.1,200)
#y = func(x)
# ax4.plot(x/1e25,y/1e9,'k-')

# Radiation temperature
arad = 7.5657e-15
P_inner = g*y_inner
Trad = (3*P_inner/arad)**(1/4)
# ax4.axhline(Trad/1e9,color='k',linestyle='--')
# ax4.text(Fb[0]/1e25,Trad/1e9-0.015,'radiation temperature')


# Redshifted
# fig5,ax5 = plt.subplots(1,1)
# ax5.plot(Fbs/1e25,Tb/1e9,'ko-')
# ax5.set_xlabel(r'F$_{b,red}$ (10$^{35}$ erg s$^{-1}$ cm$^{-2}$)', fontsize=14)
# ax5.set_ylabel(r'T$_b$ (10$^{9}$ K) ', fontsize=14)

# ax5.text(Fbs[0]/1e25+0.1,Tb[0]/1e9-0.01,(r'log $\dot{M}$ = '+str(logMDOTS[0])),fontweight='bold',color='r')
# for i in range(1,7):
#     ax5.text(Fbs[i]/1e25+0.1,Tb[i]/1e9-0.01,str(logMDOTS[i]),fontweight='bold',color='r')



#%%

## Light curve data

# Extract luminosity as a function of time
data = np.loadtxt('light_curve/prof', usecols=(0,1))
t = data[:,0]
L = data[:,1] 


# Extract base temperature as a function of time
t2,Tb2,Fb2 = [],[],[]

def append_vars(line,varz,cols): # take line of file and append its values to variable lists 
    l=line.split()
    for var,col in zip(varz,cols):
        var.append(float(l[col]))

with open('light_curve/out','r') as out:
    
    # Number of grid points
    ngrid = int(out.readline())

    # Looping through datafile
    count=0
    y,T,FF = [],[],[]
    for line in out:
        l = line.split()

        if len(l)==1:
            if count>0: 
                Tb2.append(T[0])        # base temperature corresponds to y=1e8 (first value)
                Fb2.append(FF[0])
                y,T,FF = [],[],[]       # reset arrays at every timestep
            
            count += 1      
            t2.append(float(l[0]))

        else: append_vars(line,[y,T,FF],[0,1,2])




## Follow evolution of light-curve over time and resulting wind

def lc_plot(setup='12'):
    
    if setup=='12':
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,6))
    elif setup=='22':
        fig,axes = plt.subplots(2,2,figsize=(13,13))
        ax1,ax2 = axes[0]
        ax3,ax4 = axes[1]
        
    ax1.plot(t,L/1e38,'b--',label='Calculated light curve')
    ax1.set_xlim([0,200])
    ax1.set_xlabel('t (s)', fontsize=14)
    #ax1.set_ylabel(r'L$_{b,red}$ (10$^{38}$ erg s$^{-1}$)', fontsize=14)
    ax1.set_ylabel(r'L (10$^{38}$ erg s$^{-1}$)', fontsize=14)
    
    # observed light curve : doesnt go above Edd
    L2=[]
    for i in range(len(L)):
        if L[i]>LEdd:
            L2.append(LEdd)
        else:
            L2.append(L[i])
    L2 = array(L2)
    
    ax1.axhline(LEdd/1e38,color='m',linestyle=':')
    ax1.text(180,LEdd/1e38-0.3,r'L$_{Edd}$',fontsize=14)
    ax1.plot(t,L2/1e38,'b-',label='Observed light curve')
    ax1.legend()
    
    if setup == '12':
        return fig,ax1,ax2
    elif setup == '22':
        return fig,ax1,ax2,ax3,ax4




fig6,ax61,ax62 = lc_plot()

ax62.plot(Fb/1e25,Tb/1e9,'k',lw=0.5,alpha=0.5)
#ax2.set_xlabel(r'L$_{b,red}$ (10$^{38}$ erg s$^{-1}$)', fontsize=14)
ax62.set_xlabel(r'F$_{b}$ (10$^{25}$ erg s$^{-1}$)', fontsize=14)
ax62.set_ylabel(r'T$_b$ (10$^{9}$ K) ', fontsize=14)

# Write a few Mdots as text : 18,18.1,18.2,18.3,18.4,18.5,18.6
for i in range(len(logMDOTS)):
    if logMDOTS[i]==18.0:
#        ax62.text(Lbs[i]/1e38+0.1,Tb[i]/1e9-0.01,(r'log $\dot{M}$ = '+str(logMDOTS[i])),fontweight='bold',color='r')
#        ax62.plot(Lbs[i]/1e38,Tb[i]/1e9,'ko',markerfacecolor='w')
        ax62.text(Fb[i]/1e25+0.1,Tb[i]/1e9-0.01,(r'log $\dot{M}$ = '+str(logMDOTS[i])),fontweight='bold',color='r')
        ax62.plot(Fb[i]/1e25,Tb[i]/1e9,'ko',markerfacecolor='w')
    elif logMDOTS[i] in (17.2,17.8,17.9,18,18.1,18.2,18.3,18.4,18.5,18.6):
#        ax62.text(Lbs[i]/1e38+0.1,Tb[i]/1e9-0.01,str(logMDOTS[i]),fontweight='bold',color='r')
#        ax62.plot(Lbs[i]/1e38,Tb[i]/1e9,'ko',markerfacecolor='w')
        ax62.text(Fb[i]/1e25+0.1,Tb[i]/1e9-0.01,str(logMDOTS[i]),fontweight='bold',color='r')
        ax62.plot(Fb[i]/1e25,Tb[i]/1e9,'ko',markerfacecolor='w')


ax62.axhline(Trad/1e9,color='k',linestyle='--')
ax62.text(Fb[0]/1e25,Trad/1e9-0.015,'radiation temperature')

func = interp1d(Fb,Tb,kind='cubic')
x = linspace(Fb[0],Fb[-1],200)
y = func(x)
#ax62.plot(x/1e38,y/1e9,'k-')
ax62.plot(x/1e25,y/1e9,'k-')

#ax62.set_xlim(3.5,Lbs[-1]/1e38+0.1)
ax62.set_xlim(3,Fb[-1]/1e25+0.1)
ax62.set_ylim(1.2,1.6)







# # Animation
# i,ti=0,0
# while ti<100:
#    ti = t2[i]

#    time,Lum = t[where(t<ti)],L[where(t<ti)]
#    ax61.plot(time,Lum/1e38,'g-')
   
   
#    if i>1 and Lum[-1]>LEdd:
#        p=ax62.plot([Fb2[i]/1e25],[Tb2[i]/1e9],'g.')


#        plt.pause(0.01)
# #    fig6.savefig(('light_curve/animation/%06d.png'%i))
#        p.pop(0).remove()


#    i+=1



#plt.show()




#%% Matching light curve to wind
'''
root1,root2 = [],[]
for root in roots : 
    root1.append(root[0])
    root2.append(root[1])
    
rootfunc1 = interp1d(logMDOTS,root1,kind='cubic')
rootfunc2 = interp1d(logMDOTS,root2,kind='cubic')

# x = linspace(logMDOTS[0],logMDOTS[-1])
# y1=rootfunc1(x)
# y2=rootfunc2(x)

# fig7,(ax71,ax72) = plt.subplots(1,2)
# ax71.plot(logMDOTS,root1,'b.')
# ax71.plot(x,y1,'b-')
# ax71.set_xlabel(r'log $\dot{M}$')
# ax71.set_ylabel(r'$\dot{E}/L_{Edd}$')

# ax72.plot(logMDOTS,root2,'r.')
# ax72.plot(x,y2,'r-')
# ax72.set_xlabel(r'log $\dot{M}$')
# ax72.set_ylabel(r'log $T_s$ (K)')





# At each time where luminosity is above eddington, find Mdot correpsonding to flux


fluxfunc = interp1d(Fb,logMDOTS,kind='linear')
#
#fig7,ax7 = plt.subplots(1,1)
#ax7.plot(Fb/1e25,logMDOTS,'bo')
#ax7.plot(x/1e25,y1,'b-')
#ax7.set_xlim(Fb[0]/1e25-0.5,Fb[-1]/1e25+0.5)
#ax7.set_ylim([17.5,19.5])

FEdd = LEdd/(4*pi*(RNS*1e5)**2)
    

#fig8,ax81,ax82 = lc_plot()
#plt.close('all')

#ax82.set_xlabel(r'$r$ (cm)', fontsize=14)
#ax82.set_xlabel(r'$r$ (km)', fontsize=14)
#ax82.set_ylabel(r'$v$ (cm s$^{-1}$)', fontsize=14)
#ax82.set_xlim([0,700])
#ax82.set_ylim([0.5,4e8])
#

radius_arrays,velocity_arrays = [],[]
phot_radii,phot_temps = [],[]

for i in range(len(Fb2)):
    
    # light curve
    ti = t2[i]
#    time,Lum = t[where(t<ti)],L[where(t<ti)]
#    ax81.plot(time,Lum/1e38,'g-',lw=2.5)
    print(ti)
    
    if Fb2[i]>FEdd:
        
        # wind
        if Fb2[i]<Fb[0]:
            flux = Fb[0]
        else:
            flux = Fb2[i]
            
        logmdot = fluxfunc(flux)
        
#        p=ax7.plot([flux/1e25],[logmdot],'m.')

        root = [rootfunc1(logmdot),rootfunc2(logmdot)]
        Rad, Temp, Rho, vel, Phi, Lstar, Lwind, LEdd_loc, E, P, cs, tau, rs, Edot, Ts = MakeWind(
        root, logmdot, mode='wind')

#        l = ax82.semilogy(Rad/1e5, vel, 'k-')
        
        
        # save velocities and photosphere data
        radius_arrays.append(Rad)
        velocity_arrays.append(vel)
        phot_radii.append(Rad[-1])
        phot_temps.append(Temp[-1])


#    plt.pause(0.01)
#    fig8.savefig(('light_curve/animation2/%06d.png'%i))
    
#        p.pop(0).remove()
#    if Fb2[i]>FEdd: l.pop(0).remove()
    
    
phot_temps = array(phot_temps)/1e6 

#%% Drawing circle size of photosphere with color

#fig8,ax81,ax82 = lc_plot()
#
#ax82.set_xlabel(r'km', fontsize=14)
#ax82.set_ylabel(r'km', fontsize=14)
#ax82.set_xlim([-1000,1000])
#ax82.set_ylim([-1000,1000])
#ax82.set_aspect('equal')
#
#NS = plt.Circle((0,0),12,color='k')
#ax82.add_artist(NS)


# color scale
import matplotlib.colors
norm = matplotlib.colors.Normalize(vmin=min(phot_temps)*0.98, vmax=max(phot_temps)*1.1)
cmap = plt.get_cmap('hot')


#
#fig8.subplots_adjust(right=0.9)
#cbar_ax = fig8.add_axes([0.9, 0.15, 0.02, 0.7])
#sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#sm.set_array([])
#fig8.colorbar(sm,cax=cbar_ax)
#cbar_ax.set_title(r'T (10$^6$ K)')

#j=0
#for i in range(len(Fb2)):
#    
#    # light curve
#    ti = t2[i]
#    time,Lum = t[where(t<ti)],L[where(t<ti)]
#    ax81.plot(time,Lum/1e38,'g-',lw=2.5)
#    print(ti)
    
#    if Fb2[i]>FEdd:
#        
#        circle = plt.Circle((0, 0), phot_radii[j]/1e5 , color=cmap(norm(phot_temps[j])))
#        ax82.add_artist(circle)
#        ax82.add_artist(NS)
#        j+=1

#    plt.pause(0.01)
#    fig8.savefig(('light_curve/animation3/%06d.png'%i))
    
#        p.pop(0).remove()
#    if Fb2[i]>FEdd: circle.remove()
    
    
#%% The whole thing (light curve, velocity profile, photosphere, blackbody spectrum)
plt.close('all')

eV = 1.60e-12 # eV to erg
def BB_keV(T,E): # give spectral range in keV, returns normalized flux
    nu = array(E)*1e3*eV/h
    B = nu**3/(exp(h*nu/kB/T)-1)
    B = array(B)/max(B)
    peak = E[argwhere(B==max(B))]
    return B,peak[0][0]
E = linspace(0.01,5,500)


fig,ax1,ax2,ax3,ax4 = lc_plot(setup='22')

NS = plt.Circle((0,0),12,color='k')
ax2.add_artist(NS)
#ax2.set_xlabel(r'km', fontsize=14)
ax2.set_ylabel(r'km', fontsize=14)
ax2.set_xlim([-1000,1000])
ax2.set_ylim([-1000,1000])
ax2.set_aspect('equal')

ax3.set_xlabel(r'$r$ (km)', fontsize=14)
ax3.set_ylabel(r'$v$ (cm s$^{-1}$)', fontsize=14)
ax3.set_xlim([0,700])
ax3.set_ylim([0.5,4e8])

ax4.set_xlabel(r'E (keV)', fontsize=14)
ax4.set_ylabel(r'B/B$_{max}$',fontsize=14)
ax4.set_yticks([])
ax4.axvline(1,color='k',linestyle=':')


# Colorbar for ax2
fig.subplots_adjust(right=0.95)
cbar_ax = fig.add_axes([0.9, 0.53, 0.02, 0.35])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm,cax=cbar_ax)
cbar_ax.set_title(r'T (10$^6$ K)')
    

j=0
for i in range(len(Fb2)):
    
    # light curve
    ti = t2[i]
    time,Lum = t[where(t<ti)],L[where(t<ti)]
    ax1.plot(time,Lum/1e38,'g-',lw=2.5)
    print(ti)
    
    if Fb2[i]>FEdd:
        
        circle = plt.Circle((0, 0), phot_radii[j]/1e5 , color=cmap(norm(phot_temps[j])))
        ax2.add_artist(circle)
        ax2.add_artist(NS)
        
        l = ax3.semilogy(radius_arrays[j]/1e5, velocity_arrays[j], 'k-')
        
        B,_ = BB_keV(phot_temps[j]*1e6,E)
        l2 = ax4.plot(E,B,'r')

        j+=1
        
        
    plt.pause(0.01)
#    fig.savefig(('light_curve/animation4/%06d.png'%i))
    

    if Fb2[i]>FEdd: 
        circle.remove()
        l.pop(0).remove()
        l2.pop(0).remove()
        
        




#%% Typical plot of L,Tbb,Rbb
        
fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(8,10))
    
for ax in (ax1,ax2,ax3):
    ax.set_xlim([0,300])

ax1.set_ylabel(r'L (10$^{38}$ erg s$^{-1}$)', fontsize=14) 
ax2.set_ylabel(r'kT$_{bb}$ (keV)', fontsize=14)
ax3.set_ylabel(r'R$_{bb}$ (km)', fontsize=14)
ax3.set_xlabel('Time (s)', fontsize=14)


L2 = []
for i in range(len(L)):
    if L[i]>LEdd:
        L2.append(LEdd)
    else:
        L2.append(L[i])
L2 = array(L2)

ax1.plot(t,L2/1e38)

E=np.linspace(0.01,5,10000)
kTbb,Rbb = [],[]  
j=0
for i in range(len(Fb2)):
    
    if Fb2[i]>FEdd:
        B,peak = BB_keV(phot_temps[j]*1e6,E)
        kTbb.append(peak)
        Rbb.append(phot_radii[j]/1e5)
        j+=1
    else:
#        Tbb.append(Tb2[i])
#        Tbb.append(0)
        Rbb.append(10)
        
Rbb,Tbb=array(Rbb),array(Tbb)  
ax2.plot(t[:len(kTbb)],kTbb)
ax3.plot(t[:-1],Rbb)


'''

plt.show()