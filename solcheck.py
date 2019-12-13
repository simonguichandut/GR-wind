''' Checking errors on solutions '''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.interpolate import InterpolatedUnivariateSpline as IUS  # has a derivative method
from scipy.interpolate import interp1d
from numpy import log10,linspace

from wind_GR import MakeWind,dr





def plot_stuff(radius,rs,T_points,phi_points,T_func,phi_func,dT_points,dphi_points,title):
    '''T_points and phi_points are the actual points from the solution (same for drho and dT)
       T_points and phi_func are some kind of fit of the date, like a spline, but they HAVE TO have a .derivative method()'''

    fig= plt.figure(figsize=(12,8))
    fig.suptitle(title,fontsize=15)

    gs = gridspec.GridSpec(3, 2, height_ratios=[3, 3, 2])
    axes = []
    for i in range(6): axes.append(plt.subplot(gs[i]))
    ax1,ax2,ax3,ax4,ax5,ax6 = axes

 
    ax1.set_ylabel(r'log T (K)',fontsize=14)
    ax2.set_ylabel(r'log $\phi$',fontsize=14)
    ax3.set_ylabel(r'log |$dT/dr$|',fontsize=14)
    ax4.set_ylabel(r'log |$d\phi/dr$|',fontsize=14)
    ax5.set_ylabel('Rel. error (%)',fontsize=14)
    ax6.set_ylabel('Rel. error (%)',fontsize=14)
    ax5.set_xlabel(r'log $r$ (km)',fontsize=14)
    ax6.set_xlabel(r'log $r$ (km)',fontsize=14)
    ax5.set_ylim([-10,10])
    ax6.set_ylim([-10,10])

    x=radius/1e5
    ax1.plot(x,log10(T_points),'k.',label='Solution',ms=6,alpha=0.5)
    ax1.plot(x,log10(T_func(radius)),'b-',label='Fit')
    ax2.plot(x,log10(phi_points),'k.',label='Solution',ms=6,alpha=0.5)
    ax2.plot(x,log10(phi_func(radius)),'b-',label='Fit')
    ax3.plot(x,log10(np.abs(T_func.derivative()(radius))),'b-',label='Fit derivative')
    ax3.plot(x,log10(np.abs(dT_points)),'k.',label='Direct derivative',ms=6,alpha=0.5)
    ax4.plot(x,log10(np.abs(phi_func.derivative()(radius))),'b-',label='Fit derivative')
    ax4.plot(x,log10(np.abs(dphi_points)),'k.',label='Direct derivative',ms=6,alpha=0.5)
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()

    # Errors
    relerr_rho = (dT_points-T_func.derivative()(radius))/dT_points
    relerr_T = (dphi_points-phi_func.derivative()(radius))/dphi_points
    ax5.plot(x[1:],relerr_rho[1:]*100,'k-',lw=1.5)
    ax6.plot(x[1:],relerr_T[1:]*100,'k-',lw=1.5)            # edge points of spline fit don't work

    for ax in axes: ax.axvline(rs/1e5,color='m',lw=0.5)
        
    plt.tight_layout(rect=(0,0,1,0.95))



def check_solution(logMdot, wind):

    ''' checks solution vector's direct derivatives against analytic expressions '''
    
    # Spline fit
    fT,fphi = IUS(wind.r,wind.T), IUS(wind.r,wind.phi)

    # Get points on spline
    rfine = linspace(wind.r[0],wind.r[-1],10000)
    T, phi = fT(rfine), fphi(rfine)
    
    # Analytical derivatives
    dT,dphi = [],[]
    # for ri,Ti,phii  in zip(wind.r,wind.T,wind.phi):
    for ri,Ti,phii in zip(rfine,T,phi):
        subsonic = True if ri<wind.rs else False
        # print(inwards)
        z = dr(ri,[Ti,phii],subsonic=subsonic)
        dT.append(z[0])
        dphi.append(z[1])

    # plot_stuff(wind.r,wind.rs,wind.T,wind.phi,fT,fphi,dT,dphi,'Error')
    plot_stuff(rfine,wind.rs,T,phi,fT,fphi,dT,dphi,'Error')


from IO import load_roots
x,z = load_roots()

wind = MakeWind(z[20],x[20],mode='wind')
# fig=plt.figure()
# plt.loglog(R,Phi,'b.')
check_solution(x[20],wind) 
plt.show()