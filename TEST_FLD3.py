## This was for when I did C/3lambda, which was from the wrong momentum equation

from wind_GR import *
import IO
import physics
import sys
from numpy import array

M, RNS, y_inner, tau_out, comp, EOS_type, FLD, mode, save, img = IO.load_params()
eos = physics.EOS(comp)
LEdd = 4*pi*c*GM/eos.kappa0


if FLD is False:
    # sys.exit("FLD needs to be on!!")
    print('\n\nFLD is not on fyi!\n\n')



# Testing integrations
logMdot = 18.5
params=[1.025426   ,  7.196667]
# params=[1.025,7]
# params=[1.001   ,  7.3]

# logMdot = 19
# params=[1.020872   ,  6.958264]

# logMdot = 18
# params=[1.03,7.4]


if FLD:
    winds = MakeWind(params, logMdot, mode='wind',Verbose=1)
    w1 = winds[0]
    r,T,rho,u,phi,Lstar,L,LEdd_loc,E,P,cs,tau,lam,rs,Edot,Ts = w1
    if len(winds)==1:
        r2,T2,rho2,u2,phi2,Lstar2,L2,LEdd_loc2,E2,P2,csv2,tau2,lam2 = [array([w1[i][-1]]) for i in range(13)]
    else:
        w2 = winds[1]
        r2,T2,rho2,u2,phi2,Lstar2,L2,LEdd_loc2,E2,P2,csv2,tau2,lam2,rs,Edot,Ts = w2

else:                                                           # Non-fld solutions: cut from rs
    w1 = MakeWind(params, logMdot, mode='wind',Verbose=1)
    rs,Edot,Ts=w1.rs,w1.Edot,w1.Ts
    indexrs=np.argwhere(w1.r==w1.rs)[0][0]
    r,T,rho,u,phi,Lstar,L,LEdd_loc,E,P,cs,tau,lam = [w1[i][indexrs:] for i in range(13)]
    r2,T2,rho2,u2,phi2,Lstar2,L2,LEdd_loc2,E2,P2,csv2,tau2,lam2 = [array([w1[i][-1]]) for i in range(13)]


rs2=r[-1]



### Plots

import matplotlib.pyplot as plt
fig,((ax1,ax2,ax3),(ax4,ax5,ax6))=plt.subplots(2,3,figsize=(15,8))
fig.suptitle(r'Outer integration from sonic point with $T$ and $\phi$',fontsize=16)#,y=1.08)
ax1.set_ylabel(r'T (K)',fontsize=16)
ax2.set_ylabel(r'$\phi$',fontsize=16)
ax3.set_ylabel(r'$\lambda$',fontsize=16)
ax4.set_ylabel(r'$\rho$',fontsize=16)
ax5.set_ylabel(r'$L$ (erg s$^{-1}$)',fontsize=16)
ax6.set_ylabel(r'$\tau^*=\kappa\rho r$',fontsize=16)
for ax in (ax4,ax5,ax6): ax.set_xlabel('r (cm)',fontsize=16)
for ax in (ax1,ax2,ax3,ax4,ax5,ax6): 
    ax.set_xlim([rs,5e8])
#     lab = (r'$\phi<2$') if ax==ax2 else None
#     ax.axvspan(rbad,r[-1]*2,color='gray',alpha=0.5,label=lab)
#     ax.set_xlim([0.7*r[0],1.5*r[-1]])

ax1.loglog(r,T,'k-',lw=2)     , ax1.loglog(r2,T2,'k.--',lw=2)
ax2.semilogx(r,phi,'k-',lw=2)    , ax2.loglog(r2,phi2,'k.--',lw=2)
ax3.semilogx(r,lam)                 , ax3.semilogx(r2,lam2,'--')
ax4.loglog(r,rho)                   , ax4.loglog(r2,rho2,'.--')
ax5.loglog(r,L)                     , ax5.loglog(r2,L2,'--')
ax5.loglog(r,4*pi*r**2*sigmarad*T**4,color='r',label=r'$4\pi r^2\sigma T^4$')
ax5.loglog(r2,4*pi*r2**2*sigmarad*T2**4,color='r',linestyle='--')
ax5.axhline(LEdd,color='k')
ax6.loglog(r,tau)                   , ax6.loglog(r2,tau2,'--')
ax6.axhline(3,color='k',linestyle='--',linewidth=0.5)

# ax2.legend()
ax5.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])


fig,ax=plt.subplots(1,1)
ax.set_ylabel(r'Mach',fontsize=16)
ax.loglog(r,u/cs,'.-')
ax.loglog(r2,u2/csv2,'.-')
ax.axhline(1,color='k',linestyle='--')
ax.set_xlabel('r (cm)',fontsize=16)
ax.set_xlim([0.7*r[0],1.5*r[-1]])

# b1 = eos.Beta(rho,T)
# b2 = eos.Beta(rho2,T2)
# fig,ax=plt.subplots(1,1)
# ax.set_ylabel(r'$(4-3\beta)/(4-4\beta)$',fontsize=16)
# ax.semilogx(r,(4-3*b1)/(4-4*b1))
# ax.semilogx(r2,(4-3*b2)/(4-4*b2))
# ax.axhline(1,color='k',linestyle='--')
# ax.set_xlabel('r (cm)',fontsize=16)
# ax.set_xlim([0.7*r[0],1.5*r[-1]])



# ### to see what goes wrong, return all gradient terms

def dr2(inic, r, subsonic):

    T, phi = inic[:2]
    u, rho, phi, Lstar = calculateVars_phi(r, T, phi=phi, subsonic=subsonic)

    # dT
    if FLD:
        dlnT_dlnr_term1 = -Tstar(Lstar, T, r, rho, u) / (3*FLD_Lam(Lstar,r,u,T))
    else:
        dlnT_dlnr_term1 = -Tstar(Lstar, T, r, rho, u)

    dlnT_dlnr_term2 = - 1/Swz(r) * GM/c**2/r
    dlnT_dlnr = dlnT_dlnr_term1 + dlnT_dlnr_term2

    dT_dr = T/r * dlnT_dlnr

    # dPhi
    mach = u/sqrt(B(T))
    dphi_dr_term1 = (A(T)*mach**2-1)*(3*B(T)-2*A(T)*c**2)/(4*mach*A(T)**(3/2)*c**2*r) * dlnT_dlnr 
    dphi_dr_term2 = - numerator(r, T, u)/(u*r*sqrt(A(T)*B(T)))
    dphi_dr = (A(T)*mach**2-1)*(3*B(T)-2*A(T)*c**2)/(4*mach*A(T)**(3/2)*c**2*r) * dlnT_dlnr - numerator(r, T, u)/(u*r*sqrt(A(T)*B(T)))

    # dv
    dlnv_dlnr_num = numerator(r,T,u)
    dlnv_dlnr_denom = B(T)-u**2*A(T)
    if (r-rs)<1e6 or abs(r-rs2)<1e7:
        dlnv_dlnr=0
    else:
        dlnv_dlnr = dlnv_dlnr_num/dlnv_dlnr_denom
    
    # numerator of dv
    num_term1 = GM/r/Swz(r) * (A(T)-B(T)/c**2)
    num_term3 =  - 2*B(T)
    num_term2 = -C(Lstar, T, r, rho, u)            

    return [dlnT_dlnr_term1, dlnT_dlnr_term2, dlnT_dlnr, dT_dr, 
            dphi_dr_term1, dphi_dr_term2, dphi_dr,              
            dlnv_dlnr_num, dlnv_dlnr_denom, dlnv_dlnr,           
            num_term1, num_term2, num_term3]


nobjects=13
stuff = [[] for i in range(nobjects)]


# supersonic, subsonic = True, False
subsonic = False
for Ti,phii,ri in zip(T,phi,r):

    things = dr2([Ti,phii],ri,subsonic)
    for i in range(nobjects):
        stuff[i].append(things[i])

dlnT_dlnr_term1,dlnT_dlnr_term2,dlnT_dlnr,dT_dr, dphi_dr_term1, dphi_dr_term2,dphi_dr,dlnv_dlnr_num,dlnv_dlnr_denom,dlnv_dlnr,num_term1,num_term2,num_term3 = stuff

subsonic = True
stuff = [[] for i in range(nobjects)]
for Ti,phii,ri in zip(T2,phi2,r2):

    things = dr2([Ti,phii],ri,subsonic)
    for i in range(nobjects):
        stuff[i].append(things[i])

dlnT_dlnr_term1_2,dlnT_dlnr_term2_2,dlnT_dlnr_2,dT_dr_2, dphi_dr_term1_2, dphi_dr_term2_2,dphi_dr_2,dlnv_dlnr_num_2,dlnv_dlnr_denom_2,dlnv_dlnr_2,num_term1_2,num_term2_2,num_term3_2 = stuff





fig,((ax1,ax2,ax3),(ax4,ax5,ax6))=plt.subplots(2,3,figsize=(15,8))
fig.suptitle(r'Investigating $T$ and $\phi$ gradient terms ',fontsize=16)#,y=1.08)
ax1.set_ylabel(r'dlnT/dlnr term 1',fontsize=16)
ax2.set_ylabel(r'dlnT/dlnr term 2',fontsize=16)
ax3.set_ylabel(r'dlnT/dlnr',fontsize=16)
ax4.set_ylabel(r'd$\phi$/dr term 1',fontsize=16)
ax5.set_ylabel(r'd$\phi$/dr term 2',fontsize=16)
ax6.set_ylabel(r'd$\phi$/dr',fontsize=16)
for ax in (ax4,ax5,ax6): ax.set_xlabel('r (cm)',fontsize=16)
for ax in (ax1,ax2,ax3,ax4,ax5): 
    # ax.set_xlim([0.7*r[0],1.5*r[-1]])
    ax.set_xlim([rs,5e8])

ax1.semilogx(r,dlnT_dlnr_term1)             , ax1.semilogx(r2,dlnT_dlnr_term1_2,'.') 
ax2.semilogx(r,dlnT_dlnr_term2)             , ax2.semilogx(r2,dlnT_dlnr_term2_2) 
ax3.semilogx(r,dlnT_dlnr)                   , ax3.semilogx(r2,dlnT_dlnr_2)
# for ax in (ax1,ax2,ax3): ax.set_yscale('symlog')
ax4.semilogx(r,dphi_dr_term1)               , ax4.semilogx(r2,dphi_dr_term1_2)
ax5.semilogx(r,dphi_dr_term2)               , ax5.semilogx(r2,dphi_dr_term2_2)
ax6.semilogx(r,dphi_dr)                     , ax6.semilogx(r2,dphi_dr_2)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])




plt.tight_layout(rect=[0, 0.03, 1, 0.95])

fig,((ax1,ax2,ax3),(ax4,ax5,ax6))=plt.subplots(2,3,figsize=(15,8))
fig.suptitle(r'Investigating velocity gradient terms ',fontsize=16)
ax1.set_ylabel(r'dlnv/dlnr numerator',fontsize=16)
ax2.set_ylabel(r'dlnv/dlnr denominator',fontsize=16)
ax3.set_ylabel(r'dlnv/dlnr',fontsize=16)
ax4.set_ylabel(r'GM/r term',fontsize=16)
ax5.set_ylabel(r'C term',fontsize=16)
ax6.set_ylabel(r'2B term',fontsize=16)
for ax in (ax4,ax5,ax6): ax.set_xlabel('r (cm)',fontsize=16)
for ax in (ax1,ax2,ax3,ax4,ax5,ax6): 
    # ax.set_xlim([0.7*r[0],1.5*r[-1]])
    ax.set_xlim([rs,5e8])

ax1.semilogx(r,dlnv_dlnr_num)           , ax1.semilogx(r2,dlnv_dlnr_num_2,'.')
ax2.semilogx(r,dlnv_dlnr_denom)         , ax2.semilogx(r2,dlnv_dlnr_denom_2)
ax3.semilogx(r,dlnv_dlnr)               , ax3.semilogx(r2,dlnv_dlnr_2)
ax3.set_ylim([-2,2])

ax4.semilogx(r,num_term1)               , ax4.semilogx(r2,num_term1_2)
ax5.semilogx(r,-1*array(num_term2))     , ax5.semilogx(r2,-1*array(num_term2_2),'.')
ax6.semilogx(r,-1*array(num_term3))     , ax6.semilogx(r2,-1*array(num_term3_2))


plt.tight_layout(rect=[0, 0.03, 1, 0.95])



# GM/r - C
r,num_term1,num_term2=array(r),array(num_term1),array(num_term2)

fig,(ax1,ax2)=plt.subplots(2,1,sharex=True,figsize=(10,10))
fig.suptitle(r'GM/r-C')
ax1.axhline(0,color='k',linestyle='--')
ax1.semilogx(r,(GM/r+num_term2)/(GM/r),'b--', label=r'$1-1/\Psi\cdot 1/3\lambda\cdot L/L_E\cdot \kappa/\kappa_0\cdot (4-3\beta)/(4-4\beta)$')
ax1.semilogx(r,(num_term1+num_term2)/(GM/r),'b-',label=r'$1/\zeta^2 (A-B/c^2) - 1/\Psi\cdot 1/3\lambda\cdot L/L_E\cdot \kappa/\kappa_0\cdot (4-3\beta)/(4-4\beta)$')
ax1.legend()
ax2.axhline(1,color='k',linestyle='--')
ax2.semilogx(r,1/Y(r,u),label=r'$1/\Psi$')
ax2.semilogx(r,1/(3*FLD_Lam(Lstar,r,u,T)),label=r'$1/3\lambda$')
ax2.semilogx(r,L/LEdd,label=r'$L/L_E$')
ax2.semilogx(r,eos.kappa(rho,T)/eos.kappa0,label=r'$\kappa/\kappa_0$')
beta=eos.Beta(rho,T)
ax2.semilogx(r,(4-3*beta)/(4-4*beta),label=r'$(4-3\beta)/(4-4\beta)$')
ax2.semilogx(r, 1/Y(r,u) * 1/(3*FLD_Lam(Lstar,r,u,T)) * L/LEdd * eos.kappa(rho,T)/eos.kappa0 * (4-3*beta)/(4-4*beta) , 'k-',label='x')
ax2.legend()

plt.show()
