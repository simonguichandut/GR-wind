## With the now (correct ?) equation for C

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
# params=[1.025426   ,  7.196667]
# params=[1.025,7]
params=[1.025,7.2]
# params=[1.03   ,  7.4]

# logMdot = 19
# params=[1.020872   ,  6.958264]

# logMdot = 18
# params=[1.03,7.4]


if FLD:
    winds = MakeWind(params, logMdot, mode='wind',Verbose=1)
    w1 = winds[0]
    r,T,rho,u,phi,Lstar,L,LEdd_loc,E,P,cs,tau,lam,rs,Edot,Ts = w1
else:
    w1 = MakeWind(params, logMdot, mode='wind',Verbose=1)
    r,T,rho,u,phi,Lstar,L,LEdd_loc,E,P,cs,tau,lam,rs,Edot,Ts = w1



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
# for ax in (ax1,ax2,ax3,ax4,ax5,ax6): 
    # ax.set_xlim([rs,5e8])

ax1.loglog(r,T,'k-',lw=2)     
ax2.semilogx(r,phi,'k-',lw=2)    
ax3.semilogx(r,lam,'.')                 
ax4.loglog(r,rho)                   
ax5.loglog(r,L)                    
ax5.loglog(r,4*pi*r**2*sigmarad*T**4,color='r',label=r'$4\pi r^2\sigma T^4$')
ax5.axhline(LEdd,color='k',linestyle='--',linewidth=0.5)
ax6.loglog(r,tau)                
ax6.axhline(3,color='k',linestyle='--',linewidth=0.5)

# ax2.legend()
ax5.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,6))
ax1.set_ylabel(r'Mach',fontsize=16)
ax1.loglog(r,u/cs,'-')
ax1.axhline(1,color='k',linestyle='--')
ax1.set_xlabel('r (cm)',fontsize=16)

ax2.set_ylabel(r'u (cm/s)',fontsize=16)
ax2.loglog(r,u,'-')
ax2.loglog(r,cs,color='k',linestyle='--',label=r'$c_s$')
ax2.set_xlabel('r (cm)',fontsize=16)
ax2.legend()
plt.tight_layout()



#  return all gradient terms

def dr2(inic, r):

    T, phi = inic[:2]
    u, rho, phi, Lstar = calculateVars_phi(r, T, phi=phi, subsonic=False)
    L = Lcomoving(Lstar,r,u)

    # dT
    dlnT_dlnr_term1 = -Tstar(Lstar, T, r, rho, u) / (3*FLD_Lam(Lstar,r,u,T))
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
    if (r-rs)<1e6:
        dlnv_dlnr=0
    else:
        dlnv_dlnr = dlnv_dlnr_num/dlnv_dlnr_denom
    
    # numerator of dv
    num_term1 = GM/r/Swz(r) * (A(T)-B(T)/c**2)
    num_term3 =  - 2*B(T)
    num_term2 = -C(Lstar, T, r, rho, u)            



    # thick and thin limits
    # dlnT_dlnr_thick = -Tstar(Lstar, T, r, rho, u) - 1/Swz(r) * GM/c**2/r
    dlnT_dlnr_thick = -3*rho*eos.kappa(rho,T)*Lcomoving(Lstar,r,u) / (16*pi*r*arad*c*T**4*Y(r,u))
    dT_dr_thick = T/r * dlnT_dlnr_thick
    dT_dr_thin = -Lcomoving(Lstar,r,u)/(8*pi*r**3*arad*c*T**3)
    dlnT_dlnr_thin = r/T * dT_dr_thin

    Cthick = Tstar(Lstar, T, r, rho, u) * (4 - 3*eos.Beta(rho, T))/(1-eos.Beta(rho, T)) * arad*T**4/(3*rho) 
    taus = taustar(r,rho,T)
    Cthin =  1/Y(r,u) * L/LEdd * eos.kappa(rho,T)/eos.kappa0 * GM/r * ( 1 + Y(r,u)*eos.Beta(rho,T) / (6*taus*(1-eos.Beta(rho,T))) ) 

    dlnv_dlnr_num_thick = num_term1 + num_term3 - Cthick
    dlnv_dlnr_num_thin = num_term1 + num_term3 - Cthin
    
    # True thin equation has a modified A : 1+2.5 c_s^2/c^2 instead of 1+1.5 c_s^2/c^2
    Athin = 1 + 2.5*eos.cs2(T)/c**2
    dlnv_dlnr_num_thin_v2 = GM/r/Swz(r) * (Athin-B(T)/c**2) -2*B(T) - Cthin
    dlnv_dlnr_denom_thin_v2 = B(T)-u**2*Athin

    if (r-rs)<1e6:
        dlnv_dlnr_thick = 0
        dlnv_dlnr_thin = 0
        dlnv_dlnr_thin_v2 = 0
    else:
        dlnv_dlnr_thick = dlnv_dlnr_num_thick / dlnv_dlnr_denom
        dlnv_dlnr_thin = dlnv_dlnr_num_thin / dlnv_dlnr_denom
        dlnv_dlnr_thin_v2 = dlnv_dlnr_num_thin_v2 / dlnv_dlnr_denom_thin_v2



    return [dlnT_dlnr_term1, dlnT_dlnr_term2, dlnT_dlnr, dT_dr, 
            dphi_dr_term1, dphi_dr_term2, dphi_dr,              
            dlnv_dlnr_num, dlnv_dlnr_denom, dlnv_dlnr,           
            num_term1, num_term2, num_term3,
            dT_dr_thick, dT_dr_thin,
            dlnv_dlnr_thick, dlnv_dlnr_thin,dlnv_dlnr_thin_v2]


nobjects=18
stuff = [[] for i in range(nobjects)]


for Ti,phii,ri in zip(T,phi,r):

    things = dr2([Ti,phii],ri)
    for i in range(nobjects):
        stuff[i].append(things[i])

dlnT_dlnr_term1,dlnT_dlnr_term2,dlnT_dlnr,dT_dr, dphi_dr_term1, dphi_dr_term2,dphi_dr,dlnv_dlnr_num,dlnv_dlnr_denom,dlnv_dlnr,num_term1,num_term2,num_term3,dT_dr_thick,dT_dr_thin,dlnv_dlnr_thick,dlnv_dlnr_thin,dlnv_dlnr_thin_v2 = stuff


fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2,figsize=(15,8))
ax1.loglog(r,3*lam,'k-')
ax1.set_ylabel(r'$3\lambda$')
ax1b = ax1.twinx()
ax1b.loglog(r,taustar(r,rho,T),'b-')
ax1b.axhline(1,linestyle='--',color='b')
ax1b.set_ylabel(r'$\tau^*$',color='b')
ax1b.tick_params(axis='y', labelcolor='b')

ax2.loglog(r,np.abs(dT_dr),'k-')
ax2.loglog(r,np.abs(dT_dr_thick),ls='--',label='thick')
ax2.loglog(r,np.abs(dT_dr_thin),ls='--',label='thin')
ax2.set_ylabel(r'$-dT/dr$')
ax2.legend()
ax3.loglog(r,dlnv_dlnr,'k-')
ax3.loglog(r,dlnv_dlnr_thick,ls='--',label='thick')
# ax3.loglog(r,dlnv_dlnr_thin,ls='--',label='thin')
ax3.set_ylabel(r'$dlnv/dlnr$')
ax3.legend()

ax4.set_ylabel(r'$dlnv/dlnr$')
ax4.semilogx(r,dlnv_dlnr_thin,label='thin : normal A')
ax4.semilogx(r,dlnv_dlnr_thin_v2,label='thin: modified A')
ax4.legend()


plt.tight_layout()


## Checking if the flux calculated from cons of energy matches the flux from FLD formula
F = L/(4*pi*r**2)

from scipy.interpolate import InterpolatedUnivariateSpline as IUS  # has a derivative method
YY = Y(r,u)
Er = arad*T**4
alpha = F/(c*Er)
R = 1/(eos.kappa(rho,T)*rho*YY**3) * np.abs(IUS(r,YY**4*Er).derivative()(r)) / Er
F_fld = lam*c*R*Er


fig,ax = plt.subplots(1,1)
ax.set_xlabel(r'r (cm)')
ax.set_ylabel(r'F (erg s$^{-1}$ cm$^{-2}$')
ax.loglog(r,F,'k-',label='from conservation of energy')
ax.loglog(r,F_fld,'r--',label='from FLD formula')
ax.legend()
plt.tight_layout()


# alpha and Contributions to R
R_psiprime_part = 4/(eos.kappa(rho,T)*rho) * IUS(r,YY).derivative()(r)
R_Erprime_part = YY/(eos.kappa(rho,T)*rho) * IUS(r,Er).derivative()(r) / Er

fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,10))
ax1.set_ylabel('Flux and energy density')
ax1.loglog(r,F,'k',label='F')
ax1.loglog(r,c*Er,'k--',label='cE')
ax1.legend()
ax1b=ax1.twinx()
ax1b.tick_params(axis='y', labelcolor='b')
ax1b.set_ylabel(r'$1-\alpha$',color='b')
ax1b.loglog(r,1-alpha)

ax2.set_xlabel(r'r (cm)')
ax2.set_ylabel(r'R')
ax2.semilogx(r,R_psiprime_part,label=r'$\Psi\prime$ term')
ax2.semilogx(r,R_Erprime_part,label=r'$E_r\prime$ term')
ax2.semilogx(r,R,'k--',alpha=0.7,label='total')
ax2.semilogx(r,2*YY/tau,label=r'$\tau^*/2\Psi$')
ax2.set_yscale('symlog')
ax2.legend()
plt.tight_layout()



# fig,((ax1,ax2,ax3),(ax4,ax5,ax6))=plt.subplots(2,3,figsize=(15,8))
# fig.suptitle(r'Investigating $T$ and $\phi$ gradient terms ',fontsize=16)#,y=1.08)
# ax1.set_ylabel(r'dlnT/dlnr term 1',fontsize=16)
# ax2.set_ylabel(r'dlnT/dlnr term 2',fontsize=16)
# ax3.set_ylabel(r'dlnT/dlnr',fontsize=16)
# ax4.set_ylabel(r'd$\phi$/dr term 1',fontsize=16)
# ax5.set_ylabel(r'd$\phi$/dr term 2',fontsize=16)
# ax6.set_ylabel(r'd$\phi$/dr',fontsize=16)
# for ax in (ax4,ax5,ax6): ax.set_xlabel('r (cm)',fontsize=16)
# for ax in (ax1,ax2,ax3,ax4,ax5): 
#     # ax.set_xlim([0.7*r[0],1.5*r[-1]])
#     ax.set_xlim([rs,5e8])

# ax1.semilogx(r,dlnT_dlnr_term1)            
# ax2.semilogx(r,dlnT_dlnr_term2)           
# ax3.semilogx(r,dlnT_dlnr)                  
# # for ax in (ax1,ax2,ax3): ax.set_yscale('symlog')
# ax4.semilogx(r,dphi_dr_term1)             
# ax5.semilogx(r,dphi_dr_term2)              
# ax6.semilogx(r,dphi_dr)                   

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])




# plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# fig,((ax1,ax2,ax3),(ax4,ax5,ax6))=plt.subplots(2,3,figsize=(15,8))
# fig.suptitle(r'Investigating velocity gradient terms ',fontsize=16)
# ax1.set_ylabel(r'dlnv/dlnr numerator',fontsize=16)
# ax2.set_ylabel(r'dlnv/dlnr denominator',fontsize=16)
# ax3.set_ylabel(r'dlnv/dlnr',fontsize=16)
# ax4.set_ylabel(r'GM/r term',fontsize=16)
# ax5.set_ylabel(r'C term',fontsize=16)
# ax6.set_ylabel(r'2B term',fontsize=16)
# for ax in (ax4,ax5,ax6): ax.set_xlabel('r (cm)',fontsize=16)
# for ax in (ax1,ax2,ax3,ax4,ax5,ax6): 
#     # ax.set_xlim([0.7*r[0],1.5*r[-1]])
#     ax.set_xlim([rs,5e8])

# ax1.semilogx(r,dlnv_dlnr_num)           
# ax2.semilogx(r,dlnv_dlnr_denom)        
# ax3.semilogx(r,dlnv_dlnr)              
# ax3.set_ylim([-2,2])
# ax4.semilogx(r,num_term1)               
# ax5.semilogx(r,-1*array(num_term2))    
# ax6.semilogx(r,-1*array(num_term3))     


# plt.tight_layout(rect=[0, 0.03, 1, 0.95])



# # GM/r - C
# r,num_term1,num_term2=array(r),array(num_term1),array(num_term2)

# fig,(ax1,ax2)=plt.subplots(2,1,sharex=True,figsize=(10,10))
# fig.suptitle(r'GM/r-C')
# ax1.axhline(0,color='k',linestyle='--')
# ax1.semilogx(r,(GM/r+num_term2)/(GM/r),'b--', label=r'$1-1/\Psi\cdot 1/3\lambda\cdot L/L_E\cdot \kappa/\kappa_0\cdot (4-3\beta)/(4-4\beta)$')
# ax1.semilogx(r,(num_term1+num_term2)/(GM/r),'b-',label=r'$1/\zeta^2 (A-B/c^2) - 1/\Psi\cdot 1/3\lambda\cdot L/L_E\cdot \kappa/\kappa_0\cdot (4-3\beta)/(4-4\beta)$')
# ax1.legend()
# ax2.axhline(1,color='k',linestyle='--')
# ax2.semilogx(r,1/Y(r,u),label=r'$1/\Psi$')
# ax2.semilogx(r,1/(3*FLD_Lam(Lstar,r,u,T)),label=r'$1/3\lambda$')
# ax2.semilogx(r,L/LEdd,label=r'$L/L_E$')
# ax2.semilogx(r,eos.kappa(rho,T)/eos.kappa0,label=r'$\kappa/\kappa_0$')
# beta=eos.Beta(rho,T)
# ax2.semilogx(r,(4-3*beta)/(4-4*beta),label=r'$(4-3\beta)/(4-4\beta)$')
# ax2.semilogx(r, 1/Y(r,u) * 1/(3*FLD_Lam(Lstar,r,u,T)) * L/LEdd * eos.kappa(rho,T)/eos.kappa0 * (4-3*beta)/(4-4*beta) , 'k-',label='x')
# ax2.legend()

plt.show()
