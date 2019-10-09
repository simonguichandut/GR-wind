import sys
sys.path.append(".")
sys.path.append("./analysis")

from numpy import array,pi
from IO import load_roots,load_params,read_from_file
from timescales import *

target = "."

# Export useful values for analysis for each Mdot to a text file at target directory
# Current values are : Tb,Rhob,Pb,Lb,Lb*,Rphot,Tphot,Rhophot,Lphot,Lphot*,rs,
# tsound (sound crossing time), tsound2 (rs/cs(rs)), tflow (flow crossing time), Tau : a specific timescale, currently : sound crossing time until v=1e6, then flow crossing time
# Min&Mout (masses below & above sonic point)

if target[-1]!='/': target += '/'
logMDOTS,_ = load_roots()
M,R,y_inner,tau_out,comp,mode,save,img = load_params()

from scipy.interpolate import interp1d
from scipy.integrate import quad

with open(target+'wind_values_'+comp+'.txt','w') as f:

    f.write(('{:<11s} \t '*15+'{:<11s}\n').format(
        'logMdot (g/s)','Tb (K)','rhob (g/cm3)','Pb (dyne/cm2)','Lb (erg/s)','Lb* (erg/s)','Rph (cm)','Tph (K)','rhoph (g/cm3)','Lph (erg/s)','Lph* (erg/s)','rs (cm)','tflow (s)','tsound (s)','tsound2 (s)','Tau (s)'))


    for x in logMDOTS:
        r,u,cs,rho,T,P,phi,L,Lstar,E,tau,rs = read_from_file(x) # r,u,cs are in km

        r,u,cs = r*1e5,u*1e5,cs*1e5

        tflow,tsound,tsound2,Tau = flowtime(r,u),soundtime(r,u),soundtime2(r,cs,rs),soundflowtime(r,cs,u,rs)

        # Mass above and below sonic point
        rhofunc = interp1d(r,rho,kind='cubic')

        def mass_in_shell(r): 
            return 4*pi*rhofunc(r)*r**2

        Min,err1 = quad(mass_in_shell, r[0] , rs, epsrel=1e-5)
        Mout,err2 = quad(mass_in_shell, rs , r[-1], epsrel=1e-5)
        # print(Min/Mout)

        f.write(('%0.2f \t\t '+'%0.6e \t '*14 + '%0.6e\n')%
            (x,r[0]*1e5,T[0],P[0],L[0],Lstar[0],r[-1]*1e5,T[-1],rho[-1],L[-1],Lstar[-1],rs,tflow,tsound,tsound2,Tau))
    
