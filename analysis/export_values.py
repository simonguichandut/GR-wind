import sys
sys.path.append(".")

from numpy import log10,array,pi,argmin
from IO import load_roots,load_params,read_from_file

target = "."

# Export useful values for analysis for each Mdot to a text file at target directory
# Current values are : Rb,Tb,Rhob,Pb,Lb,Lb*,Rphot,Tphot,Rhophot,Lphot,Lphot*,rs,
# tsound (sound crossing time), tsound2 (rs/cs(rs)), tflow (flow crossing time), Tau : a specific timescale, currently : sound crossing time until v=1e6, then flow crossing time
# Min&Mout (masses below & above sonic point)

if target[-1]!='/': target += '/'
logMDOTS,_ = load_roots()
M,R,y_inner,tau_out,comp,mode,save,img = load_params()

from scipy.interpolate import interp1d
from scipy.integrate import quad

with open(target+'wind_values_'+comp+'.txt','w') as f:

    f.write(('{:<11s} \t '*16+'{:<11s}\n').format(
        'logMdot (g/s)','rb (cm)','Tb (K)','rhob (g/cm3)','Pb (dyne/cm2)','Lb (erg/s)','Lb* (erg/s)','Rph (cm)','Tph (K)','rhoph (g/cm3)','Lph (erg/s)','Lph* (erg/s)','rs (cm)','tflow (s)','tsound (s)','tsound2 (s)','Tau (s)'))
    # f.write('{:<11s} \t {:<11s} \t {:<11s} \t {:<11s} \t {:<11s} \t {:<11s} \t {:<11s} \t {:<11s} \t {:<11s} \t {:<11s} \t {:<11s} \t {:<11s} \t {:<11s} \t {:<11s} \t {:<11s} \t {:<11s} \t {:<11s}\n'.format(
    #     'logMdot (g/s)','rb (cm)','Tb (K)','rhob (g/cm3)','Pb (dyne/cm2)','Lb (erg/s)','Lb* (erg/s)','Rph (cm)','Tph (K)','rhoph (g/cm3)','Lph (erg/s)','Lph* (erg/s)','rs (cm)','tflow (s)','tsound (s)','tsound2 (s)','Tau(s)'))

    for x in logMDOTS:
        r,u,cs,rho,T,P,phi,L,Lstar,E,tau,rs = read_from_file(x) # r,u,cs are in km

        func_inverse_u = interp1d(r,1/u,kind='linear')
        tflow,err = quad(func_inverse_u,r[0],r[-1],epsrel=1e-5)
        # print(tflow,err)

        func_inverse_cs = interp1d(r,1/cs,kind='cubic')
        tsound,err = quad(func_inverse_cs,r[0],r[-1],epsrel=1e-5)
        # print(tsound,err)

        # index_rs = argmin(abs(r-rs/1e5))

        tsound2 = rs/1e5/cs[argmin(abs(r-rs/1e5))]
        # print(tsound2,'\n\n')

        # Tau : sound crossing time until v=1e6, then flow crossing time
        Tau1,err = quad(func_inverse_cs,r[0],rs/1e5,epsrel=1e-5)
        Tau2,err = quad(func_inverse_u,rs/1e5,r[-1],epsrel=1e-5)
        Tau = Tau1+Tau2


        # Mass above and below sonic point
        rhofunc = interp1d(r*1e5,rho,kind='cubic')

        def mass_in_shell(r):
            return 4*pi*rhofunc(r)*r**2

        Min,err1 = quad(mass_in_shell, r[0]*1e5 , rs, epsrel=1e-5)
        Mout,err2 = quad(mass_in_shell, rs , r[-1]*1e5, epsrel=1e-5)
        # print(Min/Mout)


        f.write(('%0.2f \t\t '+'%0.6e \t '*15 + '%0.6e\n')%
            (x,r[0]*1e5,T[0],rho[0],P[0],L[0],Lstar[0],r[-1]*1e5,T[-1],rho[-1],L[-1],Lstar[-1],rs,tflow,tsound,tsound2,Tau))
        # f.write('%0.2f \t\t %0.6e \t %0.6e \t %0.6e \t %0.6e \t %0.6e \t %0.6e \t %0.6e \t %0.6e \t %0.6e \t %0.6e \t %0.6e \t %0.6e \t %0.6e \t %0.6e \t %0.6e \t %0.6e \n'%
        #     (x,r[0]*1e5,T[0],rho[0],P[0],L[0],Lstar[0],r[-1]*1e5,T[-1],rho[-1],L[-1],Lstar[-1],rs,tflow,tsound,tsound2,Tau))
