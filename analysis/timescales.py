import sys
sys.path.append(".")

from numpy import argmin
from scipy.interpolate import interp1d
from scipy.integrate import quad

def flowtime(r,u,verbose=0,rphot=None): 
    '''Flow time of fluid across the wind, defined as integral of u^-1.
       Expecting consistent units for r and u! '''
    func_inverse_u = interp1d(r,1/u,kind='linear')
    if rphot==None:
        tflow,err = quad(func_inverse_u,r[0],r[-1],epsrel=1e-5)
    else:
        tflow,err = quad(func_inverse_u,r[0],rphot,epsrel=1e-5)
    if verbose:
        print("Integrated flow time : %.3e ; Integration error : %.3e"%(tflow,err))
    return tflow

def soundtime(r,cs,verbose=0,rphot=None): 
    '''Sound crossing time across the wind, defined as integral of cs^-1.
       Expecting consistent units for r and cs! '''
    func_inverse_cs = interp1d(r,1/cs,kind='linear')
    if rphot==None:
        tsound,err = quad(func_inverse_cs,r[0],r[-1],epsrel=1e-5)
    else:
        tsound,err = quad(func_inverse_cs,r[0],rphot,epsrel=1e-5)
    if verbose:
        print("Integrated crossing time : %.3e ; Integration error : %.3e"%(tsound,err))
    return tsound

def soundtime2(r,cs,rs):
    '''Other sound crossing time, defined as rs/cs(rs)'''
    cs_at_rs = cs[argmin(abs(r-rs))]
    return rs/cs_at_rs

def soundflowtime(r,cs,u,rs,verbose=0,rphot=None):
    '''Sound crossing time until sonic point, then flow time until photosphere'''
    func_inverse_cs = interp1d(r,1/cs,kind='linear')
    func_inverse_u = interp1d(r,1/u,kind='linear')
    t1,err1 = quad(func_inverse_cs,r[0],rs,epsrel=1e-5)
    if rphot==None:
        t2,err2 = quad(func_inverse_u,rs,r[-1],epsrel=1e-5)
    else:
        t2,err2 = quad(func_inverse_u,rs,rphot,epsrel=1e-5)
    if verbose:
        print("Integrated sound crossing time until rs: %.3e ; Integration error : %.3e"%(t1,err1))
        print("Integrated flow time time until rphot: %.3e ; Integration error : %.3e"%(t2,err2))
    return t1+t2