import sys
sys.path.append(".")

from numpy import argmin
from scipy.interpolate import interp1d
from scipy.integrate import quad

def flowtime(r,u,verbose=0): 
    '''Flow time of fluid across the wind, defined as integral of u^-1.
       Expecting consistent units for r and u! '''
    func_inverse_u = interp1d(r,1/u,kind='linear')
    tflow,err = quad(func_inverse_u,r[0],r[-1],epsrel=1e-5)
    if verbose:
        print("Integrated flow time : %.3e ; Integration error : %.3e"%(tflow,err))
    return tflow

def soundtime(r,cs,verbose=0): 
    '''Sound crossing time across the wind, defined as integral of cs^-1.
       Expecting consistent units for r and cs! '''
    func_inverse_cs = interp1d(r,1/cs,kind='linear')
    tsound,err = quad(func_inverse_cs,r[0],r[-1],epsrel=1e-5)
    if verbose:
        print("Integrated crossing time : %.3e ; Integration error : %.3e"%(tsound,err))
    return tsound

def soundtime2(r,cs,rs):
    '''Other sound crossing time, defined as rs/cs(rs)'''
    cs_at_rs = cs[argmin(abs(r-rs))]
    return rs/cs_at_rs

def soundflowtime(r,cs,u,rs,verbose=0):
    '''Sound crossing time until sonic point, then flow time until photosphere'''
    func_inverse_cs = interp1d(r,1/cs,kind='linear')
    func_inverse_u = interp1d(r,1/u,kind='linear')
    t1,err1 = quad(func_inverse_cs,r[0],rs,epsrel=1e-5)
    t2,err2 = quad(func_inverse_u,rs,r[-1],epsrel=1e-5)
    if verbose:
        print("Integrated sound crossing time until rs: %.3e ; Integration error : %.3e"%(t1,err1))
        print("Integrated flow time time until rphot: %.3e ; Integration error : %.3e"%(t2,err2))
    return t1+t2