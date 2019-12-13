# First and second derivatives and jacobian

import physics
import IO

arad = 7.5657e-15
c = 2.99792458e10
sigmarad = 0.25*arad*c
M, RNS, y_inner, tau_out, comp, EOS_type, FLD, mode, save, img = IO.load_params()
eos = physics.EOS(comp)
GM = 6.6726e-8*2e33*M

def gamma(v):
    return (1-v**2/c**2)**(-1/2)

def Gamma(r):
    return (1-2*GM/c**2/r)**(1/2)

def Psi(r,v):
    return gamma(v)*Gamma(r)

def derivs(r,phi,T,subsonic):

    pass



        

