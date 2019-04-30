k = 1.38e-16
mp = 1.67e-24
from math import pi,sqrt

def andrewP(rho, T):
  rY = rho*0.5
  pednr = 9.91e-2*rY**(5/3)
  pedr=1.231e1*rY**(4/3)
  ped=1/sqrt((1/pedr**2)+(1/pednr**2)) 
  pend=8.254e-7*rY*T                    # *1e8 in Andrew's eos.cc because T8 is T/1e8
  return 1e14*sqrt(ped**(2) + pend**2)

def simonP(rho, T):
  rY = rho*0.5
  pednr = 9.91e12*rY**(5/3)
  pedr=1.231e15*rY**(4/3)
  ped=1/sqrt((1/pedr**2)+(1/pednr**2)) 
  pend=8.254e-7*rY*T
  return sqrt(ped**2 + pend**2)

print(andrewP(1e10,1e8))
print(simonP(1e10,1e8))

# same 