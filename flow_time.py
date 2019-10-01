from IO import read_from_file
from scipy.interpolate import interp1d
from scipy.integrate import quad
from numpy import linspace,argwhere,array
import matplotlib.pyplot as plt 


logmdot = 18.0

R,V,cs,rho,T,P,phi,L,Lstar,E,tau,rs = read_from_file(logmdot) # r,u,cs are in km

# plt.figure()
# plt.plot(R*1e5,V*1e5,'k.-')
# plt.show()

# cut after 20km
V = V[R<20]
R = R[R<20]


fig,ax = plt.subplots(1,1)
ax.set_title('Flow time across wind')
ax.set_xlabel('r (km)')
ax.set_ylabel('Local flow time across 1 meter (s)')

ax2 = ax.twinx()
ax2.set_ylabel('Cumulative (s)',color='b')

t_local,t_cumul = [],[]

r,u = R*1e5,V*1e5
func_inverse_u = interp1d(r,1/u,kind='linear')
rlin = [r[0]]
dr = 1e2 #1 meter

while rlin[-1]<r[-1]-dr:
    rlin.append(rlin[-1]+dr)
    t_local.append( dr/2*(func_inverse_u(rlin[-2])+func_inverse_u(rlin[-1])) )   # trapeze area

for i in range(2,len(R)):
    r,u=R[:i]*1e5,V[:i]*1e5
    func_inverse_u = interp1d(r,1/u,kind='linear')
    tflow,err = quad(func_inverse_u,r[0],r[-1],epsrel=1e-5)
    t_cumul.append(tflow)

ax.plot(array(rlin[1:])/1e5,t_local,'k-')
ax2.semilogy(R[2:],t_cumul,'b-')
plt.tight_layout()
plt.show()