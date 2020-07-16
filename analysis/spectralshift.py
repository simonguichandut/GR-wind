import sys
sys.path.append('.')  

import numpy as np
import matplotlib.pyplot as plt
from IO import read_from_file,load_params

M, RNS, y_inner, tau_out, comp, mode, save, img = load_params()
c = 2.99292458e10
GM = 6.6726e-8*2e33*M



w=read_from_file(18.0)

redshift=(1-2*GM/c**2/w.r)**(-0.5)-1
blueshift=np.sqrt((1-w.u/c)/(1+w.u/c))-1

fig,ax = plt.subplots(1,1)
ax.set_xlabel('r (km)',fontsize=14)
ax.set_ylabel(r'$z=\Delta\lambda/\lambda$',fontsize=14)
ax.plot(w.r/1e5,redshift,'r-',label=r'$1+z=\sqrt{1-2GM/c^2r}$')
ax.plot(w.r/1e5,blueshift,'b-',label=r'$1+z=\gamma(1-v/c)$')
ax.plot(w.r/1e5,redshift+blueshift,'k',label='Total')
ax.legend()

plt.tight_layout()
plt.show()

