import sys
sys.path.append(".")

import matplotlib.pyplot as plt

from wind_GR_FLD import *
import IO 

logMdots,_ = IO.load_roots()

fig,ax = plt.subplots(1,1)
ax.set_xlabel(r'log$\dot{M}$')
ax.set_ylabel(r'$\beta(r=R)$')

beta_base = []
logMdots2 = []
for logMdot in logMdots:
    try:
        w = IO.read_from_file(logMdot)
        beta_base.append(eos.Beta(w.rho[0],w.T[0],lam=1/3,R=0))
        logMdots2.append(logMdot)
    except:
        pass

ax.semilogy(logMdots2,beta_base,'k.-')
fig.savefig('analysis/base_beta.png',bbox_inches="tight",dpi=300)