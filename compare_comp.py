import matplotlib.pyplot as plt
import numpy as np 

#import wind values
vals_He = np.loadtxt('wind_values_He_IGDE_M1.4_R12_tau3_y8.txt',skiprows=1)
vals_H = np.loadtxt('wind_values_H_IGDE_M1.4_R12_tau3_y8.txt',skiprows=1)

logmdot_He,Lbs_He = vals_He.transpose()[0], vals_He.transpose()[6]
logmdot_H,Lbs_H = vals_H.transpose()[0], vals_H.transpose()[6]

plt.figure()
plt.loglog(Lbs_He,10**logmdot_He,label='He')
plt.loglog(Lbs_H,10**logmdot_H,label='H')
plt.xlabel(r'$L_b^{\infty}$ (erg/s)')
plt.ylabel(r'$\dot{M}$ (g/s)')
plt.legend()
plt.show()