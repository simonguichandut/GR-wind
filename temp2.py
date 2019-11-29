from wind_GR import *
# print(FLD)
logMdot = 18.5
params=[1.025426   ,  7.196667]

# global Mdot, Edot, rs, verbose
Mdot, Edot, Ts, verbose = 10**logMdot, params[0]*LEdd, 10**params[1], 1

rs = rSonic2(Ts,Mdot,Edot)
print(rs)


# 13157581.390192693
