import os

def write_to_file(path,logMdot,data):

    # data is expected to be list of the following arrays : R, T, Rho, u, Phi, Lstar, L, LEdd_loc, E, P, cs, tau

    filename = path + '/' + str(logMdot) + '.txt'

    with open(filename,'w') as f:
        # f.write('%s \t %s \t %s \t %s \t %s \t %s \t %s \t %s \t %s \t %s\n'%
        #     ('r (km)','u (km/s)','cs (km/s)','rho (g/cm3)','T (K)','P (dyne/cm2)','phi','L (erg/s)','L* (erg/s)','E (erg)'))
        f.write('{:<8s} \t {:<8s} \t {:<8s} \t {:<8s} \t {:<8s} \t {:<8s} \t {:<8s} \t {:<8s} \t {:<8s} \t {:<8s}\n'.format(
            'r (km)','u (km/s)','cs (km/s)','rho (g/cm3)','T (K)','P (dyne/cm2)','phi','L (erg/s)','L* (erg/s)','E (erg)'))

        R, T, Rho, u, Phi, Lstar, L, E, P, cs, tau = data
        for i in range(len(R)):
            f.write('%0.3e \t %0.3e \t %0.3e \t %0.3e \t %0.3e \t %0.3e \t %0.3e \t %0.3e \t %0.3e \t %0.3e\n'%
                (R[i]/1e5 , u[i]/1e5 , cs[i]/1e5 , Rho[i] , T[i] , P[i] , Phi[i] , L[i] , Lstar[i] , E[i]))

    
# write_to_file(['He',1.4,10,3,4],18,1)