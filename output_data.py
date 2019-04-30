import os

def write_to_file(setup,logMdot,data):

    # data is expected to be list of the following arrays : R, T, Rho, u, Phi, Lstar, L, LEdd_loc, E, P, cs, tau

    if not os.path.exists('out'):   # Assuming code is being run from main directory 
        os.mkdir('out')

    filename = '_'.join([str(setup[i]) for i in range(len(setup))]) + '_' + str(logMdot)
    path = 'out/'+filename

    # if os.path.exists(path):
    #     c = input('File exists, proceed?')
    c=1
    if c:
        with open(path,'w') as f:
            f.write('r(km) \t u(km/s) cs(km/s) \t rho(g/cm3) \t T(K) \t P(dyne/cm2) \t phi \t L(erg/s) \t L*(erg/s) \t E(erg)')

            R, T, Rho, u, Phi, Lstar, L, E, P, cs, tau = data
            for i in range(len(R)):
                f.write('%0.3e \t %0.3e \t %0.3e \t %0.3e \t %0.3e \t %0.3e \t %0.3e \t %0.3e \t %0.3e \t %0.3e'%
                    (R[i]/1e5 , u[i]/1e5 , cs[i]/1e5 , Rho[i] , T[i] , P[i] , Phi[i] , L[i] , Lstar[i] , E[i]))

    
# write_to_file(['He',1.4,10,3,4],18,1)