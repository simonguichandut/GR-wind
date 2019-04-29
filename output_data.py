import os

def write_to_file(setup,logMdot,data):

    # data is expected to be list of the following arrays : R, T, Rho, u, Phi, Lstar, L, LEdd_loc, E, P, cs, tau

    if not os.path.exists('out'):   # Assuming code is being run from main directory 
        os.mkdir('out')

    filename = '_'.join([str(setup[i]) for i in range(len(setup))]) + '_' + str(logMdot)

    with open('out/'+filename,'w') as f:
        f.write('r(km) \t u(km/s) cs(km/s) \t rho(g/cm3) \t T(K) \t P(dyne/cm2) \t phi \t L*(erg/s) \t L(erg/s) \t LEddloc(erg/s) \t E(erg)')

    
write_to_file(['He',1.4,10,3,4],18,1)