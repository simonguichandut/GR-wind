''' Input and Output '''

import os
from numpy import log10

def load_params():
    with open('params.txt','r') as f:
        next(f)
        M = float(f.readline().split()[1])
        R = float(f.readline().split()[1])
        next(f)
        next(f)
        y_inner = float(f.readline().split()[1])
        tau_out = float(f.readline().split()[1])
        comp = f.readline().split()[1]
        next(f)
        next(f)
        mode = f.readline().split()[1]
        save = f.readline().split()[1]
        img = f.readline().split()[1]
        
    return M,R,y_inner,tau_out,comp,mode,save,img


def get_name():  # We give various files and directories the same name corresponding to the setup given in the parameter file

    M,R,y_inner,tau_out,comp,_,_,_ = load_params()
    name = '_'.join( [ comp , str(M) , ('%2d'%R) , ('%1d'%tau_out) , ('%1d'%log10(y_inner)) ] )
    return name


def save_root(logMDOT,root):

    filename = get_name()
    path = 'wind_solutions/sols_' + filename + '.txt'

    if not os.path.exists(path):
        f = open(path,'w+')
        f.write('{:<7s} \t {:<10s} \t {:<10s}\n'.format(
            'logMdot' , 'Edot/LEdd' , 'log10(Ts)'))
    else:
        f = open(path,'a')

    f.write('{:<7.2f} \t {:<10f} \t {:<10f}\n'.format(
            logMDOT,root[0],root[1]))


def load_roots():

    filename = get_name()
    path = 'wind_solutions/sols_' + filename + '.txt'

    logMDOTS = []
    roots = []
    with open(path, 'r') as f:
        next(f)
        for line in f:
            stuff = line.split()
            logMDOTS.append(float(stuff[0]))
            roots.append([float(stuff[1]), float(stuff[2])])

    return logMDOTS,roots



def make_directories():

    dirname = get_name()
    path = 'results/' + dirname
    if not os.path.exists(path):   # Assuming code is being run from main directory
        os.mkdir(path)
        os.mkdir(path+'/data')
        os.mkdir(path+'/plots')



def write_to_file(logMdot,data):

    # data is expected to be list of the following arrays : R, T, Rho, u, Phi, Lstar, L, LEdd_loc, E, P, cs, tau

    dirname = get_name()
    path = 'results/' + dirname + '/data/'
    filename = path + str(logMdot) + '.txt'

    with open(filename,'w') as f:
        # f.write('%s \t %s \t %s \t %s \t %s \t %s \t %s \t %s \t %s \t %s\n'%
        #     ('r (km)','u (km/s)','cs (km/s)','rho (g/cm3)','T (K)','P (dyne/cm2)','phi','L (erg/s)','L* (erg/s)','E (erg)'))
        f.write('{:<8s} \t {:<8s} \t {:<8s} \t {:<8s} \t {:<8s} \t {:<8s} \t {:<8s} \t {:<8s} \t {:<8s} \t {:<8s}\n'.format(
            'r (km)','u (km/s)','cs (km/s)','rho (g/cm3)','T (K)','P (dyne/cm2)','phi','L (erg/s)','L* (erg/s)','E (erg)'))

        R, T, Rho, u, Phi, Lstar, L, E, P, cs, tau = data
        for i in range(len(R)):
            f.write('%0.3e \t %0.3e \t %0.3e \t %0.3e \t %0.3e \t %0.3e \t %0.3e \t %0.3e \t %0.3e \t %0.3e\n'%
                (R[i]/1e5 , u[i]/1e5 , cs[i]/1e5 , Rho[i] , T[i] , P[i] , Phi[i] , L[i] , Lstar[i] , E[i]))

    


def save_plots(figs,fignames,img):

    dirname = get_name()
    path = 'results/' + dirname + '/plots/'

    for fig,figname in zip(figs,fignames):
        fig.savefig(path+figname+img)



def clean_rootfile():

    # Find duplicates, and remove all but the latest root (assuming the last one is the correct one)
    # Sort from lowest to biggest
    from numpy import unique,sort,argwhere

    logMDOTS,roots = load_roots()
    new_logMDOTS = sort(unique(logMDOTS))

    if list(new_logMDOTS) != list(logMDOTS):

        v = []
        for x in new_logMDOTS:
            duplicates= argwhere(logMDOTS==x)
            v.append(duplicates[-1][0]) # keeping the last one

        new_roots = []
        for i in v:
            new_roots.append(roots[i])
            
        o = input('Old file will be overwritten. Proceed? (0 or 1) ')
        if o:
            filename = get_name()
            path = 'wind_solutions/sols_' + filename + '.txt'
            os.remove(path)

            for x,y in zip(new_logMDOTS,new_roots): 
                save_root(x,y)
            

# clean_rootfile()