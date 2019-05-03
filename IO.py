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






logMDOTS = [18.0, 18.25, 18.5, 18.75, 19.0, 18.5, 19.0, 18.0, 17.3, 19.0, 18.8, 18.7, 18.6, 18.5, 18.4, 18.3, 18.2, 18.1, 18.0]
roots = [[1.025, 7.35], [1.02, 7.3], [1.02, 7.2], [1.02, 7.1], [1.02, 6.96], [1.025087, 7.192502], [1.020776, 6.959367], [1.024887, 7.321983], [1.0177, 7.213938], [1.020776, 6.959367], [1.022514, 7.053857], [1.023451, 7.102189], [1.024345, 7.149243], [1.025088, 7.192513], [1.025601, 7.230439], [1.02585, 7.262548], [1.025816, 7.288609], [1.025484, 7.308488], [1.024887, 7.321963]]


def clean_rootfile():

    # Find duplicates, and remove all but the latest root (assuming the last one is the correct one)
    # Sort from lowest to biggest
    from numpy import unique,sort,argwhere

    # logMDOTS,roots = load_roots()

    new_logMDOTS = sort(unique(logMDOTS))

    v = []
    for x in new_logMDOTS:
        duplicates= argwhere(logMDOTS==x)
        v.append(duplicates[-1][0]) # keeping the last one

    new_roots = []
    for i in v:
        new_roots.append(roots[i])
        
    o = input('Old file will be overwritten. Proceed? (0 or 1')
    if o:
        filename = get_name()
        path = 'wind_solutions/sols_' + filename + '.txt'
        os.remove(path)
        
        for x,y in zip(new_logMDOTS,new_roots): 
            save_root(x,y)
        



# clean_rootfile()