''' Input and Output '''

import os
from numpy import log10,array,pi,argmin

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
        FLD = eval(f.readline().split()[1]) # boolean
        next(f)
        next(f)
        mode = f.readline().split()[1]
        save = f.readline().split()[1]
        img = f.readline().split()[1]
        
    return M,R,y_inner,tau_out,comp,FLD,mode,save,img


def get_name():  # We give various files and directories the same name corresponding to the setup given in the parameter file

    M,R,y_inner,tau_out,comp,FLD,_,_,_ = load_params()
    name = '_'.join( [ comp , str(M) , ('%2d'%R) , ('%1d'%tau_out) , ('%1d'%log10(y_inner)) ] )
    if FLD: name += '_FLD'
    return name


def save_root(logMDOT,root):

    filename = get_name()
    path = 'roots/roots_' + filename + '.txt'

    if not os.path.exists(path):
        f = open(path,'w+')
        f.write('{:<7s} \t {:<11s} \t {:<11s}\n'.format(
            'logMdot' , 'Edot/LEdd' , 'log10(Ts)'))
    else:
        f = open(path,'a')

    f.write('{:<7.2f} \t {:<10f} \t {:<10f}\n'.format(
            logMDOT,root[0],root[1]))


def load_roots():

    filename = get_name()
    path = 'roots/roots_' + filename + '.txt'

    if not os.path.exists(path):
        raise TypeError('Root file does not exist')

    else:
        logMDOTS,roots = [],[]
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

    # data is expected to be list of the following arrays : R, T, Rho, u, Phi, Lstar, L, E, P, cs, tau, rs (rs is just a number)

    dirname = get_name()
    path = 'results/' + dirname + '/data/'
    filename = path + str(logMdot) + '.txt'

    R, T, Rho, u, Phi, Lstar, L, E, P, cs, tau, rs = data

    with open(filename,'w') as f:
        f.write('{:<13s} \t {:<11s} \t {:<11s} \t {:<11s} \t {:<11s} \t {:<11s} \t {:<11s} \t {:<11s} \t {:<11s} \t {:<11s} \t {:<11s}\n'.format(
            'r (cm)','u (cm/s)','cs (cm/s)','rho (g/cm3)','T (K)','P (dyne/cm2)','phi','L (erg/s)','L* (erg/s)','E (erg)','tau'))

        for i in range(len(R)):
            f.write('%0.8e \t %0.6e \t %0.6e \t %0.6e \t %0.6e \t %0.6e \t %0.6e \t %0.6e \t %0.6e \t %0.6e \t %0.6e'%
                (R[i] , u[i] , cs[i] , Rho[i] , T[i] , P[i] , Phi[i] , L[i] , Lstar[i] , E[i], tau[i]))

            if R[i]!=rs:
                f.write('\n')
            else:
                f.write('\t sonic point\n')


    # Flux Mdot file 
    Lbs = Lstar[0]
    Fbs = Lbs/(4*pi*R[0]**2)

    filename = path + 'Flux_Mdot.txt'
    if not os.path.exists(filename):
        f = open(filename,'w+')
        f.write('{:<7s} \t {:<11s} \t {:<11s}\n'.format(
            'logMdot' , 'Fbs' , 'Lbs'))
    else:
        f = open(filename,'a')

    f.write('{:<7.2f} \t {:<10e} \t {:<10e}\n'.format(
            logMdot,Fbs,Lbs))
                


def read_from_file(logMdot,specific_file=None):

    '''outputs arrays : R, u, cs, rho, T, P, phi, L, Lstar, E, tau and sonic point rs. '''

    if specific_file != None:
        filename = specific_file
    else:
        dirname = get_name()
        path = 'results/' + dirname + '/data/'
        filename = path + str(logMdot) + '.txt'

    def append_vars(line,varz,cols): # take line of file and append its values to variable lists 
        l=line.split()
        for var,col in zip(varz,cols):
            var.append(float(l[col]))

    R, u, cs, rho, T, P, phi, L, Lstar, E, tau = [[] for i in range (11)]
    with open(filename,'r') as f:
        next(f)
        for line in f: 
            append_vars(line,[R, u, cs, rho, T, P, phi, L, Lstar, E, tau],[i for i in range(11)])
            if line.split()[-1]=='point': rs = float(line.split()[0])

    return array(R),array(u),array(cs),array(rho),array(T),array(P),array(phi),array(L),array(Lstar),array(E),array(tau),rs
    # for copy-pasting : R,u,cs,rho,T,P,phi,L,Lstar,E,tau,rs


def save_plots(figs,fignames,img):

    dirname = get_name()
    path = 'results/' + dirname + '/plots/'

    for fig,figname in zip(figs,fignames):
        fig.savefig(path+figname+img)



def clean_rootfile(warning=1):

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
            
        if warning:
            o = input('Roots file will be overwritten. Proceed? (0 or 1) ')
        else:
            o = 1
        if o:
            filename = get_name()
            path = 'roots/roots_' + filename + '.txt'
            os.remove(path)

            for x,y in zip(new_logMDOTS,new_roots): 
                save_root(x,y)
            

# def pickle_save(name):
    
#     # Save all arrays into pickle file

#     # Import Winds
#     clean_rootfile()
#     logMDOTS,roots = load_roots()

#     if not os.path.exists('pickle/'):
#         os.mkdir('pickle/')

    