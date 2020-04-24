''' Input and Output '''

import os
import numpy as np

def load_params(as_dict=True):
    with open('params.txt','r') as f:
        next(f)
        M = float(f.readline().split()[1])
        R = float(f.readline().split()[1])
        next(f)
        next(f)
        y_inner = float(f.readline().split()[1])
        tau_out = float(f.readline().split()[1])
        comp = f.readline().split()[1]
        EOS_type = f.readline().split()[1]
        FLD = eval(f.readline().split()[1]) # boolean
        next(f)
        next(f)
        mode = f.readline().split()[1]
        save = f.readline().split()[1]
        img = f.readline().split()[1]
        
    if as_dict is True:
        return {'M':M,'R':R,'y_inner':y_inner,'tau_out':tau_out,
                'comp':comp,'EOS_type':EOS_type,'FLD':FLD,'mode':mode,
                'save':save,'img':img}
    else:
        return M,R,y_inner,tau_out,comp,EOS_type,FLD,mode,save,img


def get_name():  
    # We give various files and directories the same name corresponding 
    # to the setup given in the parameter file
    params = load_params()
    name = '_'.join([ 
        params['comp'], params['EOS_type'], ('M%.1f'%params['M']), 
        ('R%2d'%params['R']) , ('tau%1d'%params['tau_out']), 
        ('y%1d'%np.log10(params['y_inner'])) ])
    if params['FLD'] == True: name += '_FLD'
    return name


def save_root(logMDOT,root):

    filename = get_name()
    path = 'roots/roots_' + filename + '.txt'

    if not os.path.exists(path):
        f = open(path,'w+')
        f.write('{:<7s} \t {:<12s} \t {:<12s}\n'.format(
            'logMdot' , 'Edot/LEdd' , 'log10(Ts)'))
    else:
        f = open(path,'a')

    f.write('{:<7.2f} \t {:<11.8f} \t {:<11.8f}\n'.format(
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
    path = 'results/' + get_name()
    if not os.path.exists(path): # Assuming code is being run from main dir
        os.mkdir(path)
        os.mkdir(path+'/data')
        os.mkdir(path+'/plots')


def write_to_file(logMdot,wind):
    # Expecting wind type namedtuple object

    dirname = get_name()
    path = 'results/' + dirname + '/data/'
    filename = path + ('%.2f'%logMdot) + '.txt'

    with open(filename,'w') as f:

        # Write header
        f.write(('{:<11s} \t'*10).format(
            'r (cm)','T (K)','rho (g/cm3)','P (dyne/cm2)','u (cm/s)',
            'cs (cm/s)','phi','L (erg/s)','L* (erg/s)','taus'))

        if load_params()['FLD'] == True:
            f.write('{:<11s}'.format('lambda'))

        f.write('\n')

        # Write values
        for i in range(len(wind.r)):
            f.write(('%0.6e \t'*10)%(
                wind.r[i], wind.T[i], wind.rho[i], wind.P[i], wind.u[i],
                wind.cs[i], wind.phi[i], wind.L[i], wind.Lstar[i], wind.taus[i]))

            if load_params()['FLD'] == True:
                f.write('%0.6e \t'%wind.lam[i])

            if wind.r[i] == wind.rs:
                f.write('sonic point')

            f.write('\n')

    print('Wind data saved at: ',filename)

    # # Flux Mdot file 
    # Lbs = wind.Lstar[0]
    # Fbs = Lbs/(4*pi*wind.r[0]**2)

    # filename = path + 'Flux_Mdot.txt'
    # if not os.path.exists(filename):
    #     f = open(filename,'w+')
    #     f.write('{:<7s} \t {:<11s} \t {:<11s}\n'.format(
    #         'logMdot' , 'Fbs' , 'Lbs'))
    # else:
    #     f = open(filename,'a')

    # f.write('{:<7.2f} \t {:<10e} \t {:<10e}\n'.format(
    #         logMdot,Fbs,Lbs))
                


def read_from_file(logMdot,specific_file=None):

    '''outputs arrays from save file and rs '''

    if specific_file != None:
        filename = specific_file
    else:
        dirname = get_name()
        path = 'results/' + dirname + '/data/'
        filename = path + ('%.2f'%logMdot) + '.txt'

    def append_vars(line,varz): 
        l=line.split()
        for col,var in enumerate(varz):
            var.append(float(l[col]))
        

    r,T,rho,P,u,cs,phi,L,Lstar,taus,lam = [[] for i in range(11)]
    varz = [r,T,rho,P,u,cs,phi,L,Lstar,taus,lam]
    with open(filename, 'r') as f:
        next(f)
        for line in f:
            if load_params()['FLD'] == True:
                append_vars(line, varz)
            else:
                append_vars(line, varz)
            
            if line.split()[-1] == 'point': 
                rs = eval(line.split()[0])

    r,T,rho,P,u,cs,phi,L,Lstar,taus,lam = (np.array(var) for var in varz)

    # Return as wind tuple object
    if load_params()['FLD'] == True:
        from wind_GR_FLD import Wind
        return Wind(rs, r, T, rho, u, phi, Lstar, L, P, cs, taus, lam)
    else:
        from wind_GR import Wind
        return Wind(rs, r, T, rho, u, phi, Lstar, L, P, cs, taus)


def save_plots(figs,fignames,img):
    dirname = get_name()
    path = 'results/' + dirname + '/plots/'
    for fig,figname in zip(figs,fignames):
        fig.savefig(path+figname+img)
        

def clean_rootfile(warning=1):

    # Find duplicates, and remove all but the latest root 
    # (assuming the last one is the correct one)
    # Sort from lowest to biggest

    logMDOTS,roots = load_roots()
    new_logMDOTS = np.sort(np.unique(logMDOTS))

    if list(new_logMDOTS) != list(logMDOTS):

        v = []
        for x in new_logMDOTS:
            duplicates = np.argwhere(logMDOTS==x)
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


def save_EdotTsrel(logMdot, Edotvals, TsvalsA, TsvalsB):

    name = get_name()
    path = 'roots/FLD/' + name

    if not os.path.exists(path):
        os.mkdir(path)

    filepath = path + '/EdotTsrel_' + ('%.2f'%logMdot) + '.txt'
    if not os.path.exists(filepath):
        f = open(filepath, 'w+')
        f.write('{:<12s} \t {:<12s} \t {:<12s}\n'.format(
                'Edot/LEdd', 'log10(TsA)', 'log10(TsB)'))
    else:
        f = open(filepath, 'a')

    for edot, tsa, tsb in zip(Edotvals, TsvalsA, TsvalsB):
        f.write('{:<11.8f} \t {:<11.8f} \t {:<11.8f}\n'.format(
                edot, tsa, tsb))

def load_EdotTsrel(logMdot):

    filepath = 'roots/FLD/' + get_name() + '/EdotTsrel_' + ('%.2f'%logMdot) + '.txt'
    # filepath = 'roots/FLD/' + get_name() + '/EdotTsrel_' + str(logMdot) + '.txt'
    if not os.path.exists(filepath):
        return False,

    else:
        Edotvals, TsvalsA, TsvalsB = [],[],[]
        with open(filepath,'r') as f:
            next(f)
            for line in f:
                Edotvals.append(eval(line.split()[0]))
                TsvalsA.append(eval(line.split()[1]))
                TsvalsB.append(eval(line.split()[2]))
        
        return True,Edotvals,TsvalsA,TsvalsB


def clean_EdotTsrelfile(logMDOT,warning=1):

    # Find duplicates, and remove all but the latest root 
    # (assuming the last one is the correct one)
    # Sort from lowest to biggest
    from numpy import unique,sort,argwhere

    _,Edotvals,TsvalsA,TsvalsB = load_EdotTsrel(logMDOT)
    new_Edotvals = sort(unique(Edotvals))

    if list(new_Edotvals) != list(Edotvals):

        v = []
        for x in new_Edotvals:
            duplicates = argwhere(Edotvals==x)
            v.append(duplicates[-1][0]) # keeping the last one

        new_TsvalsA, new_TsvalsB = [],[]
        for i in v:
            new_TsvalsA.append(TsvalsA[i])
            new_TsvalsB.append(TsvalsB[i])

        if warning:
            o = input('EdotTsrel file will be overwritten. Proceed? (0 or 1) ')
        else:
            o = 1
        if o:
            filepath = 'roots/FLD/'+get_name()+'/EdotTsrel_'+str(logMDOT)+'.txt'
            os.remove(filepath)

            save_EdotTsrel(logMDOT,new_Edotvals,new_TsvalsA,new_TsvalsB)
