''' Input and Output '''

import os
import numpy as np
import sys
#sys.path.append(".") 
## the above is so that the functions in this script work even when imported from somewhere else
#local_path = os.path.realpath('IO.py')
#print(local_path)
#print(sys.argv[0])
#
def load_params(as_dict=True):
    with open('./params.txt','r') as f:
        next(f)
        M = float(f.readline().split()[1])
        R = float(f.readline().split()[1])
        next(f)
        next(f)
        y_inner = float(f.readline().split()[1])
        comp = f.readline().split()[1]
        EOS_type = f.readline().split()[1]
        tau_out = float(f.readline().split()[1])
        FLD = eval(f.readline().split()[1]) # boolean
        Prad = f.readline().split()[1]
        
    if as_dict is True:
        return {'M':M,'R':R,'y_inner':y_inner,'tau_out':tau_out,
                'comp':comp,'EOS_type':EOS_type,'FLD':FLD,'Prad':Prad}
    else:
        return M,R,y_inner,tau_out,comp,EOS_type,FLD,Prad


def get_name():  
    # We give various files and directories the same name corresponding 
    # to the setup given in the parameter file
    params = load_params()

    name = '_'.join([ 
        params['comp'], params['EOS_type'], ('M%.1f'%params['M']), 
        ('R%2d'%params['R']) , ('y%1d'%np.log10(params['y_inner'])) ])
        
    if params['FLD'] == True: 
        name += '_FLD'
        if params['Prad'] == 'exact':
            name += '_exact'

    else:
        name += ('tau%1d'%params['tau_out'])

    return name


def save_root(logMdot,root,decimals=8):

    filename = get_name()
    path = 'roots/roots_' + filename + '.txt'

    if not os.path.exists(path):
        f = open(path,'w+')
        f.write('{:<7s} \t {:<12s} \t {:<12s}\n'.format(
            'logMdot' , 'Edot/LEdd' , 'log10(Ts)'))
    else:
        f = open(path,'a')

    if decimals == 8:
        f.write('{:<7.2f} \t {:<11.8f} \t {:<11.8f}\n'.format(
                logMdot,root[0],root[1]))
    elif decimals == 9:
        f.write('{:<7.2f} \t {:<12.9f} \t {:<12.9f}\n'.format(
                logMdot,root[0],root[1]))
    elif decimals == 10:
        f.write('{:<7.2f} \t {:<13.10f} \t {:<13.10f}\n'.format(
                logMdot,root[0],root[1]))


def load_roots(logMdot=None,specific_file=None):

    if specific_file != None:
        filename = specific_file
    else:
        name = get_name()
        path = 'roots/roots_'
        filename = path + name + '.txt'

    if not os.path.exists(filename):
        raise TypeError('Root file does not exist (%s)'%filename)

    else:
        logMdots,roots = [],[]
        with open(filename, 'r') as f:
            next(f)
            for line in f:
                stuff = line.split()
                logMdots.append(float(stuff[0]))
                roots.append([float(stuff[1]), float(stuff[2])])

        if logMdot is None:
            return logMdots,roots
        else:
            return roots[logMdots.index(logMdot)]


def make_directories():
    path = 'results/' + get_name()
    if not os.path.exists(path): # Assuming code is being run from main dir
        os.mkdir(path)
        os.mkdir(path+'/data')
        # os.mkdir(path+'/plots')


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


def read_from_file(logMdot, specific_file=None):

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
                append_vars(line, varz[:-1])
            
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


# def save_plots(figs,fignames,img):
#     dirname = get_name()
#     path = 'results/' + dirname + '/plots/'
#     for fig,figname in zip(figs,fignames):
#         fig.savefig(path+figname+img)
        

def clean_rootfile(warning=1):

    # Find duplicates, and remove all but the latest root 
    # (assuming the last one is the correct one)
    # Sort from lowest to biggest

    logMdots,roots = load_roots()
    new_logMdots = np.sort(np.unique(logMdots))

    if list(new_logMdots) != list(logMdots):

        v = []
        for x in new_logMdots:
            duplicates = np.argwhere(logMdots==x)
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

            for x,y in zip(new_logMdots,new_roots): 
                decimals = max((len(str(y[0])),len(str(y[1])))) - 2   
                # print(decimals)             
                save_root(x,y,decimals=decimals)
                        
            
# def pickle_save(name):
    
#     # Save all arrays into pickle file

#     # Import Winds
#     clean_rootfile()
#     logMdotS,roots = load_roots()

#     if not os.path.exists('pickle/'):
#         os.mkdir('pickle/')


def save_EdotTsrel(logMdot, Edotvals, TsvalsA, TsvalsB, decimals=8):

    name = get_name()
    if 'FLD' not in name:
        raise
    else:
        i = name.find('FLD')
        name = name[:i-1] + name[i+3:] # removing the FLD for brevity, it's going into a dir called FLD/ anyway

    i = name.find('y')
    name = name[:i-1] + name[i+2:] # the inner column depth parameter is irrelevant for the EdotTsrelation

    path = 'roots/FLD/' + name

    if not os.path.exists(path):
        os.mkdir(path)

    filepath = path + '/EdotTsrel_' + ('%.2f'%logMdot) + '.txt'
    if not os.path.exists(filepath):
        f = open(filepath, 'w+')
        f.write('{:<12s} \t {:<12s} \t {:<12s}\n'.format(
                'Edot/LEdd', 'log10(TsA)', 'log10(TsB)'))
        # f.write('{:<13s} \t {:<13s} \t {:<13s}\n'.format(
                # 'Edot/LEdd', 'log10(TsA)', 'log10(TsB)'))

    else:
        f = open(filepath, 'a')

    for edot, tsa, tsb in zip(Edotvals, TsvalsA, TsvalsB):
        if decimals == 8:
            f.write('{:<11.8f} \t {:<11.8f} \t {:<11.8f}\n'.format(
                    edot, tsa, tsb))
        elif decimals == 10:
            f.write('{:<13.10f} \t {:<13.10f} \t {:<13.10f}\n'.format(
                    edot, tsa, tsb))

        # s = ('{:<%d.%df}'%(decimals+3,decimals))
        # f.write(s.format(edot) + '\t' + s.format(tsa) + '\t' + s.format(tsb) + '\n')

def load_EdotTsrel(logMdot, specific_file=None):

    if specific_file is not None:
        filepath = specific_file
    else:
        name = get_name()
        i = name.find('FLD')
        name = name[:i-1] + name[i+3:]
        i = name.find('y')
        name = name[:i-1] + name[i+2:]

        filepath = 'roots/FLD/' + name + '/EdotTsrel_' + ('%.2f'%logMdot) + '.txt'
        if not os.path.exists(filepath):
            return False,

    Edotvals, TsvalsA, TsvalsB = [],[],[]
    with open(filepath,'r') as f:
        next(f)
        for line in f:
            Edotvals.append(eval(line.split()[0]))
            TsvalsA.append(eval(line.split()[1]))
            TsvalsB.append(eval(line.split()[2]))
    
    return True,Edotvals,TsvalsA,TsvalsB
    # note that Edotvals is (Edot-Mdotc^2)/LEdd, Tsvals is logTs


def clean_EdotTsrelfile(logMdot,warning=1):

    # Find duplicates, and remove all but the latest root 
    # (assuming the last one is the correct one)
    # Sort from lowest to biggest

    _,Edotvals,TsvalsA,TsvalsB = load_EdotTsrel(logMdot)
    new_Edotvals = np.sort(np.unique(Edotvals))

    # if list(new_Edotvals) != list(Edotvals):
    if True:

        v = []
        for x in new_Edotvals:
            duplicates = np.argwhere(Edotvals==x)
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

            name = get_name()
            i = name.find('FLD')
            name = name[:i-1] + name[i+3:]
            i = name.find('y')
            name = name[:i-1] + name[i+2:]

            filepath = 'roots/FLD/' + name + '/EdotTsrel_'+('%.2f'%logMdot)+'.txt'
            os.remove(filepath)

            # save_EdotTsrel(logMdot,new_Edotvals,new_TsvalsA,new_TsvalsB)
            for e,tsa,tsb in zip(new_Edotvals,new_TsvalsA,new_TsvalsB):
                decimals = max((len(str(e)),len(str(tsa)),len(str(tsb)))) - 2
                # print(decimals)
                save_EdotTsrel(logMdot,[e],[tsa],[tsb],decimals=decimals)
            





def info(logMdot, returnit=False):
    """ Print (or returns) True/False for if root exists, if EdotTsrel file exists
    (if FLD==True) and if datafile exists"""

    # Root
    logMdots,_ = load_roots()
    root_exists = (logMdot in logMdots)

    # EdotTs rel file
    if load_params()["FLD"] == True:
        try:
            load_EdotTsrel(logMdot)
        except:
            EdotTsrel_exists = False
        else:
            EdotTsrel_exists = True
    else:
        EdotTsrel_exists = "(not FLD)"

    # Datafile
    try:
        read_from_file(logMdot)
    except:
        datafile_exists = False
    else:
        datafile_exists = True



    if returnit:
        return {"root_exists":root_exists,
                "EdotTsrel_exists":EdotTsrel_exists,
                "datafile_exists":datafile_exists}
    else:
        print('Root exists \t\t: ',root_exists)
        print('EdotTsrel exists \t: ',EdotTsrel_exists)
        print('Data file exists \t: ',datafile_exists)
