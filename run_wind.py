import IO
if IO.load_params()['FLD'] == True:
    from wind_GR_FLD import MakeWind
    from RootFinding_FLD import ImproveRoot
else:
    from wind_GR import MakeWind

IO.make_directories()


def run_wind(logMdot, recursed=0):
    # if roots exist, run this command to produce the full wind and save 
    
    logMdots,roots = IO.load_roots()
    root = roots[logMdots.index(logMdot)]

    if recursed==5:
        print('Max recursed, can''t make wind for logMdot ',logMdot)
        return

    try:
        w = MakeWind(root, logMdot, mode='wind',Verbose=1)
        IO.write_to_file(logMdot, w)

    except Exception as E:

        if E.__str__() == 'Improve root':
            print('\nReinterpolating and finding better root\n')
            ImproveRoot(logMdot)
            run_wind(logMdot, recursed=recursed+1)
        
        else:
            print(E)


import sys
if __name__ == "__main__":

    i = sys.argv[1]
    
    if len(i)>10 and ',' not in i:
        sys.exit('Give Rphots separated by commas and no spaces')

    if i=='all':
        for logMdot in IO.load_roots()[0]:
                run_wind(logMdot)
    else:
        if ',' in i:
            for logMdot in (eval(x) for x in i.split(',')):
                run_wind(logMdot)
        else:
            run_wind(eval(i))

