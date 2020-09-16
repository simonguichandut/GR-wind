import IO
if IO.load_params()['FLD'] == True:
    from wind_GR_FLD import MakeWind
else:
    from wind_GR import MakeWind

logMdots,roots = IO.load_roots()
IO.make_directories()

def run_wind(logMdot):
    # if roots exist, run this command to produce the full wind and save 
    
    root = roots[logMdots.index(logMdot)]
    w = MakeWind(root, logMdot, mode='wind',Verbose=1)
    IO.write_to_file(logMdot, w)

import sys
if __name__ == "__main__":

    i = sys.argv[1]
    
    if len(i)>10 and ',' not in i:
        sys.exit('Give Rphots separated by commas and no spaces')

    if i=='all':
        for logMdot in logMdots:
                run_wind(logMdot)
    else:
        if ',' in i:
            for logMdot in (eval(x) for x in i.split(',')):
                run_wind(logMdot)
        else:
            run_wind(eval(i))

