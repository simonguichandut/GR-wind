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
    if len(sys.argv) == 2:
        if sys.argv[1] == 'all':
            for logMdot in logMdots:
                run_wind(logMdot)
        else:
            run_wind(eval(sys.argv[1]))
