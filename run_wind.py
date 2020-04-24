import IO
if IO.load_params()['FLD'] == True:
    from wind_GR_FLD import MakeWind
else:
    from wind_GR import Makewind

def run_wind(logMdot):
    # if roots exist, run this command to produce the full wind and save 

    logMdots,roots = IO.load_roots()
    root = roots[logMdots.index(logMdot)]
    w = MakeWind(root, logMdot, mode='wind',Verbose=1)
    IO.write_to_file(logMdot, w)

import sys
if __name__ == "__main__":
    if len(sys.argv) == 2:
        run_wind(eval(sys.argv[1]))
