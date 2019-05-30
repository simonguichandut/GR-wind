import os
import sys
import contextlib
import numpy as np
import scipy.integrate as integrate
from numpy import pi

def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd

@contextlib.contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    https://stackoverflow.com/a/22434262/190597 (J.F. Sebastian)
    """
    if stdout is None:
       stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied: 
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            #NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


# #Constants and parameters
# alpha=1/137.
# k=1.e-9     
# T=40.    
# V= 6.e-6
# r = 6.9673e12
# u = 1.51856e7

# #defining dy/dt's
# def f(y, t):
#        A, B, C, D, E = y
#        # the model equations
#        f0 = 1.519e21*(-2*k/T*(k - (alpha/pi)*(B+V))*A) 
#        f1 = ((3*B**2 + 3*C**2 + 6*B*C + 2*pi**2*B*T + pi**2*T**2)**-1
#              *(-f0*alpha/(3*pi**3) - 2*r*(B**3 + 3*B*C**2 + pi**2*T**2*B) 
#                - u*(D**3 - E**3)))
#        f2 = u*(D**3 - E**3)/(3*C**2)
#        f3 = -u*(D**3 - E**3)/(3*D**2)
#        f4 = u*(D**3 - E**3)/(3*E**2) + r*(B**3 + 3*B*C**2 + pi**2*T**2*B)/(3*E**2)
#        return [f0, f1, f2, f3, f4]


# t  = np.linspace(1e-15, 1e-10, 1000000)   # time grid
# y2 = [2e13, 0, 50, 50, 25]
# t2  = np.linspace(1.e-10, 1.e-5, 1000000)  

# with stdout_redirected():
#     soln2 = integrate.odeint(f, y2, t2, mxstep = 5000)
# # soln2 = integrate.odeint(f, y2, t2, mxstep = 5000)

# print(soln2)