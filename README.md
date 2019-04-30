Calculates the structure of an 1D optically thick neutron star wind in GR 
Uses a GR analog of the "phi" variable of [Joss & Melia (1987)](http://adsabs.harvard.edu/abs/1987ApJ...312..700J) to integrate away and inwards from the critical point.
Referencing equations from [Paczynski & Proczynski (1986)](http://adsabs.harvard.edu/abs/1986ApJ...302..519P).

Questions and comments -> simon.guichandut@mail.mcgill.ca

Uses a Newton-Raphson root finding approach to obtain the minimizing set of parameters (Ts,Edot) .  Pre-solved roots are available in the `solutions` subfolder.

To make plots :

    python Plots.py

Plots and tables are in the `results` subfolder, organized in directories describing the model:
Composition_M/Msun_R/km_TauOuter_logPinner -> e.g `He_1.4_10_3_4/`
