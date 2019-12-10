## Description

Calculates the structure of an 1D optically thick neutron star wind in GR 
Uses a GR analog of the "phi" variable of [Joss & Melia (1987)](http://adsabs.harvard.edu/abs/1987ApJ...312..700J) to integrate away and inwards from the critical point.
Referencing equations from [Paczynski & Proczynski (1986)](http://adsabs.harvard.edu/abs/1986ApJ...302..519P).
Uses a Newton-Raphson root finding approach to obtain the minimizing set of parameters (Ts,Edot) .  

Questions and comments -> simon.guichandut@mail.mcgill.ca

## How-to

Parameters are given in `params.txt`
* M : NS mass in solar mass units                                                             
* R : NS radius in km                                                                                        
* y_inner : Column density at the beginning of the wind in g/cm2 
* tau_outer : optical depth at the photosphere (end of validity of opt. thick approximation)
* comp : Composition of the wind
* EOS_Type : Equation of state to use.  Options are Ideal Gas (IG), Ideal Gas with Degenerate Electrons (IGDE)
* FLD : boolean for using flux-limited diffusion to allow a transition to optically thin
* mode : Either find the wind solutions (rootsolve) or use pre-solved roots to make the wind and produce plots (wind)
* save : boolean for saving data and plots (0 will show plots in python interface)
* img : image format 

To find roots :

    python RootFinding.py all           # Tries to find all roots for logmdots 17:0.05:19

    python RootFinding 18,18.5,18.75    # Specific list of logmdots

To make plots :

    python Plots.py

Plots and tables are in the `results` subfolder, organized in directories describing the model:
Composition_EOS_M/Msun_R/km_tau_outer_log(y_inner)-> e.g `He_IGDE_M1.4_R10_tau3_y8/`

Pre-solved roots are available in the `wind_solutions` subfolder, organized in directories with the same name as above.


## Example plots

![](/results/He_IGDE_M1.4_R12_tau3_y8/plots/Luminosity.png)
![](/results/He_IGDE_M1.4_R12_tau3_y8/plots/Temperature1.png)
![](/results/He_IGDE_M1.4_R12_tau3_y8/plots/Velocity.png)
