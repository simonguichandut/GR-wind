## Physics for this problem : equation of state (EOS)

from numpy import sqrt

# Constants
kB = 1.380658e-16
arad = 7.5657e-15
c = 2.99792458e10
mp = 1.67e-24

# some functions have a **kwargs argument which is there so that dummy arguments can be 
# put in the function call without adding errors. In one EOS, Beta could have 2 args, and in the
# other it could have 4 args. With kwargs we can always call it with 4 args and the last 2 arguments
# might just be dummy arguments. In that case these last two arguments must be called with key=arg (kwarg)


class EOS:

    def __init__(self, comp):

        self.comp = comp

        # X,Y,Z : H, He, metals fraction (X+Y+Z=1)
        # Zno : atomic number
    
        # Homogeneous
        if self.comp in ('He','H','Ni'):

            if self.comp == 'He':
                self.X = 0
                self.Y = 1
                self.Z = 0
                self.Zno = 2
                self.mu_I = 4

            elif self.comp == 'H' :
                self.X = 1
                self.Y = 0
                self.Z = 0
                self.Zno = 1
                self.mu_I = 1

            elif self.comp == 'Ni':
                self.X = 0
                self.Y = 0
                self.Z = 1
                self.Zno = 28
                self.mu_I = 56

            self.mu_e = 2/(1+ self.X)                                   # http://faculty.fiu.edu/~vanhamme/ast3213/mu.pdf
            self.mu = 1/(1/self.mu_I + 1/self.mu_e)

        # Heterogeneous
        else:
            if self.comp.lower() == 'solar':
                self.X, self.Y, self.Z = 0.7, 0.28, 0.02            # http://www.astro.wisc.edu/~townsend/resource/teaching/astro-310-F08/21-eos.pdf
            else:
                # Expecting "X,Y,Z"
                self.X, self.Y, self.Z = eval(comp)

            self.mu = 1/(2*self.X + 3*self.Y/4 + self.Z/2)        # http://www.astro.wisc.edu/~townsend/resource/teaching/astro-310-F08/21-eos.pdf
            self.mu_e = 2/(1+ self.X)
            self.mu_I = 1/(1/self.mu - 1/self.mu_e)

        self.kappa0 = 0.2 * (1+self.X)                              # https://www.astro.princeton.edu/~gk/A403/opac.pdf


    # Opacity

    def kff(self,rho,T):
        if self.comp in ('He','H','Ni'):
            return 1e23*self.Zno**2/(self.mu_e*self.mu_I)*rho*T**(-7/2)
        else:
            return 3.68e22 * (1-self.Z)*(1+self.X)*rho*T**(-7/2)

    def kappa(self,rho,T):
        # return kappa0/(1.0+(T/4.5e8)**0.86)     
        return self.kappa0/(1.0+(T/4.5e8)**0.86) + self.kff(rho,T)

    # Ideal gas sound speed c_s^2
    def cs2(self,T): 
        return kB*T/(self.mu*mp)

    # P and E for : Ideal gas + radiation
    def pressure(self, rho, T, **kwargs):  
        return rho*self.cs2(T) + arad*T**4/3.0 

    def internal_energy(self, rho, T, **kwargs):  
        return 1.5*self.cs2(T)*rho + arad*T**4 

    # Pressure ratio
    def Beta(self, rho, T, **kwargs):  # pressure ratio 
        Pg = rho*self.cs2(T)
        Pr = arad*T**4/3.0
        return Pg/(Pg+Pr)

    # Rest energy + enthalpy
    def H(self, rho, T, **kwargs): 
        return c**2 + (self.internal_energy(rho, T) + self.pressure(rho, T))/rho


    ## -------- Degenerate electron corrections --------

    def electrons(self, rho, T):  # From Paczynski (1983) semi-analytic formula : ApJ 267 315

        rY = rho/self.mu_e # rho*Ye = rho/mu)e
        pednr = 9.91e12 * (rY)**(5/3)     
        pedr = 1.231e15 * (rY)**(4/3)
        ped = 1/sqrt((1/pedr**2)+(1/pednr**2))
        pend = kB/mp*rY*T
        pe = sqrt(ped**2 + pend**2) # pressure
        
        f = 5/3*(ped/pednr)**2 + 4/3*(ped/pedr)**2
        Ue = pe/(f-1)               # energy density (erg cm-3)

        alpha1,alpha2 = (pend/pe)**2 , (ped/pe)**2
        
        return pe,Ue,[alpha1,alpha2,f]

    # Ions ideal gas sound speed
    def cs2_I(self, T):  
        return kB*T/(self.mu_I*mp)

    # P and E for : Ideal gas + radiation + electrons
    def pressure_e(self, rho, T, **kwargs):
        pe,_,_ = self.electrons(rho,T)
        return rho*self.cs2_I(T) + arad*T**4/3.0 + pe

    def internal_energy_e(self, rho, T, **kwargs): 
        _,Ue,_ = self.electrons(rho,T)
        return 1.5*self.cs2_I(T)*rho + arad*T**4 + Ue

    # Pressure ratios
    def Beta_I(self, rho, T, **kwargs):
        pg = rho*self.cs2_I(T)
        return pg/self.pressure_e(rho,T)

    def Beta_e(self, rho, T, **kwargs):
        pe,_,_ = self.electrons(rho,T)
        return pe/self.pressure_e(rho,T)

    # Rest energy + enthalpy
    def H_e(self, rho, T, **kwargs):  # eq 2c
        return c**2 + (self.internal_energy_e(rho, T) + self.pressure_e(rho, T))/rho





class EOS_FLD:

    # This equation of state has a variable radiation pressure, Pr=faT^4, where f=1/3 in optically thick regions, f=1 in optically thin.

    def __init__(self, comp):

        self.comp = comp

        # X,Y,Z : H, He, metals fraction (X+Y+Z=1)
        # Zno : atomic number
    
        # Homogeneous
        if self.comp in ('He','H','Ni'):

            if self.comp == 'He':
                self.X = 0
                self.Y = 1
                self.Z = 0
                self.Zno = 2
                self.mu_I = 4

            elif self.comp == 'H' :
                self.X = 1
                self.Y = 0
                self.Z = 0
                self.Zno = 1
                self.mu_I = 1

            elif self.comp == 'Ni':
                self.X = 0
                self.Y = 0
                self.Z = 1
                self.Zno = 28
                self.mu_I = 56

            self.mu_e = 2/(1+ self.X)                                   # http://faculty.fiu.edu/~vanhamme/ast3213/mu.pdf
            self.mu = 1/(1/self.mu_I + 1/self.mu_e)

        # Heterogeneous
        else:
            if self.comp.lower() == 'solar':
                self.X, self.Y, self.Z = 0.7, 0.28, 0.02            # http://www.astro.wisc.edu/~townsend/resource/teaching/astro-310-F08/21-eos.pdf
            else:
                # Expecting "X,Y,Z"
                self.X, self.Y, self.Z = eval(comp)

            self.mu = 1/(2*self.X + 3*self.Y/4 + self.Z/2)        # http://www.astro.wisc.edu/~townsend/resource/teaching/astro-310-F08/21-eos.pdf
            self.mu_e = 2/(1+ self.X)
            self.mu_I = 1/(1/self.mu - 1/self.mu_e)

        self.kappa0 = 0.2 * (1+self.X)                              # https://www.astro.princeton.edu/~gk/A403/opac.pdf


    # Opacity

    def kff(self,rho,T):
        if self.comp in ('He','H','Ni'):
            return 1e23*self.Zno**2/(self.mu_e*self.mu_I)*rho*T**(-7/2)
        else:
            return 3.68e22 * (1-self.Z)*(1+self.X)*rho*T**(-7/2)

    def kappa(self,rho,T):
        return self.kappa0/(1.0+(T/4.5e8)**0.86)     
        # return self.kappa0/(1.0+(T/4.5e8)**0.86) + self.kff(rho,T)

    # Ideal gas sound speed c_s^2
    def cs2(self,T): 
        return kB*T/(self.mu*mp)

    def rad_pressure(self, T, lam, R): # lambda and R are the flux-limited diffusion parameters (Levermore & Pomraning)
        return (lam + (lam*R)**2)*arad*T**4

    # P and E for : Ideal gas + radiation
    def pressure(self, rho, T, lam, R):  
        return rho*self.cs2(T) + self.rad_pressure(T,lam,R)

    def internal_energy(self, rho, T):  
        return 1.5*self.cs2(T)*rho + arad*T**4 

    # Pressure ratio
    def Beta(self, rho, T, lam, R):  # pressure ratio 
        Pg = rho*self.cs2(T)
        Pr = self.rad_pressure(T,lam,R)
        return Pg/(Pg+Pr)

    # Rest energy + enthalpy
    def H(self, rho, T, lam, R): 
        return c**2 + (self.internal_energy(rho, T) + self.pressure(rho, T, lam, R))/rho


    ## -------- Degenerate electron corrections --------

    def electrons(self, rho, T):  # From Paczynski (1983) semi-analytic formula : ApJ 267 315

        rY = rho/self.mu_e # rho*Ye = rho/mu)e
        pednr = 9.91e12 * (rY)**(5/3)     
        pedr = 1.231e15 * (rY)**(4/3)
        ped = 1/sqrt((1/pedr**2)+(1/pednr**2))
        pend = kB/mp*rY*T
        pe = sqrt(ped**2 + pend**2) # pressure
        
        f = 5/3*(ped/pednr)**2 + 4/3*(ped/pedr)**2
        Ue = pe/(f-1)               # energy density (erg cm-3)

        alpha1,alpha2 = (pend/pe)**2 , (ped/pe)**2
        
        return pe,Ue,[alpha1,alpha2,f]

    # Ions ideal gas sound speed
    def cs2_I(self, T):  
        return kB*T/(self.mu_I*mp)

    # P and E for : Ideal gas + radiation + electrons
    def pressure_e(self, rho, T, lam, R):
        pe,_,_ = self.electrons(rho,T)
        return rho*self.cs2_I(T) + self.rad_pressure(T,lam,R) + pe

    def internal_energy_e(self, rho, T): 
        _,Ue,_ = self.electrons(rho,T)
        return 1.5*self.cs2_I(T)*rho + arad*T**4 + Ue

    # Pressure ratios
    def Beta_I(self, rho, T, lam, R):
        pg = rho*self.cs2_I(T)
        return pg/self.pressure_e(rho,T,lam,R)

    def Beta_e(self, rho, T, lam, R):
        pe,_,_ = self.electrons(rho,T)
        return pe/self.pressure_e(rho,T,lam,R)

    # Rest energy + enthalpy
    def H_e(self, rho, T, lam, R):  # eq 2c
        return c**2 + (self.internal_energy_e(rho, T) + self.pressure_e(rho, T, lam, R))/rho
