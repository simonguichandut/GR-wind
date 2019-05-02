import numpy as np
from numpy.linalg import norm
from numpy import array
import sys

from wind_GR import MakeWind

def Jacobian(func,za,zb,*args):

    ''' First order numerical jacobian of a two input, two output function '''

    dx,dy = zb[0]-za[0] , zb[1]-za[1]
    faa,fba,fab = func(za,*args) , func([zb[0],za[1]],*args) , func([za[0],zb[1]],*args) 
    J11 = (fba[0]-faa[0])/dx
    J12 = (fab[0]-faa[0])/dy
    J21 = (fba[1]-faa[1])/dx
    J22 = (fab[1]-faa[1])/dy
    
    return [[J11,J12],[J21,J22]]


def JacobianUpdate(J0,za,zb,fa,fb,*args):
    
    ''' Jacobian update from Broyden's method '''
    
    J0,za,zb,fa,fb = array(J0),array(za),array(zb),array(fa),array(fb)
    dz,df = zb-za,fb-fa
    Jnew = J0 + 1/norm(dz)**2*np.outer(df-np.matmul(J0,dz),dz)
    return Jnew
    

def Newton_Raphson_2D(func,z0,z1,limits,*args,tol=1e-3,flagcatch=0):

    ''' Newton-Raphson root finding for a two input, two output function
        Two initial guesses have to be provided. Will soften iterations if 
        going out of bounds [xmin,xmax,ymin,ymax], or if obtaining invalid
        values. '''
    
    diff = 2*tol
    counter,f0,f1 = 0,func(z0,*args),func(z1,*args)
    Jold = [[0.00001,0.00001],[0.00001,-0.00001]]
    while (diff>tol or norm(f1)>tol) and (counter<100):

        J = Jacobian(func,z0,z1,*args)
        if counter != 0 and (True in np.isnan(J) or np.linalg.det(J)==0.0):
            print('Using Jacobian update')
            J = JacobianUpdate(Jold,z0,z1,f0,f1)
        
        pillow = 1
        znew = np.array(z1) - pillow*np.matmul(np.linalg.inv(J),f1)
        
        #  Out of bounds, catch level 1
        while znew[0]<limits[0] or znew[0]>limits[1] or znew[1]<limits[2] or znew[1]>limits[3] or (True in np.isnan(func(znew,*args))) :
            pillow/=3
            print('pillow update : ',pillow)
            znew = np.array(z1) - pillow*np.matmul(np.linalg.inv(J),f1)
            
        fnew = func(znew,*args)
            
        # solution space is not necesarily rectangular, catch level 2
        if flagcatch != 0:
            while flagcatch in fnew or (True in np.isnan(fnew)):
                pillow/=3
                print('pillow update v2: ',pillow)
                znew = np.array(z1) - pillow*np.matmul(np.linalg.inv(J),f1)
                fnew = func(znew,*args)
        
        z0,f0 = z1[:],f1[:]
        z1,f1 = znew[:],fnew[:]
        Jold = J
        print(counter+1,' : update [Edot,Ts]=',z1,'\n\n')
        
        diff = abs(norm(z1)-norm(z0))  # difference between two iterations
        counter += 1
        
    return z1


def RootFinder(logMdot,logTs0=7.3,box='on',Verbose=0):  
    
    ''' Finds the error-minimizing set of parameters (Edot,Ts) for a wind 
        with a given Mdot '''
        
    print('Starting root finding algorithm for logMdot = %.2f'%logMdot)
        
    Edotmin = 1.01 # in units of LEdd. There are likely no solutions with a lower Edot
    
    # But this Edot might not converge
    err=MakeWind([Edotmin,logTs0],logMdot,Verbose=Verbose)
    while 100 in err:
        Edotmin += 0.001
        print('\nEdotmin: ',Edotmin)
        err=MakeWind([Edotmin,logTs0],logMdot,Verbose=Verbose)
        
        if Edotmin>1.1:
            sys.exit('Inadequate initial Ts0, exiting..')
              
    # Initial guesses (the second one just needs to be closer to the solution)
    z0,z1 = [Edotmin+0.002,logTs0] , [Edotmin+0.0025,logTs0-0.01]

    # It happens that z1 does not work..
    err = MakeWind(z1,logMdot,Verbose=Verbose)
    i=1
    while 100 in err:
        print('z1 inadequate')
        z1 = [Edotmin+0.0025 , logTs0 + (-1)**i *i*0.01] # Alternating between going below and above Ts0
        err = MakeWind(z1,logMdot,Verbose=Verbose)
        i += 1  
        if i == 10:
            sys.exit('Inadequate initial Ts0, exiting..')

    limits = [1,1.1,6.5,8]
    
    print('\nStarting Rootfinding with first two iterations of (Edot,Ts) : ',z0,z1)
    root = Newton_Raphson_2D(MakeWind,z0,z1,limits,logMdot,flagcatch=100)

    return root





# Test 
#RootFinder(18)
    









#%%

#n=20
#Errors = [ [[] for i in range(n)] for k in range(2) ]
#Edotvals = np.linspace(1.001,1.04,n)
#Tsvals = np.linspace(6.8,7.5,n)
    
n=50
Errors = [ [[] for i in range(n)] for k in range(2) ]
Edotvals = np.linspace(1.01,1.05,n)
Tsvals = np.linspace(7,7.5,n)

def get_map(logMDOT): 
    
    for i,Ts in zip(np.arange(n),Tsvals):
        print('\n\n LOG TS = %.3f \n\n'%Ts)
        for Edot in Edotvals:
            z = MakeWind([Edot,Ts],logMDOT)
            Errors[0][i].append(z[0])
            Errors[1][i].append(z[1])
            print(Edot,' : ',z)
    
get_map(19)

    
# we want white to be solutions (0)
    
import pickle
[Edotvals,Tsvals,Errors]=pickle.load(open('save19.p','rb'))


import matplotlib.pyplot as plt
cmap = plt.get_cmap('RdBu')
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,8))    
ax1.patch.set_color('.25')
ax2.patch.set_color('.25')
levels1=np.arange(-1,1,0.01)
levels2=np.arange(-1,1,0.01)
im1 = ax1.contourf(Edotvals,Tsvals,np.array(Errors[0]),levels=levels1,cmap=cmap)
im2 = ax2.contourf(Edotvals,Tsvals,np.array(Errors[1]),levels=levels2,cmap=cmap)
fig.colorbar(im1,ax=ax1)
fig.colorbar(im2,ax=ax2)

ax1.set_xlabel(r'$\dot{E}/L_{Edd}$',fontsize=15)
ax2.set_xlabel(r'$\dot{E}/L_{Edd}$',fontsize=15)
ax1.set_ylabel(r'log $T_s$ (K)',fontsize=15)
ax1.set_title(r'Error #1 : ($L_{phot}-4\pi r^2\sigma T^4$)',fontsize=15)
ax2.set_title(r'Error #2 : ($R_{base}-R_{NS}$ km)',fontsize=15)    
              
            
ax1.set_xlim([Edotvals[0],Edotvals[-1]])
ax1.set_ylim([Tsvals[0],Tsvals[-1]])  
ax2.set_xlim([Edotvals[0],Edotvals[-1]])
ax2.set_ylim([Tsvals[0],Tsvals[-1]])         


    
#   Animation             
#ax1.plot([points[0][1][0]],[points[0][1][1]],marker='o',color=points[0][0])
#ax2.plot([points[0][1][0]],[points[0][1][1]],marker='o',color=points[0][0])
#plt.pause(0.01)
#fig.savefig('plots/GR_Simon/rootfinding/000001.png')
#for i in range(1,len(points)):
#    
#    x0,y0 = points[i-1][1][0],points[i-1][1][1]
#    x1,y1 = points[i][1][0],points[i][1][1]
#    c = points[i][0]
#    
#    ax1.plot([x0,x1],[y0,y1],marker='.',color=c)
#    ax2.plot([x0,x1],[y0,y1],marker='.',color=c)
#    
##    fig.savefig('plots/GR_Simon/rootfinding/%06d.png'%(i+1))
#    plt.pause(0.1)
##    


#import warnings
#warnings.filterwarnings("ignore", category=RuntimeWarning) 
#warnings.filterwarnings("ignore", category=ODEintWarning) 