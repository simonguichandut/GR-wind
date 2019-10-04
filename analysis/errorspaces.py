import sys
sys.path.append(".")

import numpy as np
from numpy.linalg import norm
from numpy import array
import pickle    
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker
from numpy import array

from wind_GR import MakeWind


def get_map(logMDOT):

    n=50
    Errors = [ [[] for i in range(n)] for k in range(2) ]
    Edotvals = np.linspace(1.01,1.05,n)
    Tsmax = 7.5
    Tsvals = np.linspace(6.9,Tsmax,n)
    
    for i,Ts in zip(np.arange(n),Tsvals):
        print('\n\n LOG TS = %.3f \n\n'%Ts)
        for Edot in Edotvals:
            try:
                z = MakeWind([Edot,Ts],logMDOT)
                print(Edot,' : ',z)
            except:
                z=[300,300]
                print("\nFatal error in integration, skipping...\n")

            Errors[0][i].append(z[0])
            Errors[1][i].append(z[1])
        
    filename = 'analysis/errorspaces/save'+str(logMDOT)+'.p'
    with open(filename,'wb') as f:
        pickle.dump([Edotvals,Tsvals,Errors],f)

    print('\n\nFINISHED \nMap file saved to ',filename)    


def plot_map(logMDOT):

    filename = 'analysis/errorspaces/save'+str(logMDOT)+'.p'
    [Edotvals,Tsvals,Errors]=pickle.load(open(filename,'rb'))

    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,8))    
    ax1.patch.set_color('.25')
    ax2.patch.set_color('.25')

    # Option 1 
#     cmap = plt.get_cmap('RdBu') # diverging colormap (white at center)
#     levels=np.arange(-1,1,0.01)
#     im1 = ax1.contourf(Edotvals,Tsvals,array(Errors[0]),levels=levels,cmap=cmap)
#     im2 = ax2.contourf(Edotvals,Tsvals,array(Errors[1]),levels=levels,cmap=cmap)
#     fig.colorbar(im1,ax=ax1)
#     fig.colorbar(im2,ax=ax2)
#     ax1.set_title(r'Error #1 : ($L_{phot}-4\pi r^2\sigma T^4$)',fontsize=15)
#     ax2.set_title(r'Error #2 : ($R_{base}-R_{NS}$)',fontsize=15)    

    # Option 2
    cmap = plt.get_cmap('YlOrRd') # diverging colormap (white at bottom)
    levs=np.logspace(-3,1,100)
    im1 = ax1.contourf(Edotvals,Tsvals,np.abs(array(Errors[0])),levs,norm=colors.LogNorm(),cmap=cmap)
    im2 = ax2.contourf(Edotvals,Tsvals,np.abs(array(Errors[1])),levs,norm=colors.LogNorm(),cmap=cmap)
    fig.colorbar(im1,ax=ax1,ticks=[1e-3, 1e-2, 1e-1, 1e0, 1e1])
    fig.colorbar(im2,ax=ax2,ticks=[1e-3, 1e-2, 1e-1, 1e0, 1e1])
    ax1.set_title(r'Error #1 : |$L_{phot}-4\pi r^2\sigma T^4$|/$L_{phot}$',fontsize=15)
    ax2.set_title(r'Error #2 : |$R_{base}-R_{NS}$|/$R_{NS}$',fontsize=15)    


    ax1.set_xlabel(r'$\dot{E}/L_{Edd}$',fontsize=15)
    ax2.set_xlabel(r'$\dot{E}/L_{Edd}$',fontsize=15)
    ax1.set_ylabel(r'log $T_s$ (K)',fontsize=15)

                
    ax1.set_xlim([Edotvals[0],Edotvals[-1]])
    ax1.set_ylim([Tsvals[0],Tsvals[-1]])  
    ax2.set_xlim([Edotvals[0],Edotvals[-1]])
    ax2.set_ylim([Tsvals[0],Tsvals[-1]])     
    
    fig.savefig('analysis/errorspaces/'+str(logMDOT)+'.png')    


# get_map(17.75)
plot_map(17.75)

# for m in [18,18.25,18.5,18.75,19]:
#         plot_map(m)

    
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
