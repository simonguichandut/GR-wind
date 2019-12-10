import sys
sys.path.append(".")

import numpy as np
from numpy.linalg import norm
from numpy import array
import pickle    


#### Plotting

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker
from numpy import array


plt.style.use('seaborn-paper')
nice_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        # Non-italic math
        "mathtext.default": "regular"
}

mpl.rcParams.update(nice_fonts)
from my_plot import set_size


def plot_map(logMDOT,img='pdf'):

    filename = 'analysis/errorspaces/save'+str(logMDOT)+'.p'
    [Edotvals,Tsvals,Errors]=pickle.load(open(filename,'rb'))

    fig,(ax1,ax2) = plt.subplots(1,2)    
    fig.subplots_adjust(wspace=0.25)
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
    levs=np.logspace(-3,1,50)
    im1 = ax1.contourf(Edotvals,Tsvals,np.abs(array(Errors[0])),levs,norm=colors.LogNorm(),cmap=cmap)
    im2 = ax2.contourf(Edotvals,Tsvals,np.abs(array(Errors[1])),levs,norm=colors.LogNorm(),cmap=cmap)
    fig.colorbar(im1,ax=ax1,ticks=[1e-3, 1e-2, 1e-1, 1e0, 1e1])
    fig.colorbar(im2,ax=ax2,ticks=[1e-3, 1e-2, 1e-1, 1e0, 1e1])
    ax1.set_title(r'Error 1 : $\vert L_{ph}-4\pi r^2\sigma T^4\vert$/$L_{ph}$')
    ax2.set_title(r'Error 2 : $\vert r_{b}-R_{NS}$/$R_{NS}\vert$')    


    ax1.set_xlabel(r'$\dot{E}/L_{Edd}$')
    ax2.set_xlabel(r'$\dot{E}/L_{Edd}$')
    ax1.set_ylabel(r'log $T_s$ (K)')

                
    ax1.set_xlim([Edotvals[0],Edotvals[-1]])
    ax1.set_ylim([Tsvals[0],Tsvals[-1]])  
    ax2.set_xlim([Edotvals[0],Edotvals[-1]])
    ax2.set_ylim([Tsvals[0],Tsvals[-1]])     
    
    
    # Add errors
    
    # 100 on left side
    E100left,Ts100left = [],[]
    for i in range(len(Edotvals)):
        for j in range(len(Tsvals)):
            if Errors[0][j][i]==100:
                E100left.append(Edotvals[i])
                Ts100left.append(Tsvals[j])
    
    err100 = 100*np.ones((len(E100left),len(Ts100left)))
    ax1.scatter(E100left,Ts100left,s=5,c='c',label='100')
    ax1.legend(loc=4)
    
    # 100 on left side
    E100right,Ts100right = [],[]
    for i in range(len(Edotvals)):
        for j in range(len(Tsvals)):
            if Errors[1][j][i]==100:
                E100right.append(Edotvals[i])
                Ts100right.append(Tsvals[j])
    
    err100 = 100*np.ones((len(E100right),len(Ts100right)))
    ax2.scatter(E100right,Ts100right,s=5,c='g',label='100')
    ax2.legend(loc=4)
    

    # plt.tight_layout()
    plt.show()


plot_map(17.2)
