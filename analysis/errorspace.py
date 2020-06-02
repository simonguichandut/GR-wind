import sys
sys.path.append(".")

import numpy as np
from numpy.linalg import norm
from numpy import array
import pickle    
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
                z = MakeWind([Edot,Ts],logMDOT, IgnoreErrors=True)
                print(Edot,' : ',z)
            except:
                pass
            #     z=[300,300]
            #     print("\nFatal error in integration, skipping...\n")

            Errors[0][i].append(z[0])
            Errors[1][i].append(z[1])
        
    filename = 'analysis/errorspaces/save'+str(logMDOT)+'.p'
    with open(filename,'wb') as f:
        pickle.dump([Edotvals,Tsvals,Errors],f)

    print('\n\nFINISHED \nMap file saved to ',filename)    



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
# from my_plot import set_size


def plot_map(logMDOT,img='pdf',xaxis='Edot'):

    filename = 'analysis/errorspaces/save'+str(logMDOT)+'.p'
    [Edotvals,Tsvals,Errors]=pickle.load(open(filename,'rb'))

    # fig,(ax1,ax2) = plt.subplots(1,2,figsize=set_size('mnras1col'))
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(5.95, 3.68))
       
    fig.subplots_adjust(wspace=0.3)
    ax1.patch.set_color('.25')
    ax2.patch.set_color('.25')

    # In the roots, Edot (/LEdd) is actually Edot-Mdotc^2!!
    if xaxis == 'Edot':
        from wind_GR import c,LEdd
        Edotvals = (Edotvals*LEdd + 10**logMDOT*c**2)/LEdd
        ax1.set_xlabel(r'$\dot{E}/L_{Edd}$')
        ax2.set_xlabel(r'$\dot{E}/L_{Edd}$')
    elif xaxis == 'Edot_minus_Mdotc2':
        ax1.set_xlabel(r'$(\dot{E}-\dot{M}c^2)/L_{Edd}$')
        ax2.set_xlabel(r'$(\dot{E}-\dot{M}c^2)/L_{Edd}$')


    ax1.set_ylabel(r'log $T_c$ (K)')


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
    # ax1.set_title(r'Error 1 : $\vert L_{ph}-4\pi r_{ph}^2\sigma T_{ph}^4\vert$/$L_{ph}$')
    ax1.set_title(r'Error 1 : $\vert L-4\pi r^2\sigma T^4\vert$/$L$')
    ax2.set_title(r'Error 2 : $\vert r_{b}-R\vert$/$R$')    

                
    ax1.set_xlim([Edotvals[0],Edotvals[-1]])
    ax1.set_ylim([Tsvals[0],Tsvals[-1]])  
    ax2.set_xlim([Edotvals[0],Edotvals[-1]])
    ax2.set_ylim([Tsvals[0],Tsvals[-1]])     
    
    
    # Add scatters for errors
    cases_left = (r'Mach $\rightarrow 1$',r'$\tau^*_{min}>3$','?','tau^*(r_c)<3')
    cases_right = ('u=0','Diverging (NaN)','?')
    color_left = ('m','c','w')
    color_right = ('r','g','k')
    legendleft,legendright = False,False
    
    for E,caseL,caseR,cl,cr in zip((100,200,300,400),cases_left,cases_right,color_left,color_right):
    
        # Left plot (error on outer int)
        Epointsleft,Tspointsleft = [],[]
        for i in range(len(Edotvals)):
            for j in range(len(Tsvals)):
                if Errors[0][j][i]==E:
                    Epointsleft.append(Edotvals[i])
                    Tspointsleft.append(Tsvals[j])
        
        if len(Epointsleft)>0:
            ax1.scatter(Epointsleft,Tspointsleft,s=3,label=caseL,color=cl)
            legendleft=True
            
        
        # Right plot (error on inner int)
        Epointsright,Tspointsright = [],[]
        for i in range(len(Edotvals)):
            for j in range(len(Tsvals)):
                if Errors[1][j][i]==E:
                    Epointsright.append(Edotvals[i])
                    Tspointsright.append(Tsvals[j])
        
        if len(Epointsright)>0:
            ax2.scatter(Epointsright,Tspointsright,s=3,label=caseR,color=cr)
            legendright=True
    
    
    if legendleft: ax1.legend(loc=4)
    if legendright: ax2.legend(loc=4)
        
    
    # plt.tight_layout()
    
    filename = str(logMDOT).replace('.','_') # dont want points in filenames
    fig.savefig('analysis/errorspaces/'+filename+'.'+img,bbox_inches='tight',format=img)    

# indivual call
# get_map(17.75)
#plot_map(18.5)


# command line call
if len(sys.argv)>1:
            
    mode = sys.argv[1]
    logMdot = sys.argv[2]

    if mode=='get':
        get_map(eval(logMdot))
    elif mode=='plot':
        img = 'pdf' if len(sys.argv)<4 else sys.argv[3]
        plot_map(eval(logMdot),img=img)
    
        






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
