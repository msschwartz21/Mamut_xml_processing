#Filetype for output should be specified in the filename

import os 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm 
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

def graphXYZTracks(tinput,labels=True,name=None,save=False,path=None):
    '''Graph x,y,z seperately for track based data. 
    Plots all tracks in black and average in red.'''

    cwd = os.getcwd()

    if path != None:
        os.chdir(path)

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(311)
    ay = fig.add_subplot(312)
    az = fig.add_subplot(313)

    for trackid in tinput:
        track = tinput[trackid]

        ax.plot(track[:,0],c='k',alpha=.4)
        ay.plot(track[:,1],c='k',alpha=.4)
        az.plot(track[:,2],c='k',alpha=.4)

    tlist = [tinput[trackid] for trackid in tinput]
    tarray = np.array(tlist)
    x = tarray[:,:,0]
    y = tarray[:,:,1]
    z = tarray[:,:,2]

    ax.plot(np.median(x,axis=0),c='r',lw=2)
    ay.plot(np.median(y,axis=0),c='r',lw=2)
    az.plot(np.median(z,axis=0),c='r',lw=2)

    if labels == True:
        for splt,dim in zip([ax,ay,az],['AP','DV','ML']):
            splt.set_ylabel(dim)
        az.set_xlabel('Time')
        ax.set_title('Velocity')
    else:
        for splt in [ax,ay,az]:
            plt.setp(splt.get_yticklabels(), visible = False)
            plt.setp(splt.get_xticklabels(), visible = False)

    if save == True and name != None:
        fig.savefig(name,dpi=1200,bbox_inches='tight',pad_inches=0)

    os.chdir(cwd)

def graph3dTracks(tinput,track_color,labels=True,name=None,save=False,path=None):
    '''Graph 3D track based data using a dicitonary with a color assigned for each track'''

    fig = plt.figure(figsize=(10,8))

    ax = fig.add_subplot(111,projection='3d')

    if labels==True:
        ax.set_xlabel('X - AP')
        ax.set_zlabel('Y - DV')
        ax.set_ylabel('Z - ML')
    else:
        plt.setp(ax.get_xticklabels(), visible = False)
        plt.setp(ax.get_yticklabels(), visible = False)

    for trackid in tinput:
        track = tinput[trackid][0]
        ax.plot(track[:,0],track[:,2],track[:,1],c=track_color[trackid])

    ax.view_init(elev=10,azim = -90)

    cwd = os.getcwd()

    if save == True and name != None:
        if path != None:
            os.chdir(path)
        filename = '3D_' + name
        fig.savefig(filename,dpi=1200,bbox_inches='tight',pad_inches=0)

    os.chdir(cwd)

def gradientColorBar(cmap,save=False,name=None,outpath=None):
    '''Graph color bar according to input colormap'''

    gradient = np.linspace(0,1,23.3)
    gradient = np.vstack((gradient,gradient))

    fig = plt.figure(figsize=(10,2))
    ax = fig.add_subplot(111)

    ax.imshow(gradient,aspect='auto',cmap=plt.get_cmap(cmap))
    plt.setp(ax.get_xticklabels(),visible=False)
    plt.setp(ax.get_yticklabels(),visible=False)
    plt.axis('off')

    cwd = os.getcwd()

    if save==True and name!=None:
        if outpath != None:
           os.chdir(outpath)
        fig.savefig(name,dpi=1200,bbox_inches='tight',pad_inches=0)

    os.chdir(cwd)

def randomColor(tinput,cmap):
    '''Defines random color for each track
    Returns: track_color: dictionary with color for each track'''

    track_ids = tinput.keys()
    
    n = len(track_ids)
    color_norm = mpl.colors.Normalize(vmin=0,vmax=n)
    scalar_map = cm.ScalarMappable(norm=color_norm, cmap = cmap)
    
    track_color = {}
    
    for i,trackid in enumerate(track_ids):
        
        c = scalar_map.to_rgba(i)
        track_color[trackid] = c

    return track_color

def threeD_gradient(tinput,cmap,labels=True,save=False,name=None,outpath=None):
    '''Graphs 3D line plots with color gradient along time. Displays corresponding color bar.
    Does not display when running %matplotlib notebook'''

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111,projection = '3d')

    idkeys = tinput.keys()
    track_ids = np.array([k for k in idkeys])
    strack_ids = track_ids[:1]

    norm = mpl.colors.Normalize(vmin=0,vmax=500)
    m = plt.cm.ScalarMappable(norm=norm,cmap=cmap)

    normcbar = mpl. colors.Normalize(vmin=2,vmax=19)
    mcbar = plt.cm.ScalarMappable(norm=normcbar,cmap=cmap)
    mcbar.set_array([])

    for i,trackid in enumerate(track_ids):

        tdata = tinput[trackid][0]
        x,y,z = tdata[:,0],tdata[:,2],tdata[:,1]

        N = len(tdata)

        cols = 'rgbcmy'

        for j in range(N-1):
            ax.plot(x[j:j+2],y[j:j+2],z[j:j+2],color=m.to_rgba(j))  #plt.cm.inferno(255*i/N))

    cbar = plt.colorbar(mcbar,label='Hours After Egg Laying')

    if labels==True:
        ax.set_xlabel('X - AP')
        ax.set_zlabel('Y - DV')
        ax.set_ylabel('Z - ML')
    else:
        plt.setp(ax.get_xticklabels(), visible = False)
        plt.setp(ax.get_yticklabels(), visible = False)

    ax.view_init(elev=10,azim = -90)

    cwd = os.getcwd()

    if save == True and name != None:
        # if outpath != None:
        #   os.chdir(outpath)

        filename = name
        fig.savefig(filename, dpi = 1200,bbox_inches='tight',pad_inches=0)

    os.chdir(cwd)

def twoD_gradient(tinput,cmap,xi,yi,xc,yc,image,shrink=1,save=False,name=None,outpath=None):
    '''Draw gradient tracks onto a 2D cropped image of a particular persepctive.
    Displays colorbar as well'''

    fig = plt.figure()
    ax = fig.add_subplot(111)

    idkeys = tinput.keys()
    track_ids = np.array([k for k in idkeys])
    strack_ids = track_ids[:1]

    norm = mpl.colors.Normalize(vmin=0,vmax=500)
    m = plt.cm.ScalarMappable(norm=norm,cmap=cmap)
    # m.set_array([])

    normcbar = mpl. colors.Normalize(vmin=2,vmax=19)
    mcbar = plt.cm.ScalarMappable(norm=normcbar,cmap=cmap)
    mcbar.set_array([])

    im = Image.open(image)
    ax.imshow(im)

    for i,trackid in enumerate(track_ids):

        tdata = tinput[trackid][0]
        x,y = tdata[:,xi],tdata[:,yi]
        xshift = x+xc
        yshift = y+yc

        N = len(tdata)

        for j in range(N-1):
            ax.plot(xshift[j:j+2],yshift[j:j+2],color=m.to_rgba(j))

    plt.setp(ax.get_yticklabels(), visible = False)
    plt.setp(ax.get_xticklabels(), visible = False)
    plt.axis('off')

    cbar = plt.colorbar(mcbar,label='Hours After Egg Laying')#,shrink=0.65,pad=0.05)
    # cbar.anchor(0.0,0.1)


    cwd = os.getcwd()

    if save == True and name != None:

        if outpath != None:
            os.chdir(outpath)

        filename = name
        fig.savefig(filename,dpi = 1200,bbox_inches='tight',pad_inches=0)
        print('Image Saved')

    os.chdir(cwd)

### Functions not in use ###

def graph3dTrackList(Ltracks,tinput,figsize=(10,8),name=None,save=False,path=None):
    '''Graph 3D track based data'''

    cwd = os.getcwd()

    if path != None:
        os.chdir(path)

    if name != None:
        fig = plt.figure(num=name,figsize=figsize)
    else:
        fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(111,projection='3d')
    ax.set_xlabel('X - AP')
    ax.set_ylabel('Y - DV')
    ax.set_zlabel('Z - ML')
    ax.legend()
        
    for trackid in Ltracks:
        track = tinput[trackid][0]
        ax.plot(track[:,0],track[:,1],track[:,2])

    if save == True and name != None:
        filename = '3D_' + name
        fig.savefig(filename,dpi=1200,bbox_inches='tight',pad_inches=0)

    os.chdir(cwd)

def graphXYZTrackList(Ltracks,tinput,labels=True,figsize=(8,16),name=None,save=False,path=None):
    '''Graph x,y,z seperately for track based data'''

    cwd = os.getcwd()

    if path != None:
        os.chdir(path)

    figx = plt.figure(figsize=figsize)
    ax = figx.add_subplot(311)
    ax.legend()
    ay = figx.add_subplot(312)
    ay.legend()
    az = figx.add_subplot(313)
    az.legend()
    
    ax.set_ylim([200,600])
    ay.set_ylim([200,600])
    az.set_ylim([200,600])

    for trackid in Ltracks:
        track = tinput[trackid][0]

        ax.plot(track[:,0])
        ay.plot(track[:,1])
        az.plot(track[:,2])
        
    if labels == False:
        plt.setp(ax.get_xticklabels(), visible = False)
        plt.setp(ax.get_yticklabels(), visible = False)

    if save == True and name != None:
        filename = 'XYZ_' + name
        figx.savefig(filename,dpi=1200,bbox_inches='tight',pad_inches=0)

    cwd = os.getcwd()