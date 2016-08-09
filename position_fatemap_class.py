import numpy as np
import matplotlib as mpl
import os
from PIL import Image
from pylab import *


class positionFatemap:
    '''Tools for plotting points according to colorscheme to track fate over time'''
    
    def __init__(self,tinput,track_lin,lineage_color):
        '''Saves input variables'''
        
        self.tinput = tinput
        self.track_lin = track_lin
        self.lineage_color = lineage_color

    def imagePlot(self,time,xc,yc,xp,yp,image,title,size=100,save=False,name=None,outpath=None):
        '''Plots position of tracks on a single image and timepoint with x,y,z shift to correct for cropping'''

        # xc,yc = Lcorrection[0],Lcorrection[1]

        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)

        Lpos,Lcolor = [],[]

        for trackid in self.tinput:

            track = self.tinput[trackid]

            Lpos.append(track[0][time])
            Lcolor.append(self.lineage_color[self.track_lin[trackid]])

        pos = np.array(Lpos)

        im = Image.open(image)#.transpose(Image.ROTATE_90)
        # Image.FLIP_TOP_BOTTOM
        ax.imshow(im)

        ax.scatter(pos[:,xp]+xc, pos[:,yp]+yc, c=Lcolor,s=size,linewidth=0.6)

        ax.set_title(title)

        # print(pos)

        plt.setp(ax.get_yticklabels(), visible = False)
        plt.setp(ax.get_xticklabels(), visible = False)
        plt.axis('off')

        fig.show()

        cwd = os.getcwd()

        if save == True and name != None:
            if outpath != None:
                os.chdir(outpath)

            filename = name
            fig.savefig(filename, dpi = 1200,bbox_inches='tight',pad_inches=0)

        os.chdir(cwd)
        
    ### Functions not in use ###
        
    def posGraph(self,time,numt):
        '''Generate position xyz graphs for n timepoints with color by lineage, no image'''
        
        Lpos, Llin, Lcolor  = [], [], []
        
        for trackid in self.tinput:

            track = self.tinput[trackid]
            
            Llin.append(self.track_lin[trackid])
            Lcolor.append(self.lineage_color[self.track_lin[trackid]])
        
            for t in time:
                Lpos.append(track[0][t])
            
        lin = np.array(Llin)
        color = np.array(Lcolor)
        
#         fig = plt.figure(num='Position Fatemap',figsize=(12,10)) #adjust figsize based on 
        
        titlepos = numt * [' XY',' YZ',' XZ']
        
        titlet = []
        for t in time:
            titlet = titlet + [t]*numt
        
        xaxis = numt * ['X','Y','X']
        yaxis = numt * ['Y','Z','Z']
        xpos = numt * [0,1,0]
        ypos = numt * [1,2,2]
        Lsplt = []
        
        splots = numt*100 + 30
        
        for i in range(numt+1):
            
            ax = plt.subplot(numt,3,i)
            ax.set_title(str(titlet[i]) + titlepos[i])
            ax.set_xlabel(xaxis[i])
            ax.set_ylabel(yaxis[i])
            
            data = Lpos[i]
            
            ax.scatter(data[:,xpos[i]],data[:,ypos[i]],s=70,c=color)

    def posGraphv2(self,stime,etime,save=False,name=None,path=None):
        '''Generate position xyz graphs for 2 timepoints with color by lineage'''
        
        cwd = os.getcwd()
        
        Lspos, Lepos, Llin, Lcolor  = [], [], [], []
        
        for trackid in self.tinput:

            track = self.tinput[trackid]
            
            Lspos.append(track[0][stime])
            Lepos.append(track[0][etime])
            Llin.append(self.track_lin[trackid])
            Lcolor.append(self.lineage_color[self.track_lin[trackid]])
            
        spos = np.array(Lspos)
        epos = np.array(Lepos)
        lin = np.array(Llin)
        color = np.array(Lcolor)
        
        fig = plt.figure(num='Start/End Position',figsize=(12,12))
        
        titlepos = [' XY',' YZ',' XZ',
                    ' XY',' YZ',' XZ']
        titlet = [stime,stime,stime,etime,etime,etime]
        xaxis = ['X','Y','X','X','Y','X']
        yaxis = ['Y','Z','Z','Y','Z','Z']
        xpos = [0,1,0,0,1,0]
        ypos = [1,2,2,1,2,2]
        data = [spos,spos,spos,epos,epos,epos]
        Lsplt = []
        
        for i,j in enumerate(range(231,237)):
            
            splt = fig.add_subplot(j)
            splt.set_title(str(titlet[i]) + titlepos[i])
            splt.set_xlabel(xaxis[i])
            splt.set_ylabel(yaxis[i])
            
            splt.scatter(data[i][:,xpos[i]],data[i][:,ypos[i]],s=70,c=color)
            
            Lsplt.append(splt)
            
        if save == True and name != None:
            if path != None:
                os.chdir(path)
            filename = 'PF'+str(stime)+'-'+str(etime)+name+'.jpg'
            fig.savefig(filename,dpi=1200)
            
        os.chdir(cwd)
            
    def multiplaneImagePlotsFM(self,timepoints,planes,impath,save=False,name=None,outpath=None):
        '''Plot three persepectives for each timepoint with plots on images'''

        fig,ax = plt.subplots(len(timepoints),len(planes),figsize=(20,20))

        Lpos = []
        Lcolor = []

        for i,time in enumerate(timepoints):
            Lpos.append([])

        for trackid in self.tinput:

            track = self.tinput[trackid]

            for j,time in enumerate(timepoints):

                Lpos[j].append(track[0][time])
                Lcolor.append(self.lineage_color[self.track_lin[trackid]])

        pos = np.array(Lpos)
        color = np.array(Lcolor)
        # print(color)

        for i,plane in enumerate(planes):

            if plane == 'XY':
                x,y = 0,1
                xlabel,ylabel = 'X - AP', 'Y - DV'
                xmin,xmax = 0,700
                ymin,ymax = 600,100
            if plane == 'YZ':
                x,y = 1,2
                xlabel,ylabel = 'Y - DV', 'Z - ML'
            if plane == 'XZ':
                x,y = 0,2
                xlabel,ylabel = 'X - AP', 'Z - ML'

            plt.setp([a.get_yticklabels() for a in ax[:,i]],visible=False)

            cwd = os.getcwd()
            os.chdir(impath)

            for j,time in enumerate(timepoints):

                filename = impath + '/' + plane + '/MAX_' + str(time) + '_3D.tif'
                im = Image.open(filename)

                ax[j,i].imshow(im)
                ax[j,i].scatter(pos[j][:,x],pos[j][:,y],c=color,s=50)
                ax[j,i].set_xlim(xmin,xmax)
                ax[j,i].set_ylim(ymin,ymax)

                if i == 0:
                    ax[j,i].set_ylabel(time,labelpad=20) 
                if j == 0:
                    ax[j,i].set_title(xlabel+' '+ylabel)

                plt.setp([a.get_xticklabels() for a in ax[j,:]],visible=False)

                im.close()

        plt.setp([a.get_yticklabels() for a in ax[:,1]], visible=False)
        plt.setp([a.get_yticklabels() for a in ax[:,2]], visible=False)

        plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,wspace=0,hspace=0.075)
        
        if save == True and name != None:
            if outpath != None:
                os.chdir(outpath)
                
            filename = 'FM_images_' + name + '.jpg'
            fig.savefig(filename)
        
        os.chdir(cwd)