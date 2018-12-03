import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm 
import os
import matplotlib.pyplot as plt
from scipy.spatial import distance
from itertools import combinations


class neighbors:
    '''Tools for doing a nearest neighbor analysis of cells'''
    
    def __init__(self,tinput,nearestnum,cmap,timepoints):
        '''Constructor, calculates nearest neighbors to begin with'''
        
        self.tinput = tinput
        idkeys = self.tinput.keys()
        self.track_ids = np.array([k for k in idkeys])
        
#         print('type',type(self.track_ids))
        self.randomColor(cmap)
        self.timepoints = timepoints
        
        self.trackArray()

        self.findNBArray(nearestnum)
        
    def randomColor(self,cmap):
        '''Define random color for each track
        Returns: self.track_color: dictionary of color per trackid'''
        
        n = len(self.track_ids)
        color_norm = mpl.colors.Normalize(vmin=0,vmax=n-1)
        scalar_map = cm.ScalarMappable(norm=color_norm, cmap = cmap)
        
        self.track_color = {}
        
        for i,trackid in enumerate(self.track_ids):
            
            c = scalar_map.to_rgba(i)
            self.track_color[trackid] = c
        
    def trackArray(self):
        '''Generate a single 3D array with the position data of all tracks
        Returns: self.tarray'''
        
        ntracks = len(self.track_ids)
        tlen = len(self.tinput[self.track_ids[0]][1])
        
        self.tarray = np.zeros((ntracks,tlen,3))
        
        for i,trackid in enumerate(self.track_ids):
            
            trackpos = self.tinput[trackid][0]
            self.tarray[i,:,:] = trackpos
            
    def calculateEucDist(self,time):
        '''Calculate euclidean distance between all tracks at specific time point
        Result: square matrix with distance between every pair of tracks'''
        
        postime = self.tarray[:,time,:]
        
        dist = distance.cdist(postime,postime,metric='euclidean')
        
        return dist
    
    def findNBArray(self,n):
        '''Find nearest neighbors for each track at each timepoint and put into single array
        Returns: self.nbarray: array with dimensions (num tracks, num timepoints, num neighbors)'''
        
        ntracks = len(self.track_ids)
        
        self.nbarray = np.zeros((len(self.track_ids),len(self.timepoints),n))
        
        for i,time in enumerate(self.timepoints):
            
            eucdist = self.calculateEucDist(time)
            eucdist[eucdist == 0] = np.nan
            
            for j,trackid in enumerate(self.track_ids):
                
                tdist = eucdist[j]
                min_index = np.argpartition(tdist,n)[:n]
                min_trackids = self.track_ids[min_index]
                        
                self.nbarray[j,i,:] = min_trackids
            
    def graphNearestNBs(self,trackid,labels=True,save=False,name=None,path=None):
        '''Graph scatterplot of tracks for each timepoint for a single track.
        Each track is assigned a unique color.'''
        
        debug = False
        
        cwd = os.getcwd()
        
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        
        
        #Grab data for a single track    
        trackdata = self.nbarray[np.where(self.track_ids == trackid),:,:]
        trackdata = trackdata[0]
        trackdata = trackdata[0]
        if debug: print('trackdata',trackdata)

        for i,time in enumerate(self.timepoints):
            nbs = trackdata[i]
            if debug: print('nbs', nbs, 'i', i)

            x = []
            y = []
            c = []

            for j,tracks in enumerate(nbs):
                if debug: print('tracks', tracks)
                if debug: print('j',j)

                if tracks != -1:
                    x.append(i)
                    y.append(np.where(self.track_ids == tracks))
                    c.append(self.track_color[tracks])

            if debug: print(x,y)

            ax.scatter(x,y,c=c,s=50,edgecolor='')


        plt.xlim(0,500)
        plt.ylim(-1,24)

        if labels == True:
            ax.set_title(trackid)
            ax.set_ylabel('Tracks')
            ax.set_xlabel('Time ')# + str(self.timepoints))

        if labels == False:
            plt.setp(ax.get_yticklabels(), visible = False)
            plt.setp(ax.get_xticklabels(), visible = False)
            
        if save == True and name != None:
            if path != None:
                os.chdir(path)
            fig.savefig(name, dpi = 1200,bbox_inches='tight',pad_inches=0)
            
            os.chdir(cwd)

    def windowConsistency(self,window):
        '''Calculate percent similarity for each timepoint using a rolling window
        Returns: self.win_cons'''

        nbshape = np.shape(self.nbarray)

        self.win_cons = np.zeros((nbshape[0],nbshape[1]))

        #Go through each track
        for i,trackid in enumerate(self.track_ids):

            print(trackid)

            tnbs = self.nbarray[i]

            #Go through each timepoint in a track
            for t, nbs in enumerate(tnbs):

                #Decide window range for each timepoint
                llim = max(0, t-window)
                rlim = min(500, t+window)

                to_compare = [set(l) for l in tnbs[llim:rlim+1]]

                comp_avg = 0
                nb_elements = 0
                for set_t1, set_t2 in combinations(to_compare, 2):
                    comp_avg += len(set_t1.intersection(set_t2)) / 5.
                    nb_elements += 1
                comp_avg /= nb_elements

                # print('t',t,'lim',llim,rlim)

                #Compare is an array for each timepoint comparing nbs of every timepoint w/in the window
                #Returns percent similar for each comparison, duplication occurs
                self.win_cons[i,t] = comp_avg

    def graphWinCons(self,labels=True,save=False,name=None,path=None):
        '''Plots window of consistency data in black with average in red'''

        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)

        tracks = self.track_ids

        for i,trackid in enumerate(tracks):

            track = self.win_cons[i]

            ax.plot(track,c='k', alpha=.4)


        ax.plot(np.median(self.win_cons, axis = 0), c='r', lw = 3)

        if labels == True:
            ax.set_ylabel('Neighborhood Consistency Score')
            ax.set_xlabel('Time')

        if labels==False:
            plt.setp(ax.get_yticklabels(), visible = False)
            plt.setp(ax.get_xticklabels(), visible = False)

        cwd = os.getcwd()

        if save==True and name!=None:
            if path!=None:
                os.chdir(path)
            fig.savefig(name,dpi=1200,bbox_inches='tight',pad_inches=0)

        os.chdir(cwd)

    ### Functions not in use ###

    def graphNeighbors(self,stime,etime,save=False,name=None,path=None):
        '''Graph tracks with unique colors in 3 individual planes at two timepoints'''
        
        cwd = os.getcwd()
        
        Lspos, Lepos, Lcolor  = [], [], []
        
        for trackid in self.tinput:

            track = self.tinput[trackid]
            
            Lspos.append(track[0][stime])
            Lepos.append(track[0][etime])
            Lcolor.append(self.track_color[trackid])
            
        spos = np.array(Lspos)
        epos = np.array(Lepos)
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
            filename = 'NB'+str(stime)+'-'+str(etime)+name+'.jpg'
            fig.savefig(filename, dpi = 1200,bbox_inches='tight',pad_inches=0)
            
        os.chdir(cwd)


    def findNearestNBs(self,time,n):
        '''Find n nearest neighbors for each track at specificed timepoint
        Returns: nb_ids: dictionary of list of neighbors for each track'''
        
        self.n = n
        ntracks = len(self.track_ids)
        
        eucdist = self.calculateEucDist(time)
        eucdist[eucdist == 0] = np.nan
#         print('eucdist',eucdist)
        
#         nb = np.zeros((ntracks,n))
        nb_index = {}
        
        for i,tid in enumerate(self.track_ids):
            a = eucdist[i]
            minI = np.argpartition(a,n)[:n]
            
            nb_index[tid] = minI
            
        nb_ids = {}
        
        for trackid in self.track_ids:
            tracks = self.track_ids[nb_index[trackid]]
            nb_ids[trackid] = tracks
            
        return nb_ids

    def findNBArrayFlex(self,n):
        '''Find all nearest neighbors with flex for n+1 and put into single array
        Returns: self.nbarray'''
        
        ntracks = len(self.track_ids)
        
        self.nbarray = np.zeros((len(self.track_ids),len(self.timepoints),n+1))
        
        for i,time in enumerate(self.timepoints):
            
            eucdist = self.calculateEucDist(time)
            eucdist[eucdist == 0] = np.nan
            
            for j,trackid in enumerate(self.track_ids):
                
                tdist = eucdist[j]
                min_index = np.argpartition(tdist,n+1)[:n+1]
                min_trackids = self.track_ids[min_index]
                
                if i > 0:
                    last = min_trackids[-1]
                    ptp = list(self.nbarray[j,i,:])
                    
                    if last not in ptp:
                        min_trackids[-1] = -1
                
                else:
                    min_trackids[-1] = -1
                        
                self.nbarray[j,i,:] = min_trackids

    def findAllNeighbors(self,n):
        '''Find nearest neighbors for all timepoints
        Returns: self.nbdata, self.nbarray'''
        
        self.nbdata = []
        
        for time in self.timepoints:
            nbs = self.findNearestNBs(time,n)
            self.nbdata.append(nbs)
            
        self.nbarray = np.zeros((len(self.track_ids),len(self.timepoints),n))
        
        for i,tpdata in enumerate(self.nbdata):
            
            for j,trackid in enumerate(self.track_ids):
                
                nbs = tpdata[trackid]
                self.nbarray[j,i,:] = nbs

    def graphNBImage(self,time,plane,save=False,name=None,path=None):
        '''Graph neighbor data on a single time point for a single plane'''
        
        cwd = os.getcwd()
        
        filename = 'Image Data\\' + plane + '\\MAX_' + str(time) + '_3D.tif'
        
        im = Image.open(filename)

        Lpos, Lcolor = [], []

        for trackid in self.tinput:

            track = self.tinput[trackid]

            Lpos.append(track[0][time])
            Lcolor.append(self.track_color[trackid])

        pos = np.array(Lpos)
        
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)

        ax.imshow(im)

        if plane == 'XY':
            x = 0
            plt.xlabel('X - AP')
            y = 1
            plt.ylabel('Y - DV')
        if plane == 'YZ':
            x = 1
            plt.ylabel('Y - DV')
            y = 2
            plt.ylabel('Z - ML')
        if plane == 'XZ':
            x = 0
            plt.xlabel('X - AP')
            y = 2
            plt.ylabel('Z - ML')

        ax.scatter(pos[:,x],pos[:,y],c=Lcolor,s=40)
        
        xmax = max(pos[:,x]) + 20
        xmin = min(pos[:,x]) - 20
        ymax = max(pos[:,y]) + 20
        ymin = min(pos[:,y]) - 20
        
        size = im.size
        
        plt.xlim(0,size[0]/2)
        plt.ylim(600,100)
        
        ax.set_title('Neighbors ' + plane + ' t=' + str(time))
        
#         plt.show()
        
        if save==True and name!=None:
            if path != None:
                os.chdir(path)
                
            filename = 'NB_' + plane + '_' + str(time) + name + '.jpg'
            fig.savefig(filename, dpi = 1200,bbox_inches='tight',pad_inches=0)
                
        os.chdir(cwd)

    def tSimilarity(self,threshold):
        '''Find the number of neighbors in a timepoint similar to the previous timepoint
        Return: self.simpertime, self.consistent_t'''

        arrayshape = np.shape(self.nbarray)

        self.simpertime = np.zeros((arrayshape[0],arrayshape[1]))

        for i, trackid in enumerate(self.track_ids):

            tnbs = self.nbarray[i]

            for t,nbs in enumerate(tnbs):

                if t>0:
                    count=0
                    for nb in nbs:
                        if nb in tnbs[t-1]:
                            count = count + 1 
                    sim[i,t] = count

        self.consistent_t = []

        for i, trackid in enumerate(self.track_ids):

            tnbs = sim[i]

            consistent = tnbs >= threshold
            sconsistent = consistent.sum()
            self.consistent_t.append(sconsistent)
            
    def nbConsistency(self,threshold):
        '''Computes different metrics of how consistent a neighbor is over time
        Returns: self.cons_avg, self.num_nbs, self.nbs_const'''

        self.cons_avg = [] #outscore = avgscore, divide by total num timepoints to normalize
        self.num_nbs = [] #outlength
        self.nbs_const = [] #Num of consistent time points per nb

        for i,trackid in enumerate(self.track_ids):

            tnbarray = self.nbarray[i]
            tnbs = list(set(tnbarray.flatten()))

            pres_nbs = np.zeros((len(tnbs),len(self.timepoints)))

            for i, nb in enumerate(tnbs):

                status = tnbarray == nb
                fltstatus = np.zeros((len(self.timepoints)))

                for j,time in enumerate(self.timepoints):

                    tstatus = status[j]
                    if True in tstatus:
                        fltstatus[j] = 1
                    else:
                        flststatus[j] = 0

                pres_nbs[i,:] = np.array(fltstatus)

            sumscore = np.sum(pres_nbs,axis=1)
            avgscore = np.average(sumscore)/len(self.timepoints)
            self.cons_avg.append(avgscore)

            self.num_nbs.append(len(tnbs))

            self.nbs_const.append(sumscore)
            
    def consistencyRatio(self,threshold):
        '''Calculate ratios of interloper:consistent and percent consistent
        Returns: self.i_c_ratio, self.i_c_normratio, self.c_perc'''

        shape = np.shape(self.nbs_const)

        self.i_c_ratio = []
        self.i_c_normratio = []
        self.c_perc = []

        for i in range(shape[0]):

            nbs = outraw[i]

            consistent = len(nbs[nbs >= threshold])
            interloper = len(nbs[nbs < threshold])

            ratio = interloper/consistent
            normratio = ratio/len(nbs)
            perc = consistent/len(nbs)

            self.i_c_ratio.append(ratio)
            self.i_c_normratio.append(normratio)
            self.c_perc.append(perc)

    def nbSimilarity(self,a,b):
        '''Calculates percent similarity between two sets of neighbors'''

        count = 0

        for nb in a:

            if nb in b:
                count = count + 1

        perc = count/len(a)

        return perc
