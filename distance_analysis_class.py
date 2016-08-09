import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import os
from copy import copy


class distAnalysis:
    '''Compares tracks based on the type of movement and clusters accordingly'''
    
    def __init__(self,tinput):
        '''Constructor, calculates distance between tracks and average derivative for each pair'''
        
        self.tracks = tinput
        idkeys = self.tracks.keys()
        self.track_ids = np.array([k for k in idkeys])
        
        self.calculateDist()
        self.distVarianceSlope()
        
    def eucDist(self, a_id, b_id, tracksource):
        '''Calculate the euclidean distance between two tracks excluding timepoints that
        do not have data for both tracks'''
        
        null = np.zeros((3))
        
        a = tracksource[a_id][0]
        b = tracksource[b_id][0]
        
        #timepoints = np.all([a != null, b != null])
        #print(timepoints)
        
        dist = np.array([])
        
        for p_a,p_b in zip(a,b):
            if np.count_nonzero(p_a) == 0 or np.count_nonzero(p_b) == 0:
                p_dist = 0
            else:
                p_dist = np.sqrt(np.sum((p_a-p_b)**2))
                
            dist = np.append(dist,p_dist)
            
        return(dist)
        
    def calculateDist(self):
        '''Calculate eucDist for all possible pairs of tracks
        Returns: self.alldist: square matrix with distance between every pair of tracks'''
        
        self.trackids = list(self.tracks.keys())
#         print(len(trackids))
        
        self.alldist = np.zeros((len(self.trackids),len(self.trackids), 501))
        #print(alldist)
        
        for i,track in enumerate(self.trackids):
            
            for j,versus in enumerate(self.trackids):
                #print(i,j)
                dist = self.eucDist(track,versus,self.tracks)
                self.alldist[i,j] = dist
                
    def slope(self,i,dist):
        '''Calculate average slope using two points around central point in 2D data'''

        if i == 0:
            a = dist[i]
        else:
            a = dist[i-1]
        if i >= np.size(dist) - 2:
            b = dist[i]
        else:
            b = dist[i+1]

        slope = np.abs((b-a)/((i+1)-(i-1)))

        return slope
    
    def distChange(self,Ldist):
        '''Calculate average derivative for a pair of tracks based on list of distance between tracks'''
        
        #print('Ldist', Ldist)

        nonzero = Ldist[~np.isnan(Ldist)] #Actually nonnan

        deriv = np.zeros(np.size(nonzero))
        
        #print('nonzero', nonzero)

        for i,dist in enumerate(nonzero):
            
            #print(i)

            pslope = self.slope(i,nonzero)

            deriv[i] = pslope
            
        #print(deriv)

        avg = np.nanmean(deriv)

        self.deriv = deriv

        return avg
                
    def distVarianceSlope(self):
        '''Calculate average derivative of distance between tracks for all tracks
        Returns: self.varslope: square matrix with average derivative of distance between all pairs of tracks'''
        
        self.varslope = np.zeros((len(self.trackids),len(self.trackids)))
        self.varslope_dict = {}
        for i,track in enumerate(self.alldist):
            
            for j,versus in enumerate(track):
                
                #print(versus)
                #print(i,j)
                var = self.distChange(versus)
                self.varslope[i, j] = var
#                 self.varslope_dict
        
#         np.mean(self.alldist, axis = 2)
#         da = (a[2:] - a[:-2])/2.
#         return self.varslope

    def graphVarslope(self,labels=True,save=False,name=None,outpath=None):
        '''Graph histogram of varslope for data distribution with option to save'''
        
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        varslope_nonan = self.varslope[~np.isnan(self.varslope)]
        ax.hist(varslope_nonan, bins = 50)

        if labels==True:
            ax.set_ylabel('Number of Track Pairs')
            ax.set_xlabel('Average derivative of the distance between tracks')

        if labels == False:
            plt.setp(ax.get_xticklabels(), visible = False)
            plt.setp(ax.get_yticklabels(), visible = False)

        cwd = os.getcwd()
        
        if save == True and name != None:
            if outpath!=None:
                os.chdir(outpath)
            fig.savefig(name, dpi = 1200,bbox_inches='tight',pad_inches=0)

        os.chdir(cwd)

    def clusterVarslope(self,labels=True,save=False,name=None,outpath=None):
        '''Cluster varslope data and run distance calculations
        Returns: self.L: linkage data'''
        
        data_for_clustering = copy(self.varslope)
        data_for_clustering[np.isnan(self.varslope)] = 4.5

        self.L = linkage(data_for_clustering, method='ward')
        fig = plt.figure(figsize=(10, 8))#,num='Clustering Based on Variance of Distance Between Tracks')
        ax = fig.add_subplot(1,1,1)
        Z = dendrogram(self.L, ax = ax, labels = self.trackids)

        if labels == True:
            ax.set_xlabel('Track IDs')
            ax.set_ylabel('Distance between nodes [um]')
        else:
            plt.setp(ax.get_xticklabels(), visible = False)
            plt.setp(ax.get_yticklabels(), visible = False)

        cwd = os.getcwd()
        
        if save == True:
            if outpath!=None:
                os.chdir(outpath)
            fig.savefig(name, dpi = 1200,bbox_inches='tight',pad_inches=0)

        os.chdir(cwd)

    def colorCluster(self,cth,labels=True,save=False,name=None,outpath=None):
        '''Apply color threshold to dendrogram based on threshold in cth'''

        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
        Z = dendrogram(self.L,ax=ax,labels=self.trackids,color_threshold=cth)

        if labels == True:
            ax.set_xlabel('Track IDs')
            ax.set_ylabel('Distance between nodes [um]')
        else:
            plt.setp(ax.get_xticklabels(), visible = False)
            plt.setp(ax.get_yticklabels(), visible = False)

        cwd = os.getcwd()

        if save==True and name!=None:
            if outpath!=None:
                os.chdir(outpath)
            fig.savefig(name,dpi=1200,bbox_inches='tight',pad_inches=0)

        os.chdir(cwd)
            
    def extractClusters(self,numclust):
        '''Extract clusters from linkage analysis based on desired number of clusters
        Returns: self.clusters: dictionary with trackids for each cluster'''
        
        tmp = fcluster(self.L, numclust, 'maxclust')
        self.clusters = dict()
        for i, k in enumerate(tmp):
            self.clusters.setdefault(k, []).append(self.trackids[i])
        print (self.clusters)
            
    def clusterAverage(self,save=False,name=None):
        '''Calculate average position of the cluster at each timepoint based on tracks in the cluster
        Returns: self.clustavg: dictionary with average cluster position array for each cluster'''
        
        self.clustavg = {}
        
        for cluster in self.clusters:
            clust = {}
            for trackid in self.clusters[cluster]:
                clust[trackid] = self.tracks[trackid]
            tmp = np.array([v[0] for v in clust.values()])
    
            avg = np.average(tmp, axis=0)
            self.clustavg[cluster] = avg
            
    def graphClusterAvg(self,Dcolor,labels=True,save=False,name=None,outpath=None):
        '''Graph the average plot of each cluster according to input color (Dcolor)'''

        self.clusterAverage()

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(311)
        ay = fig.add_subplot(312)
        az = fig.add_subplot(313)

        N = len(self.clusters.keys())

        for c in range(1,N+1):

            track = self.clustavg[c]

            ax.plot(track[:,0],c=Dcolor[c])
            ay.plot(track[:,1],c=Dcolor[c])
            az.plot(track[:,2],c=Dcolor[c])

        for splt in [ax,ay,az]:
            splt.set_ylim([150,600])

        fig.subplots_adjust(hspace=0.1)

        if labels == True:
            ax.set_ylabel('AP')
            ay.set_ylabel('DV')
            az.set_ylabel('ML')
            az.set_xlabel('Time')

        if labels == False:
            for splt in [ax,ay,az]:
                plt.setp(splt.get_yticklabels(),visible=False)
                plt.setp(splt.get_xticklabels(),visible=False)

        cwd = os.getcwd()

        if save == True and name!=None:
            if outpath!=None:
                os.chdir(outpath)
            fig.savefig(name,dpi=1200,bbox_inches='tight',pad_inches=0)

        os.chdir(cwd)


    ### Functions not in use ###

    def posGraphv2(self,stime,etime,save=False,name=None,path=None):
        '''Generate position xyz graphs for 2 timepoints with color by cluster'''
        
        cwd = os.getcwd()
        
        clcolor = ['r','g','b','y','c']
        
        Lspos, Lepos, Lcolor  = [], [], []
        
        for i,cls in enumerate(self.clusters):
            
            for trackid in self.clusters[cls]:

                track = self.tracks[trackid]

                Lspos.append(track[0][stime])
                Lepos.append(track[0][etime])
                Lcolor.append(clcolor[i])
            
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
            filename = 'PF'+str(stime)+'-'+str(etime)+name+'.jpg'
            fig.savefig(filename, dpi = 1200,bbox_inches='tight',pad_inches=0)
            
        os.chdir(cwd)