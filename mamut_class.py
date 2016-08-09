import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.spatial import distance
import matplotlib.lines as mlines
from copy import copy

class mamut:
    '''Parses mamut xml files and process tracks to prepare for analysis'''
    
    def __init__(self,filepath):
        '''Parses xml file and produces dictionaries of track data in several formats'''
        
        self.filedate = filepath[-8:-4]
        
        self.readMamutXml(filepath)
        
        self.trackFromEnd()
        self.createArrayTracks()
        
        self.track_gcom = {}
        self.createCompleteTracks(self.track_arrays,self.track_gcom)
        self.track_gcomsm = {}
        self.smoothAllTracks(self.track_gcom, self.track_gcomsm, 5)
        
        self.track_lcom = {}
        self.createLocalTracks(self.track_gcom, self.track_lcom)
        self.track_lcomsm = {}
        self.smoothAllTracks(self.track_lcom, self.track_lcomsm, 5)
        
        self.findLineages(self.track_gcom)
        self.nodesInLineages(self.track_gcom)
        
    '''#####Data Import#####'''

    def readMamutXml(self,filepath):
        '''Parse XML file and save data as class -ide variable
        Results: time_nodes, nodes, node_pos, node_time, node_channel, 
        tracks, track_time, track_edges, edges, successor, predecessor'''

        tree = ET.parse(filepath)
        model = tree.getroot()[0]

        FeatureDeclarations, AllSpots, AllTracks, FilteredTracks = list(model)

        #Dictionary of nodes at each timepoint
        self.time_nodes = {}
        #List of all nodes
        self.nodes = []
        #Dictionary of position of nodes
        self.node_pos = {}
        #Dicitonary of node timepoint
        self.node_time = {}
        self.node_channel = {}

        #Iterate through all timepoints containing spots
        for timepoint in AllSpots:
            t = int(timepoint.attrib['frame'])
            #Initialize list for each t for cell IDs
            self.time_nodes[t] = []

            #Iterate through all nodes in a timepoint
            for node in timepoint:
                node_id, x, y, z, ch = (int(node.attrib['ID']),
                                    float(node.attrib['POSITION_X']),
                                    float(node.attrib['POSITION_Y']),
                                    float(node.attrib['POSITION_Z']),
                                    int(node.attrib['SOURCE_ID']))
                self.time_nodes[t].append(node_id)
                self.nodes.append(node_id)
                #Correct from list to numpy array once installed
                self.node_pos[node_id] = np.array([x,y,z])
                self.node_time[node_id] = t
                self.node_channel[node_id] = ch

        #List of tracks
        self.tracks = []
        #Dictionary of tracks with start and stop times as strings
        self.track_time = {}
        #Dictionary of all edges in a track
        self.track_edges = {}
        #List of all edges
        self.edges = []
        #Dictionary of successors for every node
        self.successor = {}
        #Dictionary of predecessors for every node
        self.predecessor = {}

        #Iterate through all tracks
        for track in AllTracks:
            track_id,start,stop = (int(track.attrib['TRACK_ID']),
                                   float(track.attrib['TRACK_START']),
                                   float(track.attrib['TRACK_STOP']))
            self.tracks.append(track_id)
            self.track_time[track_id] = (start,stop)
            self.track_edges[track_id] = []

            #Iterate through edges in a track
            for edge in track:
                s,t = ((int(edge.attrib['SPOT_SOURCE_ID'])),
                       (int(edge.attrib['SPOT_TARGET_ID'])))
                #Check that nodes exist in list
                if s in self.nodes and t in self.nodes:
                    #Correct so that connection is going forward in time
                    if self.node_time[s] > self.node_time[t]:
                        s,t = t,s
                    self.track_edges[track_id].append((s,t))
                    self.edges.append((s,t))
                    self.successor.setdefault(s, []).append(t)
                    self.predecessor.setdefault(t, []).append(s)
    
    '''#####Track Generation#####'''
    
    def findEndNodes(self):
        '''Identify all cells that do not have a successor and are end of track
        Results: node_end: list of node ids'''
        
        self.node_end = []
        self.node_endubi = []
        key_successor = self.successor.keys()
        
        for node in self.nodes:
            if node not in key_successor:
                self.node_end.append(node)
                
        return self.node_end
        
    def trackFromEnd(self):
        '''Generate tracks based on end nodes and predecessor data, chronologically backward
        Returns: self.track_ends'''
        
        node_end = self.findEndNodes()
        
        self.track_ends = {}
        key_predecessor = self.predecessor.keys()
        
        for track in node_end:
            name = track
            self.track_ends[name] = []
            node = track
            
            while node in key_predecessor:
                self.track_ends[name].append(node)
                L_node = self.predecessor[node]
                if len(L_node) > 1:
                    print('Node', node, 'in track', name, 'has more than one predecessor')
                    break
                else:
                    node = L_node[0]
            self.track_ends[name].append(node)
                    
    def createArrayTracks(self):
        '''Uses node_end data to compile position data for each track
        Result: self.track_arrays: dictionary with 2 arrays (position, ID) for each track, 
        self.node_track: dictionary with trackid for each node'''
        
        track_keys = self.track_ends.keys()
        self.track_arrays = {}
        self.node_track = {}
        
        for key in track_keys:
            postrack = np.zeros((501,3))
            postrack[postrack==0] = np.nan
            tidtrack = np.zeros((501))
            nodetrack = self.track_ends[key]
            
            for node in nodetrack:
            
                t = int(self.node_time[node])
                pos = self.node_pos[node]
                    
                postrack[t] = pos
                tidtrack[t] = node
                self.node_track[node] = key
                
            self.track_arrays[key] = [postrack, tidtrack]
            
    def createLocalTracks(self,tinput,tout):
        '''Calculates local track movement by removing average trajectory from each track
        Result: dictionary of position array for each track in tout'''
        
        tmp = np.array([v[0] for v in tinput.values()])
        tmp[tmp==0]=np.nan

        avg_trajectory = np.nanmean(tmp, axis=0)
        
        for trackid in tinput:
            track = tinput[trackid][0]
            nodes = tinput[trackid][1]
            ltrack = track - avg_trajectory
            
            tout[trackid] = [ltrack,nodes]
            
    def trackLength(self,trackid):
        '''Return track length based on track data in self.track_arrays'''
        
        track = self.track_arrays[trackid][0]
        nonan = track[~np.isnan(track)]
        tlen = len(nonan)/3
        
        return(tlen)
    
    def createCompleteTracks(self,tinput,tout):
        '''Create dictionary of only complete tracks from tinput dataset
        Result: dictionary saved in output with only complete tracks with position data arrays'''
        
        for trackid in tinput:
            tlen = self.trackLength(trackid)
            if tlen >= 500:
                tout[trackid] = tinput[trackid]
                
    def smoothTrack(self,trackid,tinput,size):
        '''Smooth track along specified window twice as big as size input
        Result: array of position data for a single smoothed track'''
        
        track = tinput[trackid][0]
        
        tlen = len(track[:,0])
        
        x,y,z = track[:,0], track[:,1], track[:,2]
        
        sout = [np.zeros((tlen)), np.zeros((tlen)), np.zeros((tlen))]
        
        for j,dim in enumerate([x,y,z]):
            for i,pos in enumerate(dim):
                l = i - size
                if l < 0:
                    l = 0
                r = i + size
                if r > tlen:
                    r = tlen
                    
                avg = np.average(dim[l:r])
                sout[j][i] = avg
                
        strack = np.zeros((tlen,3))

        strack[:,0] = sout[0]
        strack[:,1] = sout[1]
        strack[:,2] = sout[2]
        
        return strack
    
    def smoothAllTracks(self,tinput,tout,size):
        '''Smooth all tracks in a dictionary and add to empty output dictionary'''
        
        for trackid in tinput:
            strack = self.smoothTrack(trackid,tinput,size)
            tout[trackid] = [strack,tinput[trackid][1]]
            
    '''#####Cell Divisions#####'''
    
    def findNodeInTrack(self,node,trackinput):
        '''Return trackid that contains the node in question'''
        
        tracks = []
        for trackid in trackinput:
                
            nodeids = trackinput[trackid][1]
#             print(nodeids)
            
            if node in nodeids:
                tracks.append(trackid)
                
        return(tracks)
    
    def findFirstNodes(self,trackinput):
        '''Return all starting nodes for lineages
        Result: self.nodes_start: list of node ids'''
        
        self.nodes_start = []
        
        for trackid in trackinput:
            
            tnodes = trackinput[trackid][1]
            snode = tnodes[0]
            
            self.nodes_start.append(snode)
    
    def findLineages(self,trackinput):
        '''Construct dictionary of all trackids of tracks in the descending lineage of a start node
        Result: self.lineages: dictionary of trackids per lineage, 
        self.track_lin: dictionary of lineageid per trackid'''
        
        self.lineages = {}
        self.track_lin = {}
        
        self.findFirstNodes(trackinput)
        
        for lin in self.nodes_start:
            tracks = self.findNodeInTrack(lin,trackinput)
            self.lineages[lin] = tracks
            
            for track in tracks:
                self.track_lin[track] = lin
            
    def nodesInLineages(self,trackinput):
        '''Construct dictionary with all nodes in a lineage for each node
        Result: self.node_lin: dictionary of lineage id for each node'''
        
        self.node_lin = {}
        
        for trackid in trackinput:
            
            tnodes = trackinput[trackid][1]
            lin = self.track_lin[trackid]
            
            for node in tnodes:
                self.node_lin[node] = lin
            
    def lineageColor(self,param,time,cmap):
        '''Assigns color to each lineage based on the average position of the lineage 
        in the dimension(param) and time specified
        Result: self.lineage_color, self.lineage_pos'''
        
        lins = list(self.lineages.keys())
        linpos = lins
        self.lineage_color = {}
        self.lineage_pos = {}
        
        for lin in linpos:
            
            tracks = self.lineages[lin]
            if 60020 in tracks:
                tracks.remove(60020)
            tpos = []
            
            for trackid in tracks:
                
                track = self.track_gcom[trackid][0]
                pos = track[time,param]
                tpos.append(pos)
                
            avgpos = np.average(tpos)
            
            self.lineage_pos[lin] = avgpos
            
        posvalues = self.lineage_pos.values()
        # print(posvalues)
        
        pmin = min(posvalues)
        # print(pmin)
        pmax = max(posvalues)
        # print(pmax)
        
        norm = mpl.colors.Normalize(vmin=pmin,vmax=pmax)
        m = cm.ScalarMappable(norm=norm,cmap=cmap)
        
        for lin in lins:
            pos = self.lineage_pos[lin]
            self.lineage_color[lin] = m.to_rgba(pos)
        
              
    '''#####Dataset Stats#####'''

    def sortByLength(self,trackinput):
        '''Sort tracks in list by length'''

        tracklensort = []
        tlens = []

        for i,trackid in enumerate(trackinput):
            track=trackinput[trackid][0]

            tlen = len(track[~np.isnan(track)])
            tlens.append(tlen)

            tracklensort.append((tlen,trackid))

        tracklensort.sort(key=lambda tup: tup[0])

        return tracklensort,tlens

    def graphTrackCovgGrad(self,trackinput,cmap,labels=True,save=False,name=None,outpath=None):
        '''Graph track length by sorted values using color gradient'''

        tracklens,tlens = self.sortByLength(trackinput)

        norm = mpl.colors.Normalize(vmin = min(tlens),vmax = max(tlens))
        m = plt.cm.ScalarMappable(norm=norm,cmap=cmap)

        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)

        for i,t_tuple in enumerate(tracklens):
            trackid = t_tuple[1]
            tlen = t_tuple[0]
            track=copy(trackinput[trackid][0][:,0])
            track[track>0] = 1

            ax.plot(i*track,c=m.to_rgba(tlen)) #m.to_rgba(tlen)

        if labels==True:
            ax.set_title('Temporal Coverage of Tracking Results')
            ax.set_ylabel('Tracks')
            ax.set_xlabel('Time')

        if labels==False:
            plt.setp(ax.get_yticklabels(), visible = False)
            plt.setp(ax.get_xticklabels(), visible = False)

        cwd = os.getcwd()

        if save==True and name != None:
            if outpath != None:
                os.chdir(outpath)
            fig.savefig(name,dpi=1200,bbox_inches='tight',pad_inches=0)

    def graphCellDivisions(self,labels=True,save=False,name=None,outpath=None):
        '''Creates histogram of number of cell divisions per lineage'''

        linsize = []
        for trackid in self.lineages:
            
            size = len(self.lineages[trackid])
            linsize.append(size)

        divisions = np.array(linsize) -1

        fig = plt.figure()
        ax = fig.add_subplot(111)

        bins = [-0.5,0.5,1.5,2.5,3.5,4.5]

        ax.hist(divisions,bins=bins)

        if labels==True:
            ax.set_title('Cell Divisions per Lineage')
            ax.set_xlabel('Number of Cell Divisions')
            ax.set_ylabel('Number of Lineages')

        if labels==False:
            plt.setp(ax.get_xticklabels(), visible = False)
            plt.setp(ax.get_yticklabels(), visible = False)

        cwd = os.getcwd()

        if save==True and name != None:
            if outpath != None:
                os.chdir(outpath)
            fig.savefig(name,dpi=1200,bbox_inches='tight',pad_inches=0)
        
        os.chdir(cwd)  

### Functions not in use ###

    def graphLineageDist(self,save=False):
        '''Histogram of # cells in lineage'''
        
        fig = plt.figure(num='Distibution of Cells in Lineage',figsize=(10,8))
        ax = fig.add_subplot(111)
        
        plt.title('Distibution of Cells in Lineage')
        
        lengths = []
        
        for linid in self.lineages:
            
            lin = self.lineages[linid]
            lenlin = len(lin)
            lengths.append(lenlin)
            
        ax.hist(lengths, bins=np.arange(0.5,max(lengths)+1))
        ax.set_xlabel('Number of Daughter Cells in Lineage')
        ax.set_ylabel('Number of Tracks')
        
        if save == True:
            name = 'LinSizeDist_' + self.filedate + '.jpg'
            fig.savefig(name, dpi = 1200,bbox_inches='tight',pad_inches=0)

    def graphTrackLengthDist(self,trackinput,save=False):
        '''Generate histogram showing distribution of track length in dataset'''
        
        fig = plt.figure(num='Track Length', figsize=(10,8))
        ax = fig.add_subplot(111)
        
        length = []
        
        for trackid in trackinput:
            track = trackinput[trackid][0]
            nonan = track[~np.isnan(track)]
#             print(nonan)
            tlen = len(nonan)/3
            length.append(tlen)
            
        ax.hist(length,bins=20)
        plt.title('Track Length Distribution')
        ax.set_xlabel('Track Length')
        ax.set_ylabel('Number of Tracks')
        
        if save == True:
            name = 'TrackLengthDist_' + self.filedate + '.jpg'
            fig.savefig(name, dpi = 1200,bbox_inches='tight',pad_inches=0)
            
    def colorByLength(self,tlen):
        '''Set color based on tracklength'''
        
        if tlen >= 0 and tlen < 100:
            c = 'r'
            n = '0-99'
        elif tlen >= 100 and tlen < 200:
            c = 'm'
            n = '100-199'
        elif tlen >= 200 and tlen < 300:
            c = 'y'
            n = '200-299'
        elif tlen >= 300 and tlen < 400:
            c = 'g'
            n = '300-399'
        elif tlen >= 400 and tlen < 500:
            c = 'c'
            n = '400-499'
        elif tlen >= 500:
            c = 'b'
            n = '500'
            
        return c, n
            
    def graphTrackCoverage(self,trackinput,save=False):
        '''Generate line graph that shows the temporal coverage for each track'''
        
        fig = plt.figure(num="Track Coverage",figsize=(10,8))
        ax = fig.add_subplot(111)
        
        for i,trackid in enumerate(trackinput):
            track = trackinput[trackid][0][:,0]
            track[track>0] = 1
            
            tlen = len(track[~np.isnan(track)])
#             print(tlen)
            color,n=self.colorByLength(tlen)
            
            ax.plot(i*track,c=color,label=n)
            
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
            
        r = mlines.Line2D([],[],color='r',label='0-99')
        m = mlines.Line2D([],[],color='m',label='100-199')
        y = mlines.Line2D([],[],color='y',label='200-299')
        g = mlines.Line2D([],[],color='g',label='300-399')
        c = mlines.Line2D([],[],color='c',label='400-499')
        b = mlines.Line2D([],[],color='b',label='500')
        
        plt.legend(handles=[r,m,y,g,c,b],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('Track Length Coverage')
        ax.set_xlabel('Time')
        ax.set_ylabel('Tracks')
        
        if save == True:
            name = 'TrackLengthCovg_' + self.filedate + '.jpg'
            fig.savefig(name, dpi = 1200,bbox_inches='tight',pad_inches=0)