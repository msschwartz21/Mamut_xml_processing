import numpy as np

class velocityAnalysis:
    '''Calculates velocity for tracks'''
    
    def __init__(self,tinput):
        '''Constructor'''
        
        self.tracks = tinput
        
    def calculateVelocity(self):
        '''Calculate velocity for all tracks in the dataset
        Returns: self.velocity'''
        
        self.velocity = {}
        
        for trackid in self.tracks:
            pos = self.tracks[trackid][0]
            dpos = (pos[2:]-pos[:-2])/2
            self.velocity[trackid] = dpos
            
        #This type of velocity calculation doesn't calculate the 3d slope just the x,y,z slope individually
        
    def velocityNorm(self):
        '''Calculate norm of velocity vectors to represent the speed not direction
        Returns: self.velocity_norm'''
        
        self.velocity_norm = {}
        
        for trackid in self.velocity:
            velocity = self.velocity[trackid]
#             print(velocity)
            norm =  linalg.norm(velocity, axis=1)
            self.velocity_norm[trackid] = norm
            
    def velocityUnit(self):
        '''Calculate unit vector of each velocity vector using the calculated norm
        Returns: self.velocity_unit'''
        
        self.velocity_unit = {}
        
        for trackid in self.velocity:
            velocity = self.velocity[trackid]
            norm = self.velocity_norm[trackid]
            
            unit = np.zeros((499,3))
            
            unit[:,0] = velocity[:,0]/norm
            unit[:,1] = velocity[:,1]/norm
            unit[:,2] = velocity[:,2]/norm
            
            self.velocity_unit[trackid] = unit