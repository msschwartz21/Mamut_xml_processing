import os

class readWriteDict:
    '''Basic class to take a dictionary and save it to a text file or read a text file and extract a dictionary'''
        
    def __init__(self,dtype,data,path=None):
        '''Constructor'''
        
        if dtype == 'file':
            self.filename = data
        elif dtype == 'dict':
            self.data = data
            
        if path != None:
            self.path = path
            
    def dictToText(self,filename,path=None):
        '''Save dictionary to text file to a directory if specified'''
        
        cwd = os.getcwd()
        
        if path != None:
            os.chdir(path)
        
        out = open(filename,'w')
        
        sdata = str(self.data)
        out.write(sdata)
        out.close()
        
        os.chdir(cwd)
        
    def textToDict(self,filename,path=None):
        '''Retrieve dictionary saved in txt file from directory if specified'''
        
        cwd = os.getcwd()
        
        if path != None:
            os.chdir(path)
            
        infile = open(filename, 'r')
        sdata = infile.read()
        self.data = eval(sdata)
        
        infile.close()
        os.chdir(cwd)