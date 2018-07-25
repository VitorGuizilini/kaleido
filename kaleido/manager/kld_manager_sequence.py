
import random
import numpy as np
import kaleido as kld

##### SEQUENCE
class Sequence:

    ### __INIT__
    def __init__( self , data = None , bck = None , fwd = None ):
        self.data = data
        self.bck , self.fwd = bck , fwd

    ### __GETITEM__
    def __getitem__( self , n ):
        return self.data[n]

    ### SIZE
    def size( self , n = None ):
        if n is None: return len( self.data )
        else: return self.data[n].shape[0] - self.bck - self.fwd + 1

    ### BATCH
    def batch( self , batch = 1 ):
        return kld.manager.Batch( self.split() , multiple = self.fwd > 0 , batch = batch )

    ### INS
    def ins( self , i = None , j = None ):
        if i is None:
            lst = []
            for i in range( self.size() ):
                lst.append( self.ins(i) )
            lst = np.array( lst )
            return np.reshape( lst , [ lst.shape[0] * lst.shape[1] , lst.shape[2] , lst.shape[3] ] )
        if j is None:
            lst = []
            for j in range( self.size( i ) ):
                lst.append( self.ins( i , j ) )
            return np.array( lst )
        n = self.size( i )
        return self.data[i][ j : j + self.bck ]

    ### OUTS
    def outs( self , i = None , j = None ):
        if i is None:
            lst = []
            for i in range( self.size() ):
                lst.append( self.outs(i) )
            lst = np.array( lst )
            return np.reshape( lst , [ lst.shape[0] * lst.shape[1] , lst.shape[2] , lst.shape[3] ] )
        if j is None:
            lst = []
            for j in range( self.size( i ) ):
                lst.append( self.outs( i , j ) )
            return np.array( lst )
        n = self.size( i )
        return self.data[i][ j + self.bck : j + self.bck + self.fwd ]

    ### PAIRS
    def pairs( self , i = None , j = None ):
        return self.ins(i,j) , self.outs(i,j)

    ### SPLIT
    def split( self ):

        bck , fwd = self.bck , self.fwd
        zbck , zfwd = [] , []

        for i in range( self.size() ):
            for j in range( self.size(i) ):
                zbck.append( self.data[i][ j : j + bck ] )
                if fwd > 0: zfwd.append( self.data[i][ j + bck : j + bck + fwd ] )
        if fwd == 0: return np.array( zbck )
        else: return np.array( zbck ) , np.array( zfwd )






