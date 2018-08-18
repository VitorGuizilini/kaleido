
import copy
import random
import numpy as np
from kaleido.chk import *
import kaleido as kld

##### BATCH
class Batch:

    ### __INIT__
    def __init__( self , data = None , batch_size = 1 , multiple = False ):

        if not multiple: data = [ data ]
        self.multiple , self.data = multiple , data
        self.set_batch_size( batch_size )
        self.reset( shuffle = False )

    ### GETTERS
    def batch_size( self ):  return self.b
    def num_batches( self ): return self.n
    def size( self ): return len( self.data[0] )
    def width( self ): return len( self.data )
##    def all( self ):
##        return self.data[0] if not self.multiple else self.data
#    def entry( self , d , n = None ):
#        return self.data[d][n] if n is not None else self.data[d]
    def __getitem__( self , d ):
        if self.multiple: return self.data[d]
        else: return self.data[0][d]

    ### RANGES
    def range_batches( self ): return range( self.num_batches() )
    def range_size( self ): return range( self.size() )

    ### SETTERS
    def set_batch_size( self , b ):
        self.b = b ; self.set_num_batches()
        return self.num_batches()
    def set_num_batches( self ):
        self.n = int( self.size() / self.batch_size() )
    def __setitem__( self , d , v ): self.data[d] = v

    ### RESET
    def reset( self , shuffle = False ):
        if shuffle: random.shuffle( self.rnd )
        else: self.rnd = np.arange( self.size() )
        self.i = 0

    ### SHAPE
    def shape( self , d = 0 , with_none = False ):
        shape = list( self.data[d][0].shape )
        if with_none: shape = [ None ] + shape
        return shape

    ### SLICE DATA
    def slice_data( self , idx , n ):
        if is_lst( self.data[n] ): return [ self.data[n][i] for i in idx ]
        if is_npy( self.data[n] ): return self.data[n][idx]
        return None

    ### NEXT_BATCH
    def next_batch( self , b = None ):
        if b is None: b = self.b
        if self.i + b < self.size():
            rnd = self.rnd[ self.i : self.i + b ]
            self.i += b
        else:
            left = b - ( self.size() - self.i )
            rnd = np.concatenate( [ self.rnd[ self.i: ] , self.rnd[ :left ] ] )
            self.i = left
        files = [ self.slice_data( rnd , i ) for i in range( len( self.data ) ) ]
        return files if self.multiple else files[0]

    ### COPY
    def copy( self , n = None ):
        if n is None: return Batch( copy.deepcopy( self.data ) , self.batch_size() , self.multiple )
        else: return Batch( copy.deepcopy( self.data[n] ) , self.batch_size() , False )

    ### FIRST
    def keep_first( self , n ):
        if n == 0: return
        self.data = [ self.data[i][:n] for i in range( len( self.data ) ) ]
        self.rnd = np.arange( self.size() )
        self.set_num_batches()
    def copy_first( self , n ):
        new = self.copy() ; new.keep_first( n ) ; return new

    ### SKIP
    def keep_skip( self , n , st = 0 ):
        if n == 0: return
        self.data = [ self.data[i][st::n] for i in range( len( self.data ) ) ]
        self.rnd = np.arange( self.size() )
        self.set_num_batches()
    def copy_skip( self , n , st = 0 ):
        new = self.copy() ; new.keep_skip( n , st ) ; return new

    ### SAMPLE
    def keep_sample( self , n , st = 0 ):
        if n == 0: return
        idx = kld.aux.idx_sample( n , self.size() , st )
        self.data = [ self.slice_data( idx , i ) for i in range( len( self.data ) ) ]
        self.rnd = np.arange( self.size() )
        self.set_num_batches()
    def copy_sample( self , n , st = 0 ):
        new = self.copy() ; new.keep_sample( n , st ) ; return new

    ### GET FIRST
    def get_first( self , n ):
        if not self.multiple: return self.data[0][:n]
        else: return [ self.data[i][:n] for i in range( len( self.data ) ) ]

    ### GET SKIP
    def get_skip( self , n , st = 0 ):
        if not self.multiple: return self.data[0][st::n]
        else: return [ self.data[i][st::n] for i in range( len( self.data ) ) ]

    ### GET SAMPLE
    def get_sample( self , n , st = 0 ):
        idx = kld.aux.idx_sample( n , self.size() , st )
        if not self.multiple: return self.slice_data( idx , 0 )
        else: return [ self.slice_data( idx , i ) for i in range( len( self.data ) ) ]

##### MULTIBATCH
class MultiBatch( Batch ):
    def __init__( self , data = None , batch_size = 1 ):
        Batch.__init__( self , data , batch_size , True )




