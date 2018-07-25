
import copy
import random
import numpy as np
import kaleido as kld

##### BATCH
class Batch:

    data , type = [] , None

    ### __INIT__
    def __init__( self , data = None , batch_size = 1 , multiple = False ):

        if not multiple: data = [ data ]
        if kld.chk.is_seq( data[0] ): self.type = 'seq'
        if kld.chk.is_numpy( data[0] ): self.type = 'numpy'

        self.multiple , self.data = multiple , data
        self.rnd = np.arange( self.num_files() )
        self.set_batch_size( batch_size )
        self.i = 0

    ### PRINT
    def print( self ):
        print( self.data )

    ### RESET
    def reset( self , shuffle = False ):
        if shuffle: random.shuffle( self.rnd )
        self.i = 0

    ### SHAPE
    def shape( self , n = 0 , with_none = False ):
        shape = list( self.data[n][0].shape )
        if with_none: shape = [ None ] + shape
        return shape

    ### COPY
    def copy( self , n = None ):
        if n is None: return Batch( copy.deepcopy( self.data ) , self.batch_size() , self.multiple )
        else: return Batch( copy.deepcopy( self.data[n] ) , self.batch_size() , False )

    ### IDX SAMPLE
    def idx_sample( self , n , st ):
        if self.num_files() - st <= n:
            return np.arange( st , self.num_files() )
        n , idx = float( self.num_files() - st + 1 ) / float( n ) , []
        for i in range( self.num_files() - st ):
            m = int( i * n ) + st
            if m >= self.num_files(): break
            if len( idx ) == 0 or m is not idx[-1]: idx.append( m )
        return np.array( idx )

    ### SLICE DATA
    def slice_data( self , idx , n = None ):
        if self.type is 'seq':  return [ self.data[n][i] for i in idx ]
        if self.type is 'numpy': return self.data[n][idx]

    ### KEEP FIRST
    def keep_first( self , n ):
        self.data = [ self.data[i][:n] for i in range( len( self.data ) ) ]
        self.rnd = np.arange( self.num_files() )
        self.set_num_batches()

    ### KEEP SKIP
    def keep_skip( self , n , st = 0 ):
        self.data = [ self.data[i][st::n] for i in range( len( self.data ) ) ]
        self.rnd = np.arange( self.num_files() )
        self.set_num_batches()

    ### KEEP SAMPLE
    def keep_sample( self , n , st = 0 ):
        idx = self.idx_sample( n , st )
        self.data = [ self.slice_data( idx , i ) for i in range( len( self.data ) ) ]
        self.rnd = np.arange( self.num_files() )
        self.set_num_batches()

    ### COPY FIRST
    def copy_first( self , n ):
        new = self.copy() ; new.keep_first( n ) ; return new

    ### COPY SKIP
    def copy_skip( self , n , st = 0 ):
        new = self.copy() ; new.keep_skip( n , st ) ; return new

    ### COPY SAMPLE
    def copy_sample( self , n , st = 0 ):
        new = self.copy() ; new.keep_sample( n , st ) ; return new

    ### GETTERS
    def num_files( self ):
        if self.type == 'seq': return len( self.data[0] )
        if self.type == 'numpy': return self.data[0].shape[0]

    def num_batches( self ):      return self.n
    def batch_size( self ):       return self.b
    def files( self ):            return self.data if self.multiple else self.data[0]
    def file( self , n , d = 0 ): return self.data[d][n]
    def __getitem__( self , d ):  return self.data[d]
    def __setitem__( self , d , v ): self.data[d] = v

    def first( self , n ):
        if not self.multiple: return self.data[0][:n]
        else: return [ self.data[i][:n] for i in range( len( self.data ) ) ]

    def skip( self , n , st = 0 ):
        if not self.multiple: return self.data[0][st::n]
        else: return [ self.data[i][st::n] for i in range( len( self.data ) ) ]

    def sample( self , n , st = 0 ):
        idx = self.idx_sample( n , st )
        if not self.multiple: return self.slice_data( idx , 0 )
        else: return [ self.slice_data( idx , i ) for i in range( len( self.data ) ) ]

    ### SETTERS
    def set_batch_size( self , b ):
        self.b = b ; self.set_num_batches()
        return self.num_batches()
    def set_num_batches( self ):
        self.n = int( self.num_files() / self.batch_size() )

    ### NEXT_BATCH
    def next_batch( self ):
        rnd = self.rnd[ self.i : self.i + self.b ]
        files = [ self.slice_data( rnd , i ) for i in range( len( self.data ) ) ]
        self.i += self.b
        return files if self.multiple else files[0]






