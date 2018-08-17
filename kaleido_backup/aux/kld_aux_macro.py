
import numpy as np
from kaleido.chk import *
from kaleido.lst import *

### NDIG
def ndig( n ):
    return len( str( n ) )

### SHAPE
def shape( data ):
    shape = []
    while is_lst( data ):
        shape.append( len( data ) )
        data = data[0]
    if is_npy( data ):
        shape.append( data.shape )
    return tuple( flatten( shape ) )

### TO BOL
def to_bol( v ):
    if v in [True,'True','true',1,'1','T','t']: return True
    if v in [False,'False','false',0,'0','F','f']: return False
    return None

### VARTYPE
def vartype( v , vartype = None , short = False ):
    if vartype is None:
        if type( v ) is int:   return 'INT' if short else 'int'
        if type( v ) is float: return 'FLT' if short else 'float'
        if type( v ) is str:   return 'STR' if short else 'string'
        if type( v ) is bool:  return 'BOL' if short else 'bool'
        if type( v ) is list:  return 'LST' if short else 'list'
        return 'NOT'
    else:
        if vartype in ['INT','int']:    return int( v )
        if vartype in ['FLT','float']:  return float( v )
        if vartype in ['STR','string']: return str( v )
        if vartype in ['BOL','bool']:   return to_bol( v )
        if vartype in ['NOT',None]:     return None

### MERGE DICTS
def merge_dicts( *dicts ):
    return { k : v for d in dicts for k , v in d.items() }

### IDX SAMPLE
def idx_sample( n , t , st = 0 , fn = 0 ):
    n , t = n - 1 , t - st - fn
    r , m = t % n , int( t / n )
    idx = np.arange( 0 , t - r + 1 , m )
    if r > 0:
        add = np.ceil( np.arange( 1 , n , n / r ) ).astype( np.int32 )
        for a in add:
            idx[a:] += 1
    idx[1:] -= 1
    return idx + st

### ROUND
def round( x , thr = 0.5 ):
    if thr == 0.5: return np.round( x ).astype( np.int32 )
    y = np.zeros( x.shape , dtype = np.int32 )
    y[ x > thr ] = 1
    return y

### REMSPACE
def remspace( x , s = '' ):
    return str( x ).replace( ' ' , s )
