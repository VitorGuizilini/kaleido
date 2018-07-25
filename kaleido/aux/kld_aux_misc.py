
import os
import math
import random
import numpy as np
import kaleido as kld

### INF
def inf():
    return math.inf

### PROC SEQ
def proc_seq( xx , fn , *args ):
    if kld.chk.is_seq( xx ):
        return [ fn( x , *args ) for x in xx ]
    return fn( xx , *args )

#def run( x , fn , *args ):
#    return list( map( lambda x: fn( x , *args ) , x ) )


### COPY
def copy( input ):
    return input.copy() if kld.chk.is_list( input ) else input

def join( input , sep = '/' ):
    return input[0] if len( input ) == 1 else sep.join( input )

### MINMAX
def minmax( input ):
    return np.min( input ) , np.max( input )

### LENGTH
def length( num ):
    return len( str( num ) )

### VARTYPE
def vartype( v , vartype = None ):
    if vartype is None:
        if type( v ) is int:   return 'int'
        if type( v ) is float: return 'float'
        if type( v ) is str:   return 'string'
        if type( v ) is bool:  return 'bool'
    else:
        if vartype == 'int':   return int( v )
        if vartype == 'float': return float( v )
        if vartype == 'str':   return str( v )
        if vartype == 'bool':  return bool( v )

### MERGE
def merge( dicts ):
    return { k : v for d in dicts for k , v in d.items() }

### RANDF
def randf( min = 0.0 , max = 1.0 ):
    return min + random.random() * ( max - min )

### RANDI
def randi( min , max ):
    return random.randint( min , max )
