
import kaleido as kld
from kaleido.chk import *

### MAKE
def make( input ):
    if is_tup( input ): return list( input )
    elif is_lst( input ): return input
    else: return [ input ]

### FLATTEN
def flatten( input ):
    output = []
    for i in kld.rlen( input ):
        if is_seq( input[i] ):
            for item in flatten( input[i] ): output.append( item )
        else: output.append( input[i] )
    return output

### SAMPLE
def sample( input , n , st = 0 , fn = 0 ):
    if n == 0: return input
    if nested( input ):
        return [ sample( inp , n , st , fn ) for inp in input ]
    idx = kld.aux.idx_sample( n , len( input ) , st , fn )
    return [ input[i] for i in idx ]

### APPLY
def apply( input , func ):
    return list( map( func , make( input ) ) )

### TO STR
def to_str( input ):
    return apply( input , str )

### JOIN
def join( data , sep = '/' ):
    if not is_lst( data ): return data
    return data[0] if len( data ) == 1 else sep.join( data )

### MERGE STR
def merge_str( input , sep = '' ):
    return sep.join( to_str( input ) )

### ADD
def Add( a , b ):
    if is_lst( b ):
        for i in range( len( a ) ): a[i] += b[i]
    else:
        for i in range( len( a ) ): a[i] += b
def add( a , b ):
    if is_seq( b ): return [ a[i] + b[i] for i in range( len( a ) ) ]
    else:           return [ a[i] + b    for i in range( len( a ) ) ]

### SUB
def Sub( a , b ):
    if is_lst( b ):
        for i in range( len( a ) ): a[i] -= b[i]
    else:
        for i in range( len( a ) ): a[i] -= b
def sub( a , b ):
    if is_seq( b ): return [ a[i] - b[i] for i in range( len( a ) ) ]
    else:           return [ a[i] - b    for i in range( len( a ) ) ]

### MLT
def Mlt( a , b ):
    if is_lst( b ):
        for i in range( len( a ) ): a[i] *= b[i]
    else:
        for i in range( len( a ) ): a[i] *= b
def mlt( a , b ):
    if is_seq( b ): return [ a[i] * b[i] for i in range( len( a ) ) ]
    else:           return [ a[i] * b    for i in range( len( a ) ) ]

### DIV
def Div( a , b ):
    if is_lst( b ):
        for i in range( len( a ) ): a[i] /= b[i]
    else:
        for i in range( len( a ) ): a[i] /= b
def div( a , b ):
    if is_seq( b ): return [ a[i] / b[i] for i in range( len( a ) ) ]
    else:           return [ a[i] / b    for i in range( len( a ) ) ]

#### REP
#def rep( a , b ):
#    if not is_list( a ): return [ a ] * b
#    elif len( a ) == 1: return a * b
#    return a

#### SHAPE
#def shape( input , st , fn ):
#    return input[0].shape[st:fn] if is_seq( input ) else input.shape[st:fn]

