
import os
import math
import random
import kaleido as kld

### MAKE
def make( input ):
    return input if kld.chk.is_seq( input ) else [ input ]

### ADD
def add( input , n , v ):
    if input is not None and kld.chk.is_list( input ) and len( input ) > v:
        input[n] += v

### MULT
def mult( input , value ):
    if kld.chk.is_seq( value ):
        return [ input[i] * value[i] for i in range( len( input ) ) ]
    else:
        return [ input[i] * value for i in range( len( input ) ) ]

### REP
def rep( input , n ):
    if not kld.chk.is_list( input ): return [ input ] * n
    elif len( input ) == 1: return input * 2
    return input

### SHAPE
def shape( input , st , fn ):
    return input[0].shape[st:fn] if kld.chk.is_seq( input ) else input.shape[st:fn]

### FLATTEN
def flatten( input ):
    output = []
    for i in range( len( input ) ):
        if kld.chk.is_seq( input[i] ):
            for item in flatten( input[i] ): output.append( item )
        else: output.append( input[i] )
    return output

