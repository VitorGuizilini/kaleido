
from kaleido.chk import *
from kaleido.aux import *
import functools

### PARTIAL
def partial( fn , *args , **kwargs ):
    return functools.partial( fn , *args , **kwargs )

### PROCESS
def process( data , fn , *args , **kwargs ):
    if is_lst( data ): return [ fn( d , *args , **kwargs ) for d in data ]
    else: return fn( data , *args , **kwargs )

### DO OP
def do_op( data , op ):
    if not is_tup( op ): return op( data )
    elif len( op ) == 1: return op[0]( data )
    else:                return op[0]( data , **op[1] )

### DO OPS
def do_ops( data , ops ):
    if empty( ops ): return data
    if not is_lst( ops ):
        return do_op( data , ops )
    else:
        for op in ops:
            data = do_op( data , op )
        return data

### APPLY
def apply( data , ops , apply_type = 'ind' ):
    if empty( ops ): return data
    if nested( data ):
        if apply_type == 'ind':
            ndata = []
            for i in rlen( data ):
                ndata.append( [ do_ops( data[i][j] , ops ) for j in rlen( data , i ) ] )
        elif apply_type == 'col':
            ndata = []
            for i in rlen( data ):
                ndata.append( do_ops( data[i] , ops ) )
        elif apply_type == 'row':
            ndata = [ [] for i in rlen( data ) ]
            for i in rlen( data , 0 ):
                row = do_ops( [ data[j][i] for j in rlen( data ) ] , ops )
                for j in rlen( data ): ndata[j].append( row[j] )
        return ndata
    elif is_lst( data ):
        return [ do_ops( d , ops ) for d in data ]
    else: return do_ops( data , ops )

def rapply( data , ops ): return apply( data , ops , 'row' )
def capply( data , ops ): return apply( data , ops , 'col' )

