
from kaleido.aux import *
from kaleido.chk import *

### NPY2LST
def npy2lst( data ):
    if is_lst( data ):
        data = np.concatenate( data , axis = 0 )
    return [ d for d in data ]

### TYP2STR
def typ2str( v ):
    if v == int:   return 'int'
    if v == float: return 'float'
    if v == bool:  return 'bool'
    if v == str:   return 'string'
    if v == list:  return 'list'
    return 'none'

