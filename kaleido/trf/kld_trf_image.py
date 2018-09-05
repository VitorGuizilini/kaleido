
from tqdm import tqdm
from kaleido.chk import *

### TRANSFORM11
def transform11( input ):
    if is_lst( input ):
        return [ transform11( inp ) for inp in input ]
    return input / 127.5 - 1.0

### UNTRANSFORM11
def untransform11( input ):
    if is_lst( input ):
        return [ untransform11( inp ) for inp in input ]
    return ( input + 1.0 ) * 127.5

