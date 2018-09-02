
import random
from kaleido.chk import *

### FLOAT
def f( min = 0.0 , max = 1.0 ):
    if is_seq( min ): min , max = min
    if max is None: max = min ; min = 0
    if is_seq( min ): min , max = min
    if min > max: min , max = max , min
    return min + random.random() * ( max - min )

### INT
def i( min , max = None , inc = True ):
    if is_seq( min ): min , max = min
    if max is None: max = min ; min = 0
    if min > max: min , max = max , min
    if not inc: max -= 1
    return random.randint( min , max )
