
import numpy
import tensorflow

def empty( x ):  return x is None or ( is_seq( x ) and len( x ) == 0 )
def nested( x ): return is_lst( x ) and is_lst( x[0] )
def nested2( x ): return is_lst( x ) and is_lst( x[0] ) and is_lst( x[0][0] )

def is_lst( x ): return isinstance( x , list  )
def is_tup( x ): return isinstance( x , tuple )
def is_seq( x ): return isinstance( x , ( list , tuple ) )
def is_npy( x ): return isinstance( x , numpy.ndarray )

def is_tsr( x ): return isinstance( x , tensorflow.Tensor   )
def is_var( x ): return isinstance( x , tensorflow.Variable )

def is_int( x ): return isinstance( x , int   )
def is_flt( x ): return isinstance( x , float )
def is_str( x ): return isinstance( x , str   )
def is_dct( x ): return isinstance( x , dict  )
def is_bol( x ): return isinstance( x , bool  )

def is_str_int( x ):
    if x is None: return False
    if not is_str( x ): x = str( x )
    try: int( x ) ; return True
    except ValueError: return False
def is_str_flt( x ):
    if x is None: return False
    if not is_str( x ): x = str( x )
    try: float( x ) ; return True
    except ValueError: return False
def is_type( x , t ):
    if x is None: return False
    try: t( x ) ; return True
    except: return False

