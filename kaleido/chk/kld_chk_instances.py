
import numpy as np
import tensorflow as tf

### IS TUPLE
def is_tuple( input ):
    return isinstance( input , tuple )

### IS LIST
def is_list( input ):
    return isinstance( input , ( list ) )

### IS SEQ
def is_seq( input ):
    return isinstance( input , ( list , tuple ) )

### IS 2SEQ
def is_2seq( input ):
    return is_seq( input ) and is_seq( input[0] )

### IS NUMPY
def is_numpy( input ):
    return isinstance( input , np.ndarray )

### IS TENSOR
def is_tensor( input ):
    return isinstance( input , tf.Tensor )

### IS VARIABLE
def is_variable( input ):
    return isinstance( input , tf.Variable )

### IS INT
def is_int( input ):
    return isinstance( input , int )

### IS FLOAT
def is_float( input ):
    return isinstance( input , float )

### STR
def is_str( input ):
    return isinstance( input , str )

### IS DICT
def is_dict( input ):
    return isinstance( input , dict )

### IS CALLABLE
def is_callable( input ):
    return callable( input )

### IS CLASS
def is_class( input , type ):
    return isinstance( input , type )

### STR IS INT
def is_str_int( input ):
    try:
        int( input )
        return True
    except ValueError:
        return False

### STR IS FLOAT
def is_str_float( input ):
    try:
        float( input )
        return True
    except ValueError:
        return False
