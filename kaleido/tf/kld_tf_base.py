
import os
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from kaleido.chk import *

########################################################

### SESSION
def Session():
    return tf.Session()

########################################################

### GET VARSCOPE
def get_varscope():
    return tf.get_variable_scope()._name

### SET VARSCOPE
def set_varscope( scope ):
    old = get_varscope()
    tf.get_variable_scope()._name = scope
    return old

### SET REUSE
def set_reuse( flag ):
    tf.get_variable_scope()._reuse = flag

########################################################

### SHAPE
def shape( tensor ):
    shape1 , shape2 = tensor.get_shape().as_list() , tf.shape( tensor )
    for i in range( len( shape1 ) ):
        if shape1[i] is None: shape1[i] = shape2[i]
    return shape1

########################################################

### NUM TRAINABLE PARAMS
def num_trainable_params():
    return np.sum( [ np.prod( v.get_shape().as_list() ) for v in tf.trainable_variables() ] )

########################################################

### ALL TENSORS
def all_tensors():
    return [ n for n in tf.get_default_graph().as_graph_def().node ]

### PTENSOR NAMES
def ptensor_names():
    tensors = all_tensors()
    for t in tensors: print( t.name )

### PTENSORS CHECKPOINT
def ptensors_checkpoint( path ):
    print_tensors_in_checkpoint_file( file_name = path , tensor_name = '' , all_tensors = False )

########################################################

### GLOBAL VARS
def global_vars( scope = None ):
    return tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES , scope = scope )
def pglobal_vars( scope = None ):
    for var in global_vars( scope ):
        print( var )

### TRAINABLE VARS
def trainable_vars( scope = None ):
    return tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES , scope = scope )
def ptrainable_vars( scope = None ):
    for var in trainable_vars( scope ):
        print( var )

### UPDATE OPS
def update_ops( scope = None ):
    return tf.get_collection( tf.GraphKeys.UPDATE_OPS , scope = scope )

### REGULARIZATION LOSSES
def regularization_losses( scope = None ):
    return tf.get_collection( tf.GraphKeys.REGULARIZATION_LOSSES , scope = scope )

########################################################

### INIT_OP
def init_op( vars = None ):
    if vars is None: return tf.global_variables_initializer()
    elif is_str( vars ): vars = global_vars( vars )
    return tf.variables_initializer( vars )

### INITIALIZE
def initialize( sess = None , vars = None ):
    if sess is None: sess = tf.Session()
    sess.run( init_op( vars ) )
    return sess

########################################################

### INIT
def init( init ):
    if not is_tup( init ): return init
    else: return init[0]( **init[1] )

########################################################

