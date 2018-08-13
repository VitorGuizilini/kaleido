
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import kaleido as kld

########################################################

### SESSION
def Session():
    return tf.Session()

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

### PRINT ALL TENSOR NAMES
def print_all_tensor_names():
    tensors = all_tensors()
    for t in tensors: print( t.name )

########################################################

### GLOBAL VARS
def global_vars( scope = None ):
    return tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES , scope = scope )

### TRAINABLE VARS
def trainable_vars( scope = None ):
    return tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES , scope = scope )

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
    else: return tf.variables_initializer( vars )

### INITIALIZE
def initialize( sess = None , vars = None ):
    if sess is None: sess = tf.Session()
    sess.run( init_op( vars ) )
    return sess

########################################################

#### APPLY OP
#def apply_op( input , ops , **kwargs ):
#    if kld.chk.empty( ops ): return input
#    if ( not kld.chk.is_seq( ops ) ) or \
#       ( kld.chk.is_seq( ops ) and len( ops ) == 1 ) or \
#       ( kld.chk.is_seq( ops ) and kld.chk.is_dict( ops[1] ) ):
#           ops = [ ops ]
#    for op in list( ops ):
#        if not kld.chk.is_seq( op ): input = op( input , **kwargs )
#        elif len( op ) == 1: input = op[0]( input , **kwargs )
#        else: input = op[0]( input , **op[1] , **kwargs )
#    return input

### APPLY INIT
def apply_init( init ):
    if not kld.chk.is_seq( init ): return init
    else: return init[0]( **init[1] )

########################################################

