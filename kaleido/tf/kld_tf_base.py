
import os
import tensorflow as tf
from tensorflow.contrib.framework.python.framework import checkpoint_utils
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import kaleido as kld
from kaleido.chk import *

########################################################

### SESSION
def Session( graph = None ):
    return tf.Session( graph = graph )

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

### TENSORS
def tensors():
    return [ n for n in tf.get_default_graph().as_graph_def().node ]

### PRINT TENSORS
def print_tensors():
    all_tensors = tensors()
    for tensor in all_tensors: print( tensor.name )

### PRINT VARS CKPT
def print_vars_ckpt( path ):
    print_tensors_in_checkpoint_file( file_name = path + '/kld_model' ,
                                      tensor_name = '' , all_tensors = False )

########################################################

### IMPORT META
def import_meta( path ):
    return tf.train.import_meta_graph( path + '/kld_model.meta' ,
                                       clear_devices = True )

### RESTORE META
def restore_meta( path , sess ):
    saver = import_meta( path )
    saver.restore( sess , path + '/kld_model' )
    return saver

### VARS2CONSTS
def vars2consts( nodes , sess ):
    return tf.graph_util.convert_variables_to_constants( sess ,
                        tf.get_default_graph().as_graph_def() , nodes )

### SAVE CONSTS
def save_consts( name , consts , encrypt = False ):
    with tf.gfile.GFile( name , "wb" ) as f:
        data = consts.SerializeToString()
        if encrypt: data = kld.cry.encrypt( data )
        f.write( data )

### FREEZE
def freeze( path , name , encrypt = False ):
    sess = Session()
    restore_meta( path , sess )
    nodes = [ tsr.name for tsr in tensors() if 'KLD_Nodes' in tsr.name ]
    consts = vars2consts( nodes , sess )
    save_consts( name , consts , encrypt )
    return nodes , sess
def freezecry( path , name ):
    return freeze( path , name , encrypt = True )

### UNFREEZE
def unfreeze( name , decrypt = False ):
    with tf.gfile.GFile( name , "rb" ) as f:
        data = f.read()
        if decrypt: data = kld.cry.decrypt( data )
    graph_def = tf.GraphDef()
    graph_def.ParseFromString( data )
    with tf.Graph().as_default() as graph:
        tf.import_graph_def( graph_def , name = 'frozen' )
    return frozen_nodes( graph ) , Session( graph )
def unfreezecry( name ):
    return unfreeze( name , decrypt = True )

### NODES TO FREEZE
def nodes_to_freeze( *nodes ):
    with tf.variable_scope( 'KLD_Nodes' ):
        for node in nodes:
            tf.identity( node[0] , node[1] )

### FROZEN NODES
def frozen_nodes( graph ):
    nodes = {}
    for op in graph.get_operations():
        if 'KLD_Nodes' in op.name:
            nodes[ op.name.split('/')[-1] ] = op
    return nodes

########################################################

### GRAPH
def graph():
    return tf.get_default_graph().as_graph_def()

### GLOBAL VARS
def global_vars( scope = None ):
    return tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES , scope = scope )
def print_global_vars( scope = None ):
    for var in global_vars( scope ):
        print( var )

### TRAINABLE VARS
def trainable_vars( scope = None ):
    return tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES , scope = scope )
def print_trainable_vars( scope = None ):
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

