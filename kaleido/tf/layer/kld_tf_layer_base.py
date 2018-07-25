
import tensorflow as tf
import kaleido as kld

### APPLY OP
def apply_op( input , op , **kwargs ):
    if op is None: return input
    return op[0]( input , **op[1] , **kwargs ) if kld.chk.is_seq( op ) else op( input , **kwargs )

### APPLY OPS
def apply_ops( input , ops ):
    if ops is None:
        return input
    else:
        if not kld.chk.is_seq( ops ):
            return kld.tf.layer.apply_op( input , ops )
        else:
            for op in ops:
                input = kld.tf.layer.apply_op( input , op )
            return input

### APPLY INIT
def apply_init( init ):
    if not kld.chk.is_seq( init ): return init
    else: return init[0]( **init[1] )

### SET KSIZE
def set_ksize( ksize , dims ):
    if not kld.chk.is_seq( ksize ):
        return [ ksize ] * dims
    return ksize

### SET STRIDES
def set_strides( strides , dims ):
    if strides is None:
        return [ 1 ] * ( dims + 2 )
    if not kld.chk.is_seq( strides ):
        return [ 1 ] + [ strides ] * dims + [ 1 ]
    if len( strides ) == dims:
        return [ 1 ] + strides + [ 1 ]
    return strides

### SET SHAPEOUT
def set_shapeout( shapeout , input , strides , channels ):
    shapein = kld.tf.shape( input )
    if shapeout is None:
        shapeout = kld.list.mult( shapein , strides )
    if len( shapeout ) == len( shapein ):
        shapeout = shapeout[1:-1]
    return [ shapein[0] ] + shapeout + [ channels ]


