
import tensorflow as tf
import kaleido as kld

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
        shapeout = kld.lst.mlt( shapein , strides )
    if len( shapeout ) == len( shapein ):
        shapeout = shapeout[1:-1]
    return [ shapein[0] ] + shapeout + [ channels ]


