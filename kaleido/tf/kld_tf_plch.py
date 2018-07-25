
import tensorflow as tf
import kaleido as kld

### PLCH
def plch( shape , name = None , dtype = tf.float32 , first_none = False ):
    if kld.chk.is_seq( shape ): shape = list( shape )
    if first_none: shape[0] = None
    return tf.placeholder( dtype , shape , name )

### PLCHF
def plchf( shape , name = None , first_none = False ):
    return plch( shape = shape , name = name , dtype = tf.float32 , first_none = first_none )

