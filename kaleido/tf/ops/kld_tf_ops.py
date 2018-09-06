
import tensorflow as tf
import kaleido as kld

######################

### TOFLOAT
def toFloat( t ):
    return tf.cast( t , tf.float32 )

######################

### TOTAL VARIATION LOSS
def total_variation_loss( img ):

    b , h , w , d = kld.tf.shape( img )

    x_tv_size = toFloat( h * ( w - 1 ) * d )
    y_tv_size = toFloat( ( h - 1 ) * w * d )
    b = toFloat( b )

    x_tv = tf.nn.l2_loss( img[ : ,  : , 1: , : ] - img[ : , : , :w - 1 , : ] )
    y_tv = tf.nn.l2_loss( img[ : , 1: ,  : , : ] - img[ : , :h - 1 , : , : ] )

    loss = 2.0 * ( x_tv / x_tv_size
                 + y_tv / y_tv_size ) / b

    return loss

######################

### GRAM MATRIX
def gram_matrix( tensor ):

    b , h , w , c = kld.tf.shape( tensor )
    chw = toFloat( c * h * w )

    feats = tf.reshape( tensor , ( b , h * w , c ) )
    feats_T = tf.transpose( feats , perm = [ 0 , 2 , 1 ] )
    gram = tf.matmul( feats_T , feats ) / chw

    return gram

######################

### PHASE SHIFT1
def phase_shift1( X , r ):
    bs , a , b , _ = kld.tf.shape( X )
    X = tf.reshape( X , ( bs , a , b , r , r ) )
    X = tf.split( X , a , 1 )
    X = tf.concat( [ tf.squeeze( x , axis = 1 ) for x in X ] , 2 )
    X = tf.split( X , b , 1 )
    X = tf.concat( [ tf.squeeze( x , axis = 1 ) for x in X ] , 2 )
    X = tf.reshape( X , ( bs , a * r , b * r , 1 ) )
    return X

### PHASE SHIFT
def phase_shift( X , r , c = 1 ):
    if c == 1: return phase_shift1( X , r )
    else: return tf.concat( [ phase_shift1( x , r ) for x in tf.split( X , c , 3 ) ] , 3 )

######################

### LLRELU
def llrelu( x , leak = 0.2 ):
    f1 , f2 = 0.5 * ( 1 + leak ) , 0.5 * ( 1 - leak )
    return f1 * x + f2 * abs( x )

######################

