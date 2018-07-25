
import tensorflow as tf
import kaleido as kld

### CONV
def conv( d , fn , input , name , channels , ksize , strides , **args ):

    args = kld.aux.merge( [ kld.tf.layer.default , args ] )

    ksize = kld.tf.layer.set_ksize( ksize , d )
    strides = kld.tf.layer.set_strides( strides , d )

    shapein = kld.tf.shape( input )
    wgts_shape = ksize + [ shapein[d+1] , channels ]
    bias_shape = [ channels ]

    with tf.variable_scope( name_or_scope = name , reuse = tf.AUTO_REUSE ):

        wgts = kld.tf.variable( wgts_shape , 'wgts' , 'wgts' , **args )
        bias = kld.tf.variable( bias_shape , 'bias' , 'bias' , **args )

        input = kld.tf.layer.apply_ops( input , args['prev'] )
        output = fn( input = input , filter = wgts , strides = strides ,
                     padding = args['padding'] )
        output = tf.add( output , bias )
        output = kld.tf.layer.apply_ops( output , args['post'] )

    return output

