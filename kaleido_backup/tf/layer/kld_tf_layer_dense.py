
import tensorflow as tf
import kaleido as kld

### DENSE
def dense( input , name , channels , **args ):

    name = kld.lst.merge_str( name )
    args = kld.aux.merge_dicts( kld.tf.layer.default , args )

    shapein = kld.tf.shape( input )
    wgts_shape = [ shapein[1] , channels ]
    bias_shape = [ channels ]

    with tf.variable_scope( name_or_scope = name , reuse = tf.AUTO_REUSE ):

        wgts = kld.tf.variable( wgts_shape , 'wgts' , 'wgts' , **args )
        bias = kld.tf.variable( bias_shape , 'bias' , 'bias' , **args )

        input = kld.apply( input , args['prev'] )
        output = tf.matmul( input , wgts )
        output = tf.add( output , bias )
        output = kld.apply( output , args['post'] )

    return output

