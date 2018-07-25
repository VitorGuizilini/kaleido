
import tensorflow as tf
import kaleido as kld

### DENSE
def dense( input , name , nodes , **args ):

    args = kld.aux.merge( [ kld.tf.layer.default , args ] )

    shapein = kld.tf.shape( input )
    wgts_shape = [ shapein[1] , nodes ]
    bias_shape = [ nodes ]

    with tf.variable_scope( name_or_scope = name , reuse = tf.AUTO_REUSE ):

        wgts = kld.tf.variable( wgts_shape , 'wgts' , 'wgts' , **args )
        bias = kld.tf.variable( bias_shape , 'bias' , 'bias' , **args )

        input = kld.tf.layer.apply_ops( input , args['prev'] )
        output = tf.matmul( input , wgts )
        output = tf.add( output , bias )
        output = kld.tf.layer.apply_ops( output , args['post'] )

    return output

