
import tensorflow as tf
import kaleido as kld

### TRCONV
def trconv( d , fn , input , name , channels , ksize , strides , shapeout = None , **args ):

    name = kld.lst.merge_str( name )
    args = kld.aux.merge_dicts( kld.tf.layer.default , args )

    if kld.chk.is_tensor( shapeout ):
        shapeout = kld.tf.shape( shapeout )

    ksize = kld.tf.layer.set_ksize( ksize , d )
    strides = kld.tf.layer.set_strides( strides , d )

    shapein = kld.tf.shape( input )
    shapeout = kld.tf.layer.set_shapeout( shapeout , input , strides , channels )
    wgts_shape = ksize + [ channels , shapein[d+1] ]
    bias_shape = [ channels ]

    with tf.variable_scope( name_or_scope = name , reuse = tf.AUTO_REUSE ):

        wgts = kld.tf.variable( wgts_shape , 'wgts' , 'wgts' , **args )
        bias = kld.tf.variable( bias_shape , 'bias' , 'bias' , **args )

        input = kld.apply( input , args['prev'] )
        output = fn( value = input , filter = wgts , output_shape = shapeout , strides = strides ,
                     padding = args['padding'] )
        output = tf.add( output , bias )
        output = kld.apply( output , args['post'] )

    return output

