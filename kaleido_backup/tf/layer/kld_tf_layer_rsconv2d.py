
import tensorflow as tf
import kaleido as kld

### RSCONV2D
def rsconv2d( input , name , channels , ksize , shapeout , **args ):

    name = kld.lst.merge_str( name )
    args = kld.aux.merge_dicts( kld.tf.layer.default , args )

    shapein = kld.tf.shape( input )
    if kld.chk.is_tsr( shapeout ):
        shapeout = kld.tf.shape( shapeout )

    if kld.chk.is_seq( shapeout ):
        size = shapeout if len( shapeout ) == 2 else shapeout[1:-1]
    else:
        size = shapein[1:-1]
        for i in range( len( size ) ): size[i] *= shapeout

    output = tf.image.resize_images( input , size = size )
    output = kld.tf.layer.conv2d( output , name , channels , ksize , None , **args )

    return output

