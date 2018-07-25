
import tensorflow as tf
import kaleido as kld

### TRCONV2D
def trconv2d( input , name , channels , ksize , strides , shapeout = None , **args ):
    return kld.tf.layer.trconv( 2 , tf.nn.conv2d_transpose ,
                           input , name , channels , ksize , strides , shapeout , **args )
