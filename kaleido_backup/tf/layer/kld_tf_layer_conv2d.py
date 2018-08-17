
import tensorflow as tf
import kaleido as kld

### CONV2D
def conv2d( input , name , channels , ksize , strides , **args ):
    return kld.tf.layer.conv( 2 , tf.nn.conv2d ,
                           input , name , channels , ksize , strides , **args )
