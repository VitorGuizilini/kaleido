
import tensorflow as tf
import kaleido as kld

### CONV3D
def conv3d( input , name , channels , ksize , strides , **args ):
    return kld.tf.layer.conv( 3 , tf.nn.conv3d ,
                           input , name , channels , ksize , strides , **args )


