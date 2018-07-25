
import tensorflow as tf
import kaleido as kld

### TRCONV3D
def trconv3d( input , name , channels , ksize , strides , shapeout = None , **args ):
    return kld.tf.layer.trconv( 3 , tf.nn.conv3d_transpose ,
                           input , name , channels , ksize , strides , shapeout , **args )
