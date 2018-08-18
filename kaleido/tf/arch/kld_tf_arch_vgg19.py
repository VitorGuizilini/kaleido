
import scipy.io
import numpy as np
import tensorflow as tf

##### VGG19
class vgg19:

    MODEL_FILE_NAME = 'imagenet-vgg-verydeep-19.mat'

    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    ### __INIT__
    def __init__( self ):
        data = scipy.io.loadmat( '../../data/architectures/' + self.MODEL_FILE_NAME )
        self.mean_pixel = np.array( [ 123.68 , 116.779 , 103.939 ] )
        self.weights = data['layers'][0]

    ### CONV LAYER
    def conv_layer( self , input , weights , bias ):
        conv = tf.nn.conv2d( input , tf.constant( weights ) ,
                                     strides = ( 1 , 1 , 1 , 1 ) , padding = 'SAME' )
        return tf.nn.bias_add( conv , bias )

    ### POOL LAYER
    def pool_layer( self , input ):
        return tf.nn.max_pool( input , ksize = ( 1 , 2 , 2 , 1 ) ,
                                       strides = ( 1 , 2 , 2 , 1 ) , padding = 'SAME' )

    ### PREPROCESS
    def preprocess( self , input ):
        return input - self.mean_pixel

    ### POSTPROCESS
    def postprocess( self , input ):
        return input + self.mean_pixel

    ### BUILD
    def build( self , input , name = None , preprocess = True ):

        if preprocess:
            input = self.preprocess( input )

        net , curr = {} , input
        with tf.variable_scope( name ):
            for i , layer in enumerate( self.layers ):

                kind = layer[:4]

                if kind == 'conv':

                    kernels = self.weights[i][0][0][2][0][0]
                    bias    = self.weights[i][0][0][2][0][1]

                    kernels = np.transpose( kernels , ( 1 , 0 , 2 , 3 ) )
                    bias    = bias.reshape(-1)

                    curr = self.conv_layer( curr , kernels , bias )

                elif kind == 'relu':

                    curr = tf.nn.relu( curr )

                elif kind == 'pool':

                    curr = self.pool_layer( curr )

                net[layer] = curr

        return net
