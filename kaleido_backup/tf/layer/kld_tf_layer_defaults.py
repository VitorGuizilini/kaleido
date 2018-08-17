
import tensorflow as tf
import kaleido as kld

default = { 'wgts_init' : tf.initializers.truncated_normal( mean = 0.0 , stddev = 0.1 ) ,
            'wgts_trainable' : True , 'wgts_dtype' : tf.float32 ,

            'bias_init' : tf.initializers.zeros ,
            'bias_trainable' : True , 'bias_dtype' : tf.float32 ,

            'prev' : None , 'post' : None ,
            'padding' : 'SAME'
          }

### SET DEFAULTS
def defaults( dict ):
    kld.tf.layer.default = kld.aux.merge_dicts( kld.tf.layer.default , dict )
