
import scipy.io
import numpy as np
import tensorflow as tf
import kaleido as kld

def get( arch , *args , **kwargs ):
    if arch == 'vgg19': return kld.tf.arch.vgg19( *args , **kwargs )
