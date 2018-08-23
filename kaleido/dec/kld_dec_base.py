
import numpy as np
import kaleido as kld

### VARSCOPE
def varscope( scope ):
    def secondary( function ):
        def wrapper( *args , **kwargs ):
            oldscope = kld.tf.get_varscope()
            kld.tf.set_varscope( scope )
            function( *args , **kwargs )
            kld.tf.set_varscope( oldscope )
        return wrapper
    return secondary
