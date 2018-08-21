
import tensorflow as tf
import kaleido as kld

##### VRS
class vrs:

    ### __INIT__
    def __init__( self , args , scope , *pargs , **kwargs ):

        self.args = args

        if kld.chk.is_str( pargs[0] ):
            name = pargs[0]
            pargs = tuple( pargs[1:] )
        else: name = None

        with tf.variable_scope( scope , reuse = tf.AUTO_REUSE ) as scope2:
            with tf.name_scope( scope2.original_name_scope ):
                if name is not None:
                    with tf.name_scope( name ):
                        self.build( *pargs , **kwargs )
                else: self.build( *pargs , **kwargs )



