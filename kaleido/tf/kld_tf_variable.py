
import tensorflow as tf
import kaleido as kld

### VARIABLE
def variable( shape , name , pref = '' , **args ):
    if pref in args:
        var = args[pref]
        if kld.chk.is_numpy(    var ): return kld.tf.variable( var , name )
        if kld.chk.is_variable( var ): return var
    if kld.chk.is_numpy( shape ):
        args.pop( 'init' , None )
        args['initializer'] , shape = tf.constant( shape ) , None
    dict = {}
    for key in args:
        if pref in key:
            if pref is '': new_key = key
            else: new_key = key[len(pref)+1:]
            if new_key == 'init': new_key = 'initializer'
            elif new_key == 'reg': new_key = 'regularizer'
            if new_key == 'initializer': dict[new_key] = kld.tf.layer.apply_init( args[key] )
            else: dict[new_key] = args[key]
    return tf.get_variable( shape = shape , name = name , **dict )
