
import tensorflow as tf
import kaleido as kld

### RNN
def rnn( input , name , channels , init_state = None , in_drop = None , out_drop = None , **args ):

    name = kld.lst.merge_str( name )
    args = kld.aux.merge_dicts( kld.tf.layer.default , args )
    in_drop , out_drop = kld.lst.make( in_drop ) , kld.lst.make( out_drop )

    batch_size = kld.tf.shape( input )[0]

    with tf.variable_scope( name_or_scope = name , reuse = tf.AUTO_REUSE ):

        cells = [ tf.nn.rnn_cell.LSTMCell( size ) for size in channels ]
        for i in range( len( cells ) ):
            in_dropi  = in_drop[0]  if len( in_drop  ) == 1 else in_drop[i]
            out_dropi = out_drop[0] if len( out_drop ) == 1 else out_drop[i]
            if in_dropi is not None or out_dropi is not None:
                cells[i] = tf.contrib.rnn.DropoutWrapper( cells[i] ,
                        input_keep_prob = in_dropi , output_keep_prob = out_dropi )
        cells = tf.nn.rnn_cell.MultiRNNCell( cells )

        if init_state is None: init_state = cells.zero_state( batch_size , dtype = tf.float32 )

        input = kld.tf.apply_op( input , args['prev'] )
        output , state = tf.nn.dynamic_rnn( cells , input ,
                            initial_state = init_state , dtype = tf.float32 )
        output = kld.tf.apply_op( output , args['post'] )

    return output , state


