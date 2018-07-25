
import numpy as np
import tensorflow as tf
import kaleido as kld

##### BOARD
class Board:

    ### __INIT__
    def __init__( self , path , sess = None ):
        path , self.sess = path , sess
        self.writer = tf.summary.FileWriter( path + '/tensorboard' , tf.get_default_graph() )
        self.dict = {}

    ### GET SCALAR
    def get_scalar( self , name ):
        if kld.chk.is_seq( name ):
            for item in name: self.get_scalar( item )
        else:
            plch = tf.placeholder( tf.float32 , None , name = 'Logs_' + name )
            summ = tf.summary.scalar( name , plch )
            self.dict[ name ] = [ plch , summ ]

    ### GET IMAGE
    def get_image( self , name , max_outputs = 1000 ):
        if kld.chk.is_seq( name ):
            for item in name: self.get_image( item )
        else:
            plch = tf.placeholder( tf.float32 , [ None ] * 4 , name = 'Logs_' + name )
            summ = tf.summary.image( name , plch , max_outputs = max_outputs )
            self.dict[ name ] = [ plch , summ ]

    ### STORE
    def store( self , name , data , step ):
        if kld.chk.is_seq( name ):
            for i , item in enumerate( name ):
                self.store( item , data[i] , step )
        else:
            plch , summ = self.dict[ name ]
            if kld.chk.is_numpy( data ):
                if len( data.shape ) == 3:
                    data = np.expand_dims( data , axis = 3 )
            outp = self.sess.run( summ , feed_dict = { plch : data } )
            self.writer.add_summary( outp , step )

    ### SCALAR
    def scalar( self , name , data , step ):
        name = kld.list.make( name )
        for item in name:
            if item not in self.dict:
                self.get_scalar( name )
        self.store( name , data , step )

    ### IMAGE
    def image( self , name , data , step ):
        name = kld.list.make( name )
        for item in name:
            if item not in self.dict:
                self.get_image( name )
        self.store( name , data , step )
