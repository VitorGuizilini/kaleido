
import os
import scipy
import numpy as np
import tensorflow as tf
import kaleido as kld
import shutil

##### SAVER
class Saver:

    ### __INIT__
    def __init__( self , path , sess = None ):
        self.path , self.sess = path , sess
        self.stored = {}

############################################ MODEL

    ### RESTORE MODEL
    def restore_model( self , name , vars = None ):
        if name not in self.stored: self.start_model( name , vars )
        saver , vars , path = self.stored[name]
        if tf.train.checkpoint_exists( path ):
            saver.restore( self.sess , path )
            print( '#############################')
            print( '######### RESTORED!!!' , kld.path.dir( path ) )
            print( '#############################')
        else:
            kld.tf.initialize( self.sess , vars )
            print( '#################################')
            print( '######### NOT RESTORED...' , kld.path.dir( path ) )
            print( '#################################')

    ### START MODEL
    def start_model( self , name , vars = None , max_to_keep = 1 ):
        if kld.chk.is_seq( name ):
            for item in name: self.start_model( item )
        else:
            path = self.path + '/models/' + name ; kld.path.mkdir( path )
            self.stored[name] = [ tf.train.Saver( max_to_keep = max_to_keep ,
                                                  var_list  = vars ) , vars , path + '/kld_model' ]

    ### MODEL
    def model( self , name , vars = None ):
        if kld.chk.is_seq( name ):
            for i in range( len( name ) ): self.model( name[i] , vars )
        else:
            if name not in self.stored: self.start_model( name , vars )
            saver , vars , path = self.stored[name]
            saver.save( self.sess , path )

############################################ SCALAR

    ### RESTORE SCALAR
    def restore_scalar( self , name , default ):
        path = self.stored[name][1]
        if os.path.isfile( path ):
            scalar = np.loadtxt( path )
            self.stored[name][0] = scalar
            return scalar
        else:
            return self.scalar( name , default )

    ### START SCALAR
    def start_scalar( self , name , default ):
        if kld.chk.is_seq( name ):
            scalars = []
            for i in range( len( name ) ):
                defaulti = default[i] if kld.chk.is_seq( default ) else default
                scalars.append( self.start_scalar( name[i] , defaulti ) )
            return scalars
        else:
            path = self.path + '/data/'
            kld.path.mkdir( path )
            self.stored[name] = [ [] , path + name + '.txt' ]
            return self.restore_scalar( name , default )

    ### SCALAR
    def scalar( self , name , data = None ):
        if kld.chk.is_seq( name ):
            scalars = []
            for i in range( len( name ) ):
                datai = data[i] if kld.chk.is_seq( data ) else data
                scalars.append( self.scalar( name[i] , datai ) )
            return scalars
        else:
            if name not in self.stored: self.start_scalar( name , data )
            path = self.stored[name][1]
            if data is not None:
                self.stored[name][0] = data
                np.savetxt( path , [ data ] )
            return self.stored[name][0]

############################################ NUMPY

    ### RESTORE NUMPY
    def restore_numpy( self , name , default ):
        path = self.stored[name][1]
        if os.path.isfile( path ):
            numpy = np.load( path )
            self.stored[name][0] = numpy
            return numpy
        else:
            return self.numpy( name , default )

    ### START NUMPY
    def start_numpy( self , name , default ):
        if kld.chk.is_seq( name ):
            numpys = []
            for i in range( len( name ) ):
                defaulti = default[i] if kld.chk.is_seq( default ) else default
                numpys.append( self.start_numpy( name[i] , defaulti ) )
            return numpys
        else:
            path = self.path + '/data/'
            kld.path.mkdir( path )
            self.stored[name] = [ [] , path + name + '.npy' ]
            return self.restore_numpy( name , default )

    ### NUMPY
    def numpy( self , name , data = None ):
        if kld.chk.is_seq( name ):
            numpys = []
            for i in range( len( name ) ):
                datai = data[i] if kld.chk.is_seq( data ) else data
                numpys.append( self.numpy( name[i] , datai ) )
            return numpys
        else:
            if name not in self.stored: self.start_numpy( name , data )
            path = self.stored[name][1]
            if data is not None:
                self.stored[name][0] = data
                np.save( path , data )
            return self.stored[name][0]

############################################ LIST

    ### RESTORE LIST
    def restore_list( self , name ):
        list , path = self.stored[name]
        if os.path.isfile( path ):
            with open( path , 'r' ) as file:
                for line in file:
                    floats = [ float( x ) for x in line.split() ]
                    if len( floats ) == 1: floats = floats[0]
                    list.append( floats )
        return list

    ### START LIST
    def start_list( self , name ):
        if kld.chk.is_seq( name ):
            lists = []
            for i in range( len( name ) ):
                lists.append( self.start_list( name[i] ) )
            return lists
        else:
            path = self.path + '/data/'
            kld.path.mkdir( path )
            self.stored[name] = [ [] , path + name + '.txt' ]
            return self.restore_list( name )

    ### LIST
    def list( self , name , data = None ):
        if kld.chk.is_seq( name ):
            lists = []
            for i in range( len( name ) ):
                datai = data[i] if kld.chk.is_seq( data ) else data
                lists.append( self.list( name[i] , data[i] ) )
            return lists
        else:
            if name not in self.stored: self.start_list( name )
            list , path = self.stored[name]
            if data is not None:
                list.append( data )
                np.savetxt( path , list )
            return list

############################################ IMAGE

    ### START IMAGE
    def start_image( self , name ):
        if kld.chk.is_seq( name ):
            for item in name: self.start_image( item )
        else:
            path = self.path + '/images/' + name
            self.stored[name] = path

    ### IMAGE
    def image( self , name , plt , suffix , folders = None ):
        if kld.chk.is_seq( name ):
            for i in range( len( name ) ): self.image( name[i] , plt[i] , folders[i] , suffix[i] )
        else:
            if name not in self.stored: self.start_image( name )
            path = self.stored[name]
            if folders is not None:
                for folder in kld.list.make( folders ): path += '/' + folder
            kld.path.mkdir( path )
            if suffix is not None: path += '/' + suffix
            path += '.png'
            if kld.chk.is_numpy( plt ):
                scipy.misc.imsave( path , plt )
            else: plt.savefig( path )

############################################ FILE

    ### START FILE
    def start_file( self , name , files ):
        files = kld.list.make( files )
        if kld.chk.is_seq( name ):
            for i in range( len( name ) ):
                self.start_file( name[i] , files[i] )
        else:
            path = self.path + '/files/' + name + '/'
            kld.path.mkdir( path )
            self.stored[name] = [ files , path ]

    ### FILE
    def file( self , name , files = None ):
        if kld.chk.is_seq( name ):
            for i in range( len( name ) ):
                self.file( name[i] , files[i] )
        else:
            if name not in self.stored: self.start_file( name , files )
            files , path = self.stored[name]
            for file in files:
                dest = file.split('/')[-1]
                if file[-3] == '.':
                    shutil.copy( file , path + dest )
                else:
                    shutil.rmtree( path + dest , ignore_errors = True )
                    shutil.copytree( file , path + dest )

############################################ DICT

    ### LOAD DICT
    def load_dict( self , name ):
        data = {}
        f = open( name , 'r' )
        for line in f:
            info = line.split( ' ' )
            key , val , type = info[0][2:] , info[1:-2] , info[-2]
            for i in range( len( val ) ): val[i] = kld.aux.vartype( val[i] , type )
            if len( val ) == 1: val = val[0]
            data[key] = val
        f.close()
        return data

    ### SAVE DICT
    def save_dict( self , name , data ):
        f = open( name , 'w' )
        for key in data:
            f.write( '--' + key + ' ' )
            value = data[key]
            if kld.chk.is_seq( value ):
                for v in value: f.write( str( v ) + ' ' )
                f.write( kld.aux.vartype( value[0] ) )
            else:
                f.write( str( value ) + ' ' )
                f.write( kld.aux.vartype( value ) )
            f.write( ' \n' )
        f.close()
        return data

    ### RESTORE DICT
    def restore_dict( self , name , default = {} ):
        path = self.stored[name][1]
        if os.path.isfile( path ):
            dict = self.load_dict( path )
            self.stored[name][0] = dict
            return dict
        else:
            return self.dict( name , default )

    ### START DICT
    def start_args( self , name , default = {} ):
        return self.start_dict( name , default )
    def start_dict( self , name , default = {} ):
        if kld.chk.is_seq( name ):
            dicts = []
            for i in range( len( name ) ):
                dicts.append( self.start_dict( name[i] ) )
            return dicts
        else:
            path = self.path + '/data/'
            kld.path.mkdir( path )
            self.stored[name] = [ {} , path + name + '.txt' ]
            return self.restore_dict( name , default )

    ### DICT
    def args( self , name , data = None ):
        return self.dict( name ) if data is None else self.dict( name , data.__dict__ )
    def dict( self , name , data = None ):
        if kld.chk.is_seq( name ):
            dicts = []
            for i in range( len( name ) ):
                datai = data[i] if kld.chk.is_seq( data ) else data
                dicts.append( self.dict( name[i] , datai ) )
            return dicts
        else:
            if name not in self.stored: self.start_dict( name )
            path = self.stored[name][1]
            if data is not None:
                self.stored[name][0] = data
                self.save_dict( path , data )
            return self.stored[name][0]
