
import os
import cv2
import numpy as np
import tensorflow as tf
import kaleido as kld
import shutil

##### SAVER
class Saver:

    ### __INIT__
    def __init__( self , path , sess = None , restart = False , free = False ):
        self.path , self.sess , self.stored , self.free = path , sess , {} , free
        if restart: self.restart()

    ### RESTART
    def restart( self ):
        kld.log.restart( self.path )

############################################ MODEL

    ### RESTORE MODEL
    def restore_model( self , name , vars = None ):
        if self.path is None: kld.tf.initialize( self.sess , vars ) ; return
        if name is None: kld.tf.initialize( self.sess , vars ) ; return
        if name not in self.stored: self.start_model( name , vars )
        saver , vars , path = self.stored[name]
        dir = kld.pth.dir( path )
        if kld.pth.exists( dir ) and tf.train.checkpoint_exists( path ):
            try:
                saver.restore( self.sess , path )
                str = '###### MODEL RESTORED!!! ' + dir
                print( '=' * len( str ) + '\n' + str + '\n' + '=' * len( str ) )
            except:
                kld.tf.initialize( self.sess , vars )
                str = '###### MODEL DOES NOT FIT!!! ' + dir
                print( '=' * len( str ) + '\n' + str + '\n' + '=' * len( str ) )
        else:
            kld.tf.initialize( self.sess , vars )
            str = '###### MODEL NOT FOUND... ' + dir
            print( '=' * len( str ) + '\n' + str + '\n' + '=' * len( str ) )

    ### RESTORE SCOPE
    def restore_scope( self , scope = None ):
        if scope is None: scope = kld.tf.get_varscope()
        self.restore_model( scope.lower() , scope )

    ### START MODEL
    def start_model( self , name , vars = None , max_to_keep = 1 ):
        if self.path is None: return
        if kld.chk.is_tup( name ):
            for item in name: self.start_model( item )
        else:
            if vars is not None and kld.chk.is_str( vars ):
                vars = kld.tf.global_vars( vars[1:] if vars[0] == '/' else vars )
            path = self.path + ( '/models/' if not self.free else '/' ) + name
            self.stored[name] = [ tf.train.Saver( max_to_keep = max_to_keep ,
                                                  var_list  = vars ) , vars , path + '/kld_model' ]

    ### MODEL
    def model( self , name , vars = None , restore = False ):
        if self.path is None: return
        if name is None: return
        if kld.chk.is_tup( name ):
            for item in name: self.model( item , vars )
        else:
            if name not in self.stored:
                if restore: self.restore_model( name , vars )
                else: self.start_model( name , vars )
            saver , vars , path = self.stored[name]
            kld.pth.mkdir( kld.pth.dir( path ) )
            saver.save( self.sess , path )

    ### SCOPE
    def scope( self , scope = None ):
        if scope is None: scope = kld.tf.get_varscope()
        self.model( scope.lower() , scope )

############################################ IMAGE

    ### START IMAGE
    def start_image( self , name ):
        if self.path is None: return
        if kld.chk.is_tup( name ):
            for item in name: self.start_image( item )
        else:
            path = self.path + ( '/images/' if not self.free else '/' ) + name
            self.stored[name] = path

    ### IMAGE
    def image( self , name , plt , suffix , folders = None ):

        if self.path is None: return
        if kld.chk.is_tup( name ):
            for i in kld.aux.rlen( name ):
                foldersi = folders[i] if kld.chk.is_tup( folders ) else folders
                self.image( name[i] , plt[i] , suffix[i] , foldersi )
        else:
            if name not in self.stored: self.start_image( name )
            path = self.stored[name]
            if folders is not None:
                for folder in kld.lst.make( folders ): path += '/' + folder
            kld.pth.mkdir( path )
            if suffix is not None: path += '/' + kld.lst.merge_str( suffix )
            path += '.png'
            if kld.chk.is_npy( plt ):
                if plt.dtype != np.uint8:
                    if np.max( plt ) <= 1.5:
                        plt = np.clip( plt * 255.0 , 0.0 , 255.0 ).astype( np.uint8 )
                else: plt = np.clip( plt , 0.0 , 255.0 )
                if len( plt.shape ) == 3 and plt.shape[2] == 3:
                    cv2.imwrite( path , plt[:,:,::-1] )
                else: cv2.imwrite( path , plt )
            else: plt.savefig( path )

############################################ FILE

    ### START FILE
    def start_file( self , name , files ):
        if self.path is None: return
        files = kld.lst.make( files )
        if kld.chk.is_tup( name ):
            for i in kld.aux.rlen( name ):
                self.start_file( name[i] , files[i] )
        else:
            path = self.path + '/files/' if not self.free else '/'
            if name != '': path += name + '/'
            kld.pth.mkdir( path )
            self.stored[name] = [ files , path ]

    ### FILE
    def file( self , name , files = None ):
        if self.path is None: return
        if kld.chk.is_tup( name ):
            for i in kld.aux.rlen( name ):
                self.file( name[i] , files[i] )
        else:
            if name not in self.stored: self.start_file( name , files )
            nfiles , path = self.stored[name]
            if files is None: files = nfiles
            else: files = self.stored[name][0] = list( set( nfiles + kld.lst.make( files ) ) )
            for file in files:
                dest = file.split('/')[-1]
                if file[-3] == '.':
                    shutil.copy( file , path + dest )
                else:
                    shutil.rmtree( path + dest , ignore_errors = True )
                    shutil.copytree( file , path + dest )

############################################ SCALAR/NUMPY/LIST/DICT

################################# RESTORE

    def load_scalar( self , path ):
        return float( np.loadtxt( path ) )
    def load_numpy( self , path ):
        return np.load( path )
    def load_list( self , path ):
        with open( path , 'r' ) as file:
            list = []
            for line in file:
                floats = [ float( x ) for x in line.split() ]
                list.append( floats )
        return list
    def load_dict( self , path ):
        with open( path , 'r' ) as file:
            data = {}
            for line in file:
                info = line.split( ' ' )
                key , val , type = info[0][2:] , info[1:-2] , info[-2]
                for i in range( len( val ) ): val[i] = kld.aux.vartype( val[i] , type[:3] )
                if len( val ) == 1 and len( type ) == 3: val = val[0]
                data[key] = val
        return data

    def restore( self , name , default , fnload , fnaccess ):
        if self.path is None: return
        path = self.stored[name][1]
        if os.path.isfile( path ):
            value = fnload( path )
            self.stored[name][0] = value
            return value
        else: return default

    def restore_scalar( self , name , default ):
        return self.restore( name , default , self.load_scalar , self.scalar )
    def restore_numpy( self , name , default ):
        return self.restore( name , default , self.load_numpy , self.numpy )
    def restore_list( self , name , default ):
        return self.restore( name , default , self.load_list , self.list )
    def restore_dict( self , name , default ):
        return self.restore( name , default , self.load_dict , self.dict )
    def restore_args( self , name , default ):
        return self.restore_dict( name , default )

################################# START

    def start( self , name , default , fnstart , fnrestore , ext ):
        if self.path is None: return default
        if kld.chk.is_tup( name ):
            values = []
            for i in kld.aux.rlen( name ):
                defaulti = default[i] if kld.chk.is_tup( default ) else default
                values.append( self.start( name[i] , defaulti , fnstart , fnrestore , ext ) )
            return values
        else:
            path = self.path + ( '/data/' if not self.free else '/' )
            self.stored[name] = [ [] , path + name + ext ]
            return fnrestore( name , default )

    def start_scalar( self , name , default ):
        return self.start( name , default , self.start_scalar , self.restore_scalar , '.txt' )
    def start_numpy( self , name , default ):
        return self.start( name , default , self.start_numpy , self.restore_numpy , '.npy' )
    def start_list( self , name , default ):
        return self.start( name , default , self.start_list , self.restore_list , '.txt' )
    def start_dict( self , name , default ):
        return self.start( name , default , self.start_dict , self.restore_dict , '.txt' )
    def start_args( self , name , default ):
        return self.start_dict( name , vars( default ) )

################################# ACCESS

    def save_scalar( self , path , data , struct ):
        if data is not None:
            np.savetxt( path , [ data ] )
            return data
        return struct
    def save_numpy( self , path , data , struct ):
        if data is not None:
            np.save( path , data )
            return data
        return struct
    def save_list( self , path , data , struct ):
        if data is not None:
            struct += data if kld.chk.nested( data ) else [ data ]
            with open( path , 'w' ) as file:
                for line in struct:
                    if kld.chk.is_lst( line ):
                        for val in line: file.write( str( float( val ) ) + ' ' )
                    else: file.write( str( float( line ) ) + ' ' )
                    file.write( '\n' )
        return struct
    def save_dict( self , path , data , struct ):
        if data is not None:
            with open( path , 'w' ) as file:
                for key in sorted( data ):
                    file.write( '--' + key + ' ' )
                    value = data[key]
                    if kld.chk.is_lst( value ):
                        for v in value: file.write( str( v ) + ' ' )
                        file.write( kld.aux.vartype( value[0] , short = True ) + '+' )
                    else:
                        file.write( str( value ) + ' ' )
                        file.write( kld.aux.vartype( value , short = True ) )
                    file.write( ' \n' )
            return data
        return struct

    def access( self , name , data , fnstart , fnsave ):
        if self.path is None: return data
        if kld.chk.is_tup( name ):
            values = []
            for i in kld.aux.rlen( name ):
                datai = data[i] if kld.chk.is_tup( data ) else data
                values.append( self.access( name[i] , datai , fnstart , fnsave ) )
            return values
        else:
            if name not in self.stored: fnstart( name , None )
            struct , path = self.stored[name]
            if data is not None:
                kld.pth.mkdir( kld.pth.dir( path ) )
                self.stored[name][0] = fnsave( path , data , struct )
            return self.stored[name][0]

    def scalar( self , name , data = None ):
        return self.access( name , data , self.start_scalar , self.save_scalar )
    def numpy( self , name , data = None ):
        return self.access( name , data , self.start_numpy , self.save_numpy )
    def list( self , name , data = None ):
        return self.access( name , data , self.start_list , self.save_list )
    def dict( self , name , data = None ):
        return self.access( name , data , self.start_dict , self.save_dict )
    def args( self , name , data = None ):
        return self.dict( name ) if data is None else self.dict( name , data.__dict__ )

##########################################################################3

##### FREESAVER
class FreeSaver( Saver ):
    def __init__( self , path , sess = None , restart = False ):
        Saver.__init__( self , path , sess , restart , free = True )
