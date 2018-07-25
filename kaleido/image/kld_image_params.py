
import scipy.misc
import numpy as np
import kaleido as kld

### PREP PARAMS
def prep_params( params ):
    params = kld.list.make( params )
    for i in range( len( params ) ):
        if not kld.chk.is_class( params[i] , Params ):
            params[i] = kld.image.Params( ' '.join( kld.list.make( params[i] ) ) )
    return params[0] if len( params ) == 1 else params

### DATA FROM ARGS
def data_from_args( *args ):
    data = []
    for arg in args:
        if kld.chk.is_class( arg , Params ): data.append( arg.data )
        elif kld.chk.is_str( arg ): data.append( arg.split( ' ' ) )
        else: data.append( arg )
    return kld.list.flatten( data )

### GET INTERP
def get_interp( data , i ):
    interp = 'bilinear'
    if len( data ) > i+1:
        if data[i+1] == 'nearest' or data[i+1] == 'bilinear':
            interp = data[i+1] ; i += 1
    return interp , i

##### PARAMS
class Params:

    ### __INIT__
    def __init__( self , *args ):
        self.prepare( *args )

    ### PREPARE
    def prepare( self , *args ):
        i , params , actions = 0 , {} , []
        self.data = data = data_from_args( *args )
        self.size = size = None
        while i < len( data ):
            ### LOADS
            if   data[i] == 'rgb' : self.color = 'RGB'
            elif data[i] == 'mono': self.color = 'L'
            ### ACTIONS
            elif data[i] == 'size':
                if not kld.chk.is_str_int( data[i+1] ): size = float( data[i+1] ) ; i += 1
                elif len( data ) > i+2 and kld.chk.is_str_int( data[i+2] ):
                    size = [ int( data[i+1] ) , int( data[i+2] ) ] ; i += 2
                else: size = int( data[i+1] ) ; i += 1
                interp , i = get_interp( data , i )
                actions.append( [ kld.image.resize , { 'size' : kld.aux.copy( size ) , 'interp' : interp } ] )
            elif data[i] == 'roti' or data[i] == 'rotp':
                func = kld.image.rotate if data[i] == 'roti' else kld.image.rotate_prob
                if len( data ) > i+2 and kld.chk.is_str_float( data[i+2] ):
                    angle = [ float( data[i+1] ) , float( data[i+2] ) ] ; i += 2
                else: angle = float( data[i+1] ) ; i += 1
                interp , i = get_interp( data , i )
                actions.append( [ func , { 'angle' : angle , 'interp' : interp } ] )
            elif data[i] == 'crop':
                t , b , l , r = int( data[i+1] ) , int( data[i+2] ) , int( data[i+3] ) , int( data[i+4] )
                actions.append( [ kld.image.crop_borders , { 't' : t , 'b' : b , 'l' : l , 'r' : r } ] ) ; i += 4
                kld.list.add( self.size , 0 , - t - b ) ; kld.list.add( size , 1 , - l - r )
            elif data[i] == 'cropt':
                t = int( data[i+1] ) ; kld.list.add( size , 0 , -t )
                actions.append( [ kld.image.crop_borders , { 't' : t } ] ) ; i += 1
            elif data[i] == 'cropb':
                b = int( data[i+1] ) ; kld.list.add( size , 0 , -b )
                actions.append( [ kld.image.crop_borders , { 'b' : b } ] ) ; i += 1
            elif data[i] == 'cropl':
                l = int( data[i+1] ) ; kld.list.add( size , 1 , -l )
                actions.append( [ kld.image.crop_borders , { 'l' : l } ] ) ; i += 1
            elif data[i] == 'cropr':
                r = int( data[i+1] ) ; kld.list.add( size , 1 , -r )
                actions.append( [ kld.image.crop_borders , { 'r' : r } ] ) ; i += 1
            elif data[i] == 'cropcnt':
                if len( data ) > i+2 and kld.chk.is_str_int( data[i+2] ):
                    size = [ int( data[i+1] ) , int( data[i+2] ) ] ; i += 2
                else: size = [ int( data[i+1] ) ] * 2 ; i += 1
                actions.append( [ kld.image.crop_center , { 'size' : kld.aux.copy( size ) } ] )
            elif data[i] == 'croprnd':
                if len( data ) > i+2 and kld.chk.is_str_int( data[i+2] ):
                    size = [ int( data[i+1] ) , int( data[i+2] ) ] ; i += 2
                else: size = [ int( data[i+1] ) ] * 2 ; i += 1
                actions.append( [ kld.image.crop_random , { 'size' : kld.aux.copy( size ) } ] )
            elif data[i] == 'fliptb':
                prob = float( data[i+1] ) if len( data ) > i+1 and kld.chk.is_str_float( data[i+1] ) else 1.0
                actions.append( [ kld.image.flip , { 'axis' : 0 , 'prob' : prob } ] )
            elif data[i] == 'fliplr':
                prob = float( data[i+1] ) if len( data ) > i+1 and kld.chk.is_str_float( data[i+1] ) else 1.0
                actions.append( [ kld.image.flip , { 'axis' : 1 , 'prob' : prob } ] )
            elif data[i] == 'mult':
                mult , sat = float( data[i+1] ) , None ; i += 1
                if len( data ) > i+1 and kld.chk.is_str_float( data[i+1] ):
                    sat = float( data[i+1] ) ; i += 1
                actions.append( [ kld.image.mult , { 'mult' : mult , 'sat' : sat } ] )
            elif data[i] == 'div':
                div , sat = float( data[i+1] ) , None ; i += 1
                if len( data ) > i+1 and kld.chk.is_str_float( data[i+1] ):
                    sat = float( data[i+1] ) ; i += 1
                actions.append( [ kld.image.div , { 'div' : div , 'sat' : sat } ] )
            i += 1
        self.actions = actions
        self.size = size
