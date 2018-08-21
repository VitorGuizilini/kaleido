
import cv2
import random
import numpy as np
import kaleido as kld
from kaleido.chk import *

########################### TUPLES

### VALCALC
def valcalc( val , dim ):
    if is_lst( val ):
        val = [ valcalc( val[i] , dim ) for i in range( 2 ) ]
        return kld.rnd.i( val )
    else:
        if is_flt( val ): return int( val * dim )
        else: return val if val >= 0 else dim + val + 1

### VALSCALE
def valscale( val , shape ):
    big = max( shape[0] , shape[1] )
    if val < 0: val = val + big + 1
    return ( int( shape[0] * val / big ) ,
             int( shape[1] * val / big ) )

########################### PREPS

### PREP TUPLE
def prep_tuple( val , shape , no_zero = True , swap = False ):
    if is_lst( shape ): shape = shape[0]
    if is_npy( shape ): shape = shape.shape
    if not is_seq( val ):
        if is_int( val ): val = valscale( val , shape )
        else: val = [ int( val * shape[i] ) for i in range( 2 ) ]
    elif is_lst( val ):
        big = max( shape[0] , shape[1] )
        val = [ valcalc( val[i] , big ) for i in range( 2 ) ]
        val = valscale( kld.rnd.i( val ) , shape )
    else:
        if val[0] == 0 and no_zero:
            base = valcalc( val[1] , shape[1] )
            val = ( int( base * shape[0] / shape[1] ) , base )
        elif val[1] == 0 and no_zero:
            base = valcalc( val[0] , shape[0] )
            val = ( base , int( base * shape[1] / shape[0] ) )
        else: val = [ valcalc( val[i] , shape[i] ) for i in range( 2 ) ]
    return tuple( val ) if not swap else ( val[1] , val[0] )

### PREP INTERP
def prep_interp( interp ):
    if interp == 'nearest'  : interp = cv2.INTER_NEAREST
    if interp == 'bilinear' : interp = cv2.INTER_LINEAR
    if interp == 'bicubic'  : interp = cv2.INTER_CUBIC
    return interp

########################### RESIZES

### DO RESIZE
def do_resize( image , size , interp , **kwargs ):
    return cv2.resize( image , size , interpolation = interp , **kwargs )
### RESIZE
def resize( image , size , interp = 'nearest' , **kwargs ):
    size , interp = prep_tuple( size , image , swap = True ) , prep_interp( interp )
    return kld.process( image , do_resize , size , interp , **kwargs )

########################### FLIPS

### DO FLIP
def do_flip( image , axis ):
    return np.flip( image , axis )
### FLIP
def flip( image , axis , prob ):
    if prob is not None and kld.rnd.f() < prob: return image
    return kld.process( image , do_flip , axis )
def fliptb( image , prob = None ): return flip( image , 0 , prob )
def fliplr( image , prob = None ): return flip( image , 1 , prob )

########################### TRIMS

### DO TRIM
def do_trim( image , h , w , t , b , l , r ):
    return image[ t : h - b , l : w - r ]
### TRIM
def trim( image , t = 0 , b = 0 , l = 0 , r = 0 , prob = None ):
    if prob is not None and kld.rnd.f() < prob: return image
    h , w = image[0].shape[:2] if is_lst( image ) else image.shape[:2]
    t , b = valcalc( t , h ) , valcalc( b , h )
    l , r = valcalc( l , w ) , valcalc( r , w )
    return kld.process( image , do_trim , h , w , t , b , l , r )

########################### CROPS

### DO CROP
def do_crop( image , center , size ):
    return image[ int( center[0] - size[0] / 2 ) : int( center[0] + size[0] / 2 ) ,
                  int( center[1] - size[1] / 2 ) : int( center[1] + size[1] / 2 ) ]
### CROP
def crop( image , size , center = None , prob = None ):
    if prob is not None and kld.rnd.f() < prob: return image
    h , w = image[0].shape[:2] if is_lst( image ) else image.shape[:2]
    size = prep_tuple( size , image )
    shape = ( h - size[0] , w - size[1] )
    if center is None:
          center = ( kld.rnd.i( 0 , shape[0] ) , kld.rnd.i( 0 , shape[1] ) )
    else: center = prep_tuple( center , shape , no_zero = False )
    center = ( center[0] + size[0] / 2 , center[1] + size[1] / 2 )
    return kld.process( image , do_crop , center , size )

########################### ROTATES

### DO ROTATE
def do_rotate( image , angle , center , fit , **kwargs ):
    if angle == 90:
        axes = [ 1 , 0 , 2 ] if len( image.shape ) == 3 else [ 1 , 0 ]
        return np.transpose( image , axes )
    elif angle == 180:
        return np.flip( np.flip( image , 0 ) , 1 )
    elif angle == 270:
        axes = [ 1 , 0 , 2 ] if len( image.shape ) == 3 else [ 1 , 0 ]
        return np.transpose( np.flip( np.flip( image , 0 ) , 1 ) , axes )
    else:
        size = ( image.shape[1] , image.shape[0] )
        M = cv2.getRotationMatrix2D( center , angle , 1.0 )
        if fit:
            old_size , rad = size , np.deg2rad( angle )
            sinr , cosr = np.sin( rad ) , np.cos( rad )
            size = ( int( abs( sinr * size[1] ) + abs( cosr * size[0] ) ) ,
                     int( abs( sinr * size[0] ) + abs( cosr * size[1] ) ) )
            M[0,2] += ( size[0] - old_size[0] ) / 2
            M[1,2] += ( size[1] - old_size[1] ) / 2
        return cv2.warpAffine(image , M , size , **kwargs )
### ROTATE
def rotate( image , angle , center = 0.5 , prob = None , fit = False , **kwargs ):
    if prob is not None and kld.rnd.f() < prob: return image
    if kld.chk.is_lst( angle ): angle = kld.rnd.f( angle[0] , angle[1] )
    if angle == 0.0: return image
    center = prep_tuple( 0.5 if fit else center , image , no_zero = False , swap = True )
    return kld.process( image , do_rotate , angle , center , fit , **kwargs )

########################### MATHS

### DO MATH
def do_math( image , add , mlt , sat ):
    image = image * mlt + add
    if sat is not None:
        if sat[0] is not None: image[ np.where( image < sat[0] ) ] = sat[0]
        if sat[1] is not None: image[ np.where( image > sat[1] ) ] = sat[1]
    return image
### MATH
def math( image , add = 0.0 , mlt = 1.0 , sat = None , prob = None ):
    if prob is not None and kld.rnd.f() < prob: return image
    if add == 0.0 and mlt == 1.0: return image
    if kld.chk.is_lst( add ): add = kld.rnd.f( add[0] , add[1] )
    if kld.chk.is_lst( mlt ): mlt = kld.rnd.f( mlt[0] , mlt[1] )
    return kld.process( image , do_math , add , mlt , sat )

########################### CONVERTS

### NORM
def norm( image , type ):
    if image.dtype == np.uint8: return image
    a , b , c = cv2.split( image )
    if type in [ 'hsv' , 'hls' ]:
        image = cv2.merge( [ a / 360.0 , b , c ] )
    elif type in [ 'lab' ]:
        image = cv2.merge( [ a / 100.0 , ( b + 127.0 ) / 254.0 , ( c + 127.0 ) / 254.0 ] )
    elif type in [ 'luv' ]:
        image = cv2.merge( [ a / 100.0 , ( b + 134.0 ) / 354.0 , ( c + 140.0 ) / 262.0 ] )
    return image

### UNNORM
def unnorm( image , type ):
    if image.dtype == np.uint8: return image
    a , b , c = cv2.split( image )
    if type in [ 'hsv' , 'hls' ]:
        image = cv2.merge( [ a * 360.0 , b , c ] )
    elif type in [ 'lab' ]:
        image = cv2.merge( [ a * 100.0 , b * 254.0 - 127.0 , c * 254.0 - 127.0 ] )
    elif type in [ 'luv' ]:
        image = cv2.merge( [ a * 100.0 , b * 354.0 - 134.0 , c * 262.0 - 140.0 ] )
    return image

### DO CONVERT
def do_convert( image , map , mode , **kwargs ):
    image = unnorm( image , map[:3] )
    image = cv2.cvtColor( image , mode , **kwargs )
    image = norm( image , map[4:] )
    return image
### CONVERT
def convert( image , map , **kwargs ):
    if   map == 'rgb2hsv': mode = cv2.COLOR_RGB2HSV
    elif map == 'hsv2rgb': mode = cv2.COLOR_HSV2RGB
    elif map == 'rgb2hls': mode = cv2.COLOR_RGB2HLS
    elif map == 'hls2rgb': mode = cv2.COLOR_HLS2RGB
    elif map == 'rgb2lab': mode = cv2.COLOR_RGB2LAB
    elif map == 'lab2rgb': mode = cv2.COLOR_LAB2RGB
    elif map == 'rgb2luv': mode = cv2.COLOR_RGB2LUV
    elif map == 'luv2rgb': mode = cv2.COLOR_LUV2RGB
    return kld.process( image , do_convert , map , mode , **kwargs )

########################### SAMPLES

### SAMPLE PROB
def sample_prob( image , prob , seed = None ):
    if is_lst( image ):
        return [ sample_prob( im , prob , seed ) for im in image ]
    else:
        sampled = image.copy()
        if prob > 0.999: return sampled
        rng = random.Random( seed )
        idx = np.where( sampled > 0 )
        for i in range( idx[0].shape[0] ):
            if rng.random() > prob:
                sampled[ idx[0][i] , idx[1][i] ] = 0
        return sampled

### SAMPLE AREA
def sample_area( image , wdw , seed = None ):
    wdw = int( wdw )
    if is_lst( image ):
        return [ sample_area( im , wdw , seed ) for im in image ]
    else:
        sampled = image.copy()
        if wdw == 0: return sampled
        rng = random.Random( seed )
        idxi , idxj = list( np.where( sampled > 0 ) )
        rnd = np.arange( len( idxi ) ) ; rng.shuffle( rnd )
        idxi , idxj = idxi[rnd] , idxj[rnd]
        for k in range( idxi.shape[0] ):
            i , j = idxi[k] , idxj[k]
            if sampled[i,j] > 0:
                sampled[ max( 0 , i - wdw ) : min( sampled.shape[0] , i + wdw + 1 ) ,
                         max( 0 , j - wdw ) : min( sampled.shape[1] , j + wdw + 1 ) ] = 0
                sampled[i,j] = image[i][j]
        return sampled


