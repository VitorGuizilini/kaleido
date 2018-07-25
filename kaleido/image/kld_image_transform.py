
import random
import scipy.misc
import numpy as np
import kaleido as kld

### SAMPLE PROB
def sample_prob( image , prob , seed = None ):
    if prob > 0.999: return image
    if len( image.shape ) == 3:
        return np.array( [ sample_prob( image[i] , prob , seed ) for i in range( image.shape[0] ) ] )
    else:
        rng = random.Random( seed )
        sampled = image.copy()
        idx = np.where( sampled > 0 )
        for i in range( idx[0].shape[0] ):
            if rng.random() > prob:
                sampled[ idx[0][i] , idx[1][i] ] = 0
        return sampled

### SAMPLE AREA
def sample_area( image , wdw ):
    wdw = int( wdw )
    if wdw == 0: return image
    if len( image.shape ) == 3:
        return np.array( [ sample_area( image[i] , wdw ) for i in range( image.shape[0] ) ] )
    sampled = image.copy()
    idx = np.where( sampled > 0 )
    for k in range( idx[0].shape[0] ):
        i , j = idx[0][k] , idx[1][k]
        if sampled[i,j] > 0:
            tmp = sampled[i,j]
            sampled[ max( 0 , i - wdw ) : min( sampled.shape[0] , i + wdw + 1 ) ,
                     max( 0 , j - wdw ) : min( sampled.shape[1] , j + wdw + 1 ) ] = 0
            sampled[i,j] = tmp
    return sampled

### BIN3D
def bin3d( image , bins ):
    if len( image.shape ) == 3:
        return [ bin3d( image[i] , bins ) for i in range( image.shape[0] ) ]
    else:
        image3d = np.zeros( ( bins , image.shape[0] , image.shape[1] ) )
        for i in range( image.shape[0] ):
            for j in range( image.shape[1] ):
                k = min( bins , int( ( bins + 1 ) * image[i,j] ) )
                if k > 0: image3d[ k - 1 , i , j ] = image[i,j]
        return image3d

### RESIZE
def do_resize( image , size , interp ):
    return scipy.misc.imresize( image , size , interp = interp )
def resize( image , size , interp = 'bilinear' ):
    if kld.chk.is_int( size ):
        h , w = image.shape[0:2] ; m = max( h , w )
        size = [ int( h / m * size ) , int( w / m * size ) ]
    return kld.aux.proc_seq( image , do_resize , size , interp )

### ROTATE
def do_rotate( image , angle , interp ):
    if angle == 0.0: return image
    return scipy.misc.imrotate( image , angle , interp = interp )
def rotate( image , angle , interp = 'bilinear' ):
    if kld.chk.is_seq( angle ):
        angle = kld.aux.randf( angle[0] , angle[1] )
    return kld.aux.proc_seq( image , do_rotate , angle , interp )

### ROTATE PROB
def rotate_prob( image , angle , interp = 'bilinear' ):
    angle = angle[0] if kld.aux.randf() < angle[1] else 0.0
    return kld.aux.proc_seq( image , do_rotate , angle , interp )

### CROP BORDERS
def do_crop_borders( image , h , w , t , b , l , r ):
    return image[ t : h - b , l : w - r ]
def crop_borders( image , t = 0 , b = 0 , l = 0 , r = 0 ):
    h , w = kld.list.shape( image , 0 , 2 )
    return kld.aux.proc_seq( image , do_crop_borders , h , w , t , b , l , r )

### CROP CENTER
def do_crop_center( image , h , w , size ):
    return image[ int( h/2 - size[0]/2 ) : int( h/2 + size[0]/2 ) ,
                  int( w/2 - size[1]/2 ) : int( w/2 + size[1]/2 ) ]
def crop_center( image , size ):
    size = kld.list.rep( size , 2 )
    h , w = kld.list.shape( image , 0 , 2 )
    return kld.aux.proc_seq( image , do_crop_center , h , w , size )

### CROP RANDOM
def do_crop_random( image , t , l , size ):
    return image[ t : t + size[0] , l : l + size[1] ]
def crop_random( image , size ):
    size = kld.list.rep( size , 2 )
    h , w = kld.list.shape( image , 0 , 2 )
    t = kld.aux.randi( 0 , h - size[0] )
    l = kld.aux.randi( 0 , w - size[1] )
    return kld.aux.proc_seq( image , do_crop_random , t , l , size )

### FLIP
def do_flip( image , axis , flip ):
    return np.flip( image , axis ) if flip else image
def flip( image , axis , prob = 1.0 ):
    return kld.aux.proc_seq( image , do_flip , axis , prob < kld.aux.randf() )

### MULT
def do_mult( image , mult , sat ):
    image = mult * image
    if sat is not None:
        if mult > 1.0: image[ np.where( image > sat ) ] = sat
        if mult < 1.0: image[ np.where( image < sat ) ] = sat
    return image
def mult( image , mult , sat = None ):
    if mult == 1.0: return image
    return kld.aux.proc_seq( image , do_mult , mult , sat )

### DIV
def div( image , div , sat = None ):
    return kld.image.mult( image , 1.0 / div , sat )


