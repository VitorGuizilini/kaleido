
import scipy.misc
import numpy as np
import kaleido as kld

### MAKE PARAMS
def make_params( params ):
    if kld.chk.is_seq( params ): return [ make_params( param ) for param in params ]
    else: return kld.image.Params( params ) if kld.chk.is_str( params ) else params

### LOAD
def load( file , params ):
    params = make_params( params )
    if kld.chk.is_seq( file ):
        if kld.chk.is_seq( params ):
              return [ kld.image.load( file[i] , params[i] ) for i in range( len( file ) ) ]
        else: return [ kld.image.load( f , params ) for f in file ]
    else:
        params = make_params( params )
        image = scipy.misc.imread( file , mode = params.color ).astype( np.float32 )
        return kld.image.process( image , params )

### PROCESS
def process( image , params ):
    params = make_params( params )
    if kld.chk.is_seq( image ):
        if kld.chk.is_seq( params ):
              return [ kld.image.process( image[i] , params[i] ) for i in range( len( image ) ) ]
        else: return [ kld.image.process( im , params ) for im in image ]
    else:
        params = make_params( params )
        for action in params.actions:
            image = action[0]( image , **action[1] )
        return image

### LPROCESS
def lprocess( batch , params ):
    params , new_batch = make_params( params ) , []
    for j in range( len( batch ) ): new_batch.append( [] )
    for i in range( len( batch[0] ) ):
        line = []
        for j in range( len( batch ) ): line.append( batch[j][i] )
        for action in params.actions: line = action[0]( line , **action[1] )
        for j in range( len( batch ) ): new_batch[j].append( line[j] )
    return new_batch

### LOAD SHAPE
def load_shape( file , params ):
    params = make_params( params )
    return list( kld.image.load( file , params ).shape )

### SAVE
def save( image , file ):
    if kld.chk.is_seq( image ):
        [ kld.image.save( image[i] , file + str(i) ) for i in range( len( image ) ) ]
    else:
        kld.path.mkdir( kld.path.dir( file ) )
        if file[-4] is not '.': file += '.jpg'
        scipy.misc.imsave( file , image )


