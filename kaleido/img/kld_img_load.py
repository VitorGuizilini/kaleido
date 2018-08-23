
import cv2
import numpy as np
import kaleido as kld

### LOAD
def load( file , color = None , ops = None ):
    if kld.chk.is_lst( file ):
        if kld.chk.is_lst( color ):
              return [ kld.img.load( file[i] , color[i] , ops ) for i in range( len( file ) ) ]
        else: return [ kld.img.load( f , color , ops ) for f in file ]
    else:
        norm = color is not None and color[-1] == 'n'
        if norm: color = color[:-1]
        if color in [ 'bgr' , 'rgb' , 'hsv' , 'lab' ]:
            mode = cv2.IMREAD_COLOR
        elif color in [ 'gray' ]:
            mode = cv2.IMREAD_GRAYSCALE
        else:
            mode = cv2.IMREAD_UNCHANGED
        image = cv2.imread( file , mode )
        if norm: image = ( image / 255.0 ).astype( np.float32 )
        if color == 'rgb': image = image[:,:,::-1]
        if color == 'hsv': image = cv2.cvtColor( image , cv2.COLOR_BGR2HSV )
        if color == 'lab': image = cv2.cvtColor( image , cv2.COLOR_BGR2LAB )
        return kld.apply( image , ops )
