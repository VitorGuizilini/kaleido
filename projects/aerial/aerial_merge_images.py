
import numpy as np
import kaleido as kld

######################

def draw( gtimage , gtlabel , recimage , reclabel ):
    gtover , recover = gtimage.copy() , recimage.copy()
    gtover[:,:,2] , recover[:,:,2] = gtlabel , reclabel
    return kld.plt.block( 2 , 3 , [  gtimage  , gtlabel  , gtover  ,
                                     recimage , reclabel , recover ] )

######################

parser = kld.mng.Parser( 'Image Merger for Aerial' )
parser.add_int(  'num'     , d = 6                   , h = 'Image number to be reconstructed')
parser.add_str(  'path'    , d = 'aerial/cerrado'    , h = 'Path to input data' )
parser.add_str(  'dataset' , d = '5_128_128_nearest' , h = 'Dataset name' )
parser.add_lstr( 'output'  , d = [ 'netC/trained01' , 'training' , 'valid' ] ,
                                                       h = 'Pointers to output log to be used' )
args = parser.args()

######################

path_gtimg = '../../data/%s/images/image_%02d.jpg' % ( args.path_data , args.num )
path_gtlbl = '../../data/%s/labels/label_%02d.jpg' % ( args.path_data , args.num )
path_rcimg = '../../data/%s_%s/images/%02d' % ( args.path_data , args.dataset , args.num )
path_rclbl = '../../logs/%s/%s/images/%s/individual/%s' % ( args.path_data , args.path_output[0] ,
                                                args.path_output[1] , args.path_output[2] )

scl , cx , cy = args.dataset.split('_')[:3]
scl , cx , cy = int( sc ) , int( cx ) , int( cy )
hx , hy = int( cx / 2 ) , int( cy / 2 )

resize = kld.partial( kld.img.resize , size = 1.0 / scl , interp = 'nearest' )
gtimage = kld.img.load( path_gtimg , 'rgbn'  , resize )
gtlabel = kld.img.load( path_gtlbl , 'grayn' , resize )

nx , ny = int( gtimage.shape[0] / hx ) , int( gtimage.shape[1] / hy )
shape = [ hx * nx , hy * ny ]

gtimage = gtimage[ :shape[0] , :shape[1] ]
gtlabel = gtlabel[ :shape[0] , :shape[1] ]

recimage = np.zeros( shape + [3] , dtype = np.float32 )
reclabel = np.zeros( shape       , dtype = np.float32 )

h , w , _ = gtimage.shape
kld.plt.adjust( w = 20 , p = h / w / 1.5 )
saver = kld.log.Saver( 'merger_%02d' % args.num , free = True )
saver.restart()

files_rcimg = kld.mng.Folder( path_rcimg ).files( pat = '*.png' )
recimages = kld.img.load( files_rcimg , 'rgbn'  )
folders_rclbl = kld.mng.Folder( path_rclbl , recurse = 1 )

for k in range( 200 ):

    str = '*_%04d_out.png' % k
    files_rclbl = folders_rclbl.files( pat = str )
    reclabels = kld.img.load( files_rclbl , 'grayn' )

    if len( reclabels ) > 0:

        print( '### CREATING %s %s from %s on %s for Epoch %04d' % (
                        args.path_data , args.num , args.dataset , args.path_output[0] , k ) )

        cnt = 0
        for i in range( nx - 1 ):
            for j in range( ny - 1 ):
                stx , sty = i * hx , j * hy
                fnx , fny = stx + cx , sty + cy
                recimage[ stx : fnx , sty : fny ] = recimages[cnt]
                reclabel[ stx : fnx , sty : fny ] = reclabels[cnt]
                cnt += 1

        saver.image( 'compares' , draw( gtimage , gtlabel , recimage , reclabel  ) ,
                                           'merge_%03d_%02d' % ( k , args.num ) )
        saver.image( 'epochs' , reclabel , 'image_%03d_%02d' % ( k , args.num ) )


