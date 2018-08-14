
import numpy as np
import kaleido as kld
from aerial_model import Model

######################

### COLORS
def colors( images ):
    actions = [ ( kld.img.convert , { 'map' : 'rgb2hsv' } ) ,
                ( kld.img.convert , { 'map' : 'rgb2lab' } ) ]
    nimages = images.copy()
    for act in actions:
        temp = kld.apply( images , act )
        for i in range( len( images ) ):
            nimages[i] = np.concatenate( [ nimages[i] , temp[i] ] , axis = 2 )
    return nimages

### AUGMENT
def augment( images , labels ):
    actions = [ ( kld.img.fliplr ) ,
                ( kld.img.fliptb ) ,
                ( kld.img.rotate , { 'angle' : 90  } ) ,
                ( kld.img.rotate , { 'angle' : 180 } ) ,
                ( kld.img.rotate , { 'angle' : 270 } ) ]
    nimages , nlabels = images.copy() , labels.copy()
    for act in actions:
        nimages += kld.apply( images , act )
        nlabels += kld.apply( labels , act )
    return nimages , nlabels

######################

parser = kld.mng.Parser( 'Aerial Cerrado' )
kld.tf.models.argsA( parser )

parser.add_rstr( 'net'     , d = None       , h = 'Version of the network used'        )
parser.add_str(  'dataset' , d = 'original' , h = 'Dataset to be used (cerrado_)'      )
parser.add_bol(  'augment' , d = False      , h = 'Augment training data'              )
parser.add_bol(  'colors'  , d = False      , h = 'Include other colorspaces'          )

parser.add_lint( 'num_data'    , d = [0,0] , h = 'Number of sampled Train/Valid points' , q = 2 )
parser.add_lint( 'batch_sizes' , d = [1,1] , h = 'Batch size for Train/Valid data'      , q = 2 )

parser.add_lbol( 'eval_draw' , d = [False,True]      , h = 'Train/Valid draw flags' , q = 2 )
parser.add_lstr( 'eval_capt' , d = ['train','valid'] , h = 'Train/Valid captions'   , q = 2 )

parser.add_int( 'eval_num' , d = 6 , h = 'Folder to use for validation' )
parser.add_str( 'log_path' , d = '../../logs/aerial/cerrado/' , h = 'Folder where information is stored' )

args = parser.args()

######################

data_path = '../../data/aerial/cerrado_' + args.dataset
folders = kld.mng.Folder( data_path , recurse = 2 )
images , labels = folders.split( [ 'images' , 'labels' ] )

idx_valid = [ 6 ]
images_train , images_valid = images.split_files( idx_valid , [ '*.png' , '*.jpg' ] )
labels_train , labels_valid = labels.split_files( idx_valid , [ '*.png' , '*.jpg' ] )

[ images_train , labels_train ] = kld.lst.sample( [ images_train , labels_train ] , args.num_data[0] )
[ images_valid , labels_valid ] = kld.lst.sample( [ images_valid , labels_valid ] , args.num_data[1] )

[ images_train , images_valid ] = kld.img.load( [ images_train , images_valid ] , 'rgbn'  )
[ labels_train , labels_valid ] = kld.img.load( [ labels_train , labels_valid ] , 'grayn' )

if args.colors:  images_train , images_valid = colors( images_train ) , colors( images_valid )
if args.augment: images_train , labels_train = augment( images_train , labels_train )

######################

model = Model( [ images_train , labels_train ] ,
               [ images_valid , labels_valid ] , args )
if args.train: model.train()
else: model.test()
