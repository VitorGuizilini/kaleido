
import os
import importlib
import kaleido as kld

### MKDIR
def mkdir( folder ):
    if not os.path.exists( folder ): os.makedirs( folder )

### RMFILE
def rmfile( file ):
    if os.path.isfile( file ): os.remove( file )

### LEVEL
def level( path , n = 0 ):
    return path.split('/')[-n-1]

### NAME
def name( path ):
    name = level( path )
    if name[-4] == '.': name = name[:-4]
    return name

### DIR
def dir( path , n = 0 ):
    return '/'.join( path.split('/')[:-n-1] )

### MODULE
def module( file , attr = None ):
    mod = importlib.import_module( file )
    return mod if attr is None else getattr( mod , attr )
