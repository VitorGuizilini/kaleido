
import glob
import kaleido as kld

### READ FILES
def read_files( folder , pattern = '*' , sort = True , only_names = False ):
    files = glob.glob( folder + '/' + pattern )
    if sort: # Sort files
        files = sorted( files )
    if only_names: # Remove path from files
        for ( i , f ) in enumerate( files ):
            files[i] = f.split('/')[-1]
    return files

##### FOLDER
class Folder:

    ### __INIT__
    def __init__( self , *args ):
        self.folders = kld.list.make( kld.aux.join( args ) )

    ### PRINT
    def print( self ):
        print( self.folders )

    ### SIZE
    def size( self ):
        return len( self.folders )

    ### PATH
    def path( self , idx , level = None ):
        if level is None: return self.folders[idx]
        else: return self.folders[idx].split('/')[-level-1]

    ### __GETITEM__
    def __getitem__( self , n ):
        return self.folders[n]

    ### RECURSE
    def recurse( self , level ):
        new_folders = []
        for folder in self.folders:
            files = read_files( folder , '*/' )
            for n in range( level - 1 ):
                files = kld.list.flatten( files )
                for i in range( len( files ) ):
                    files[i] = read_files( files[i] , '*/' )
            new_folders.append( files )
        self.folders = kld.list.flatten( new_folders )

    ### APPEND
    def append( self , append ):
        for i in range( len( self.folders ) ):
            if self.folders[i][-1] is not '/':
                self.folders[i] += '/'
            self.folders[i] += append

    ### SEPARATE
    def separate( self , idx = None ):
        if idx is None: return [ self.keep( i ) for i in range( self.size() ) ]
        else: return self.remove( idx ) , self.keep( idx )

    ### KEEP
    def keep( self , idx ):
        idx = kld.list.make( idx )
        return Folder( [ self.folders[i] for i in idx ] )

    ### REMOVE
    def remove( self , rem ):
        rem = kld.list.make( rem )
        idx = list( range( self.size() ) )
        rem.sort( reverse = True )
        for r in rem: idx.pop( r )
        return Folder( [ self.folders[i] for i in idx ] )

    ### FILES FROM
    def files_from( self , paths = '' , pattern = '*' , max = None ):
        files, paths = [] , kld.list.make( paths )
        for i in range( len( paths ) ):
            filesi = []
            for f in self.folders:
                filesi.append( read_files( f + '/' + paths[i] , pattern ) )
            files.append( kld.list.flatten( filesi ) )
        if max is not None: files = [ files[i][:max] for i in range( len( files ) ) ]
        if len( files ) == 1: return files[0]
        else: return files
