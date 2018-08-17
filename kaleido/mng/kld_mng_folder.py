
import glob
import kaleido as kld

### READ FILES
def read_files( folder , pats = '*' , sort = True , only_names = False ):
    pats = kld.lst.make( pats )
    files = kld.lst.flatten( [ glob.glob( folder + '/' + pat ) for pat in pats ] )
    if sort:
        files = sorted( files )
    if only_names:
        for ( i , f ) in enumerate( files ):
            files[i] = f.split('/')[-1]
    return files

##### FOLDER
class Folder:

    ### __INIT__
    def __init__( self , paths , recurse = 0 ):
        self.folders = kld.lst.flatten( kld.lst.make( paths ) )
        if recurse > 0: self.recurse( recurse )

    def __getitem__( self , n ): return self.folders[n]
    def data( self ): return self.folders
    def size( self ): return len( self.folders )

    ### PATH
    def path( self , idx , level = None ):
        if level is None: return self.folders[idx]
        else: return self.folders[idx].split('/')[-level-1]

    ### RECURSE
    def recurse( self , level ):
        new_folders = []
        for folder in self.folders:
            files = read_files( folder , '*/' )
            for n in range( level - 1 ):
                files = kld.lst.flatten( files )
                for i in range( len( files ) ):
                    files[i] = read_files( files[i] , '*/' )
            new_folders.append( files )
        self.folders = kld.lst.flatten( new_folders )

    ### STR TO IDX
    def str_to_idx( self , idx ):
        idx = kld.lst.make( idx )
        for i , val in enumerate( idx ):
            if kld.chk.is_str( val ):
                for j , folder in enumerate( self.folders ):
                    if val in folder:
                        idx.append( j )
                idx.pop( i )
        return idx

    ### APPEND
    def append( self , append ):
        for i in range( len( self.folders ) ):
            if self.folders[i][-1] is not '/':
                self.folders[i] += '/'
            self.folders[i] += append

    ### SPLIT
    def split( self , paths ):
        paths = kld.lst.make( paths )
        if len( paths ) == 0: return [ self.keep( i ) for i in range( self.size() ) ]
        if len( paths ) == 1: return self.remove( paths[0] ) , self.keep( paths[0] )
        else: return [ self.keep( path ) for path in paths ]

    ### KEEP
    def keep( self , paths ):
        paths = kld.lst.make( paths )
        folders = [ Folder( [ self.folders[i] for i in self.str_to_idx( path ) ] ) for path in paths ]
        return folders[0] if len( folders ) == 1 else folders

    ### REMOVE
    def remove( self , paths ):
        paths , folders = kld.lst.make( paths ) , []
        for path in paths:
            rem = self.str_to_idx( path )
            idx = list( range( self.size() ) )
            rem.sort( reverse = True )
            for r in rem: idx.pop( r )
            folders.append( Folder( [ self.folders[i] for i in idx ] ) )
        return folders[0] if len( folders ) == 1 else folders

    ### FILES
    def files( self , paths = '' , pat = '*' , max = None ):
        files, paths = [] , kld.lst.make( paths )
        for i in range( len( paths ) ):
            filesi = []
            for f in self.folders:
                filesi.append( read_files( f + '/' + paths[i] , pat ) )
            files.append( kld.lst.flatten( filesi ) )
        if max is not None and max > 0: files = [ files[i][:max] for i in range( len( files ) ) ]
        return files[0] if len( files ) == 1 else files

    ### KEEP FILES
    def keep_files( self , paths , pat = '*' , max = None ):
        keeps = self.keep( paths )
        files = [ keep.files( pat = pat , max = max ) for keep in keeps ]
        return files[0] if len( files ) == 1 else files

    ### REMOVE FILES
    def remove_files( self , paths , pat = '*' , max = None ):
        rems = self.remove( paths )
        files = [ rem.files( pat = pat , max = max ) for rem in rems ]
        return files[0] if len( files ) == 1 else files

    ### SPLIT FILES
    def split_files( self , paths , pat = '*' , max = None ):
        splits = self.split( paths )
        files = [ split.files( pat = pat , max = max ) for split in splits ]
        return files[0] if len( files ) == 1 else files

