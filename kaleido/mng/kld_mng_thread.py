
import threading
import kaleido as kld

class Thread:

    ### LIST
    def list( self , lst , func , n , *args , **kwargs ):
        self.data , self.n = lst , n

        d = len( lst )
        b = d // n

        self.lst = [ [] for _ in range( d ) ]

        self.threads = []
        for i in range( n ):
            st , fn = ( i ) * b , ( i + 1 ) * b
            self.threads.append( threading.Thread( target = self.iter ,
                                args = ( func , lst , st , fn ) , kwargs = kwargs ) )
            self.threads[-1].start()

    def iter( self , func , lst , st , fn , *args , **kwargs ):
        self.lst[st:fn] = func( lst[st:fn] , *args , **kwargs )

    ### JOIN
    def join( self ):
        if kld.chk.is_lst( self.threads ):
            [ thread.join() for thread in self.threads ]
        else: self.threads.join()
        return self.lst
