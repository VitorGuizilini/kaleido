
import time

##### TIMER
class Timer:

    ### __INIT__
    def __init__( self ):
        self.time = time.time()

    ### RESET
    def reset( self ):
        curr = time.time() - self.time
        self.time = time.time()
        return curr

    ### TICK
    def tick( self ):
        return time.time() - self.time

    ### REACH
    def reach( self , sec = 0 , min = 0 , hour = 0 ):
        tot = sec + 60 * ( min + 60 * hour )
        return self.time > tot

### TIMERS
def Timers( n ):
    return [ Timer() for _ in range( n ) ]

