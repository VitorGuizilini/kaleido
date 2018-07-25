
import numpy as np

##### LRATE
class LRate:

    ### __INIT__
    def __init__( self , type , value = 0.0 , start = 0.0 , finish    = 0.0 ,
                                drop  = 0.5 , jump  =  0  , steep     =  1  ,
                                wait  =  0  , step  =  0  , num_steps =  0  ):

        if wait > 0 and wait < 1:
            wait = int( wait * num_steps )

        self.drate  = start - finish
        self.dtotal = num_steps - wait - 1
        self.stfn , self.fnst = start / finish , finish / start

        if jump > 0 and jump < 1:
            jump = int( jump * self.dtotal )
        steep *= 10.0 / self.dtotal

        self.middle = self.dtotal / 2.0
        self.beg = start - self.drate / ( 1 + np.exp( - steep * (             - self.middle ) ) )
        self.end = start - self.drate / ( 1 + np.exp( - steep * ( self.dtotal - self.middle ) ) )

        self.type , self.value , self.start , self.finish = type , value , start , finish
        self.wait , self.step , self.num_steps = wait , step , num_steps
        self.drop , self.jump , self.steep = drop , jump , steep

    ### CURR STEP
    def curr_step( self ):
        return self.step

    ### NEXT
    def next( self ):

        self.step += 1
        if self.type == 'constant': return self.value
        if self.step <= self.wait: return self.start
        dstep = self.step - self.wait - 1

        if self.type == 'discrete':
            return self.start * np.power( self.drop , math.floor( ( dstep ) / self.jump ) )
        if self.type == 'linear':
            return self.start - self.drate * dstep / self.dtotal
        if self.type == 'timedecay':
            decay = ( self.stfn - 1.0 ) / self.dtotal
            return self.start / ( 1.0 + decay * dstep )
        if self.type == 'expdecay':
            decay = - ( np.log( self.fnst ) ) / self.dtotal
            return self.start * np.exp( - decay * dstep )
        if self.type == 'logistic':
            val = self.start - self.drate / ( 1 + np.exp( - self.steep * ( dstep - self.middle ) ) )
            return ( val - self.end ) / ( self.beg - self.end ) * self.drate + self.finish
