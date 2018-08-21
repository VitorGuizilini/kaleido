
from tqdm import tqdm
import kaleido as kld

##### BASEA
class baseA:

    ### PREPARE
    def prepare( self , args , path_load , path_save ):

        self.args = args
        self.sess = kld.tf.Session()
        self.loader = kld.log.Saver( path_load , self.sess )
        self.saver = kld.log.Saver( path_save , self.sess , restart = args.restart )
        self.start_epoch = int( self.loader.start_scalar( 'epoch' , 0 ) )
        self.phase = 'training' if args.train else 'testing'
        self.width = max( 100 , len( self.display() ) )
        self.saver.file( '' , kld.pth.callernames( 2 ) )
        self.saver.args( 'args' , args )
        self.build( args )

        try: args.plot_every *= args.eval_every
        except: pass

    ### LOOPEPOCH
    def loopEpoch( self , data , epoch , leave = True , enum = False ):
        text = kld.dsp.count( 'Epoch' , epoch , self.args.num_epochs )
        return kld.dsp.loopBar( data.range_batches() , text , self.width , leave , enum )

    ### LOOPEVAL
    def loopEval( self , data , text , leave = True , enum = False ):
        if not kld.chk.is_lst( data ) and not kld.chk.is_int( data ):
            data = data.range_batches()
        text = 'Eval {}'.format( text.upper() )
        return kld.dsp.loopBar( data , text , self.width , leave , enum )

######################################################

