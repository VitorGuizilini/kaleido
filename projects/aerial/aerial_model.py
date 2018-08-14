
import kaleido as kld

##### MODEL
class Model( kld.tf.models.baseA ):

######################################################

    ### __INIT__
    def __init__( self , data_train , data_valid , args ):

        path = '../../logs/%s/version%s/' % ( args.path , args.version )
        args.load_path = None if args.load is None else path + args.load
        args.save_path = None if args.save is None else path + args.save

        self.data_train = kld.mng.MultiBatch( data_train , batch_size = args.batch_sizes[0] )
        self.data_valid = kld.mng.MultiBatch( data_valid , batch_size = args.batch_sizes[1] )
        self.data = [ self.data_train , self.data_valid ]

        net_name = 'aerial_network' + args.version
        net = kld.pth.module( 'networks.' + net_name , 'Network' )
        self.net = net( self.data_train )

        self.args = args
        self.postInit( files = [ kld.pth.fname( __file__ ) ,
                                 'networks/' + net_name + '.py' ] )

        h , w , _ = self.data_train.shape(0)
        kld.plt.adjust( w = 20 , p = h / w / 1.5 )

######################################################

    ### PREPARE_PLOT
    def prepare_plot( self , image , label , output ):

        imlabel = image.copy();  imlabel[:,:,2] = label
        imoutput = image.copy(); imoutput[:,:,2] = output
        plt = kld.plt.block( 2 , 3 , [ image[:,:,:3] , label  , imlabel[:,:,:3] ,
                                       image[:,:,:3] , output , imoutput[:,:,:3] ] )
        return plt

######################################################

    ### DISPLAY
    def display( self , name , epoch , laprf = [ 0 ] * 5 ):
        str =  kld.dsp.count( name , epoch , self.args.num_epochs )
        return '{} | Loss: {:<12.10f} ' \
                  '| Acc.: {:<8.6f} | Prec.: {:<8.6f} ' \
                  '| Recl.: {:<8.6f} | F-Meas.: {:<8.6f}'.format(
                    str , laprf[0] , laprf[1] , laprf[2] , laprf[3] , laprf[4] )

    ### DRAW FIRST
    def draw_first( self , data , caption ):

        _ , label = data
        for j in range( len( label ) ):
            file = '%03d_%04d_lbl' % ( j , 0 )
            folder = 'individual/%s/%03d' % ( caption , j )
            self.saver.image( self.phase , label[j] , file , folder )

######################################################

    ### OPTIMIZE LOOP
    def optimize_loop( self , data , epoch , lrate ):

        data.reset( shuffle = True )
        for i in self.loopOptim( data , epoch ):
            image , label = data.next_batch()
            self.sess.run( self.net.optim ,
                    { self.net.input : image , self.net.label : label ,
                      self.net.drop : self.net.dropval , self.net.phase : True ,
                      self.net.lrate : lrate } )

    ### EVALUATE LOOP
    def evaluate_loop( self , data , epoch , caption , draw_flag ):

        laprf = [ 0 ] * 5
        data.reset( shuffle = False )
        for btc in self.loopEval( data , caption ):

            image , label = data.next_batch()
            output , loss = self.sess.run( [ self.net.output , self.net.loss ] ,
                    { self.net.input : image , self.net.label : label ,
                      self.net.drop : 1.0 , self.net.phase : False } )
            kld.lst.Add( laprf , [ loss ] + kld.stt.aprf( label , output ) )

            if draw_flag:
                for j in range( len( output ) ):
                    btcj = btc * data.batch_size() + j
                    file = '%03d_%04d_out' % ( btcj , epoch )
                    folder = 'individual/%s/%03d' % ( caption , btcj )
                    self.saver.image( self.phase , output[j] , file , folder )
                    if kld.chk.iter_to( epoch , self.args.plot_every , self.args.num_epochs ):
                        file = '%04d_%03d' % ( epoch , btcj )
                        folder = 'sequence/%s/%04d' % ( caption , epoch )
                        plt = self.prepare_plot( image[j] , label[j] , output[j] )
                        self.saver.image( self.phase , plt , file , folder )

        return kld.lst.div( laprf , data.num_batches() )

    ### EVALUATE
    def evaluate( self , data , epoch ):

        laprf = self.evaluate_all( data , epoch )
        self.save_iter( laprf , epoch )

######################################################

    ### TRAIN
    def train( self ):

        epoch = int( self.saver.start_scalar( 'epoch' , 0 ) )
        LRate = kld.mng.LRate( 'linear' , start = 1e-4 , finish = 1e-7 ,
                               num_steps = self.args.num_epochs * self.data_train.num_batches() )

        self.draw_prev( self.data )
        self.evaluate( self.data , epoch )
        for epoch in range( epoch + 1 , self.args.num_epochs + 1 ):
            self.optimize_loop( self.data_train , epoch , LRate.next() )
            if kld.chk.iter_to( epoch , self.args.eval_every , self.args.num_epochs ):
                self.evaluate( self.data , epoch )

    ### TEST
    def test( self ):

        epoch = int( self.saver.start_scalar( 'epoch' , 0 ) )

        self.draw_prev( self.data )
        self.evaluate( self.data , epoch )


######################################################
