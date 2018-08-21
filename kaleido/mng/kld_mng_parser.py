
import sys
from argparse import Namespace
from kaleido.aux import *
from kaleido.chk import *
from kaleido.dsp import *
from kaleido.cvt import *
from kaleido.log import *
from kaleido.lst import *

### CONVERT
def convert( val , nargs , cls ):
    if val is None: return val , False
    if nargs is '+':
        if type( val ) is not list: return val , True
        for i in range( len( val ) ):
            val[i] , flg = convert( val[i] , '?' , cls )
            if flg: return val , True
    else:
        if cls is str:
            return val , not is_str( val )
        elif cls is bool:
            nval = to_bol( val )
            if nval is None: return val , True
            else: return nval , False
        else:
            try: val = float( val )
            except: return val , True
            if cls == int:
                if val.is_integer(): val = int( val )
                else: return val , True
    return val , False

##### PARSER
class Parser:

    ### __INIT__
    def __init__( self , desc = None ):
        self.desc = desc
        self.inputs , self.entries , self.info = {} , [] , {}
        self.add_bol( 'help' , False )
        self.add_str( 'args' , None )

    ### PARSE
    def parse( self ):

        args = sys.argv[1:]
        if len( args ) > 0 and args[0][:2] != '--':
            print( '### KALEIDO ERROR: WRONG PREFIX (not --)' ); exit()
        while len( args ) > 0:
            if len( args[0] ) > 2 and args[0][:2] == '--':
                name , input = args.pop(0)[2:] , []
                while len( args ) > 0 and ( len( args[0] ) < 2 or args[0][:2] != '--' ):
                    input.append( args.pop(0) )
                if name in self.inputs:
                    print( '### KALEIDO ERROR: REPEATED ARGUMENT (%s)' % name ); exit()
                else: self.inputs[name] = input

        load = None
        if 'args' in self.inputs and self.inputs['args'] is not None:
            path = self.inputs['args'][0].split('/')
            if len( path ) == 1: path , name = '.' , path[0]
            else: path , name = join( path[:-1] ) , path[-1]
            loader = Saver( path , free = True )
            load = loader.start_dict( name , self.inputs )

        args = self.inputs.copy()
        for entry in self.entries:
            name = entry['name']
            default = load[name] if load is not None and name in load else entry['default']
            if name not in args: args[name] = default if is_lst( default ) else [ default ]

            if len( args[name] ) == 0:
                if entry['type'] == bool and entry['nargs'] == '?':
                    args[name] = not to_bol( default )
                else: entry['error'] = 'NO ARGUMENTS'
            elif entry['nargs'] == '?':
                if len( args[name] ) == 1:
                    args[name] = args[name][0]
                elif entry['type'] != bool or len( args[name] ) > 1:
                    entry['error'] = 'NUMBER OF ARGUMENTS ({} != 1)'.format( len( args[name] ) )
            else:
                if entry['qty'] is not None and len( args[name] ) != entry['qty']:
                    entry['error'] = 'NUMBER OF ARGUMENTS ({} != {})'.format( len( args[name] ) , entry['qty'] )

        for input in self.inputs:
            for entry in self.entries:
                if input == entry['name']: break
            else: self.entries.append( { 'name' : input , 'error' : 'NOT AN ARGUMENT' ,
                                         'type' : None , 'choices' : None , 'help' : None } )

        return Namespace( **args )

    ### HELP
    def help( self ):

        has_choice = has_help = False
        data , headers = [] , [ 'ARGUMENT' , 'TYPE' , 'DEFAULT' , 'CHOICES' , 'DESCRIPTION' ]
        maxs = [ len( headers[0] ) , len( headers[1] ) , len( headers[2] ) , len( headers[3] ) , len( headers[4] ) ]
        for entry in self.entries:

            if entry['name'] in ['help','args']: continue
            name = '--{}'.format( entry['name'] )
            if entry['required']: name += '*'
            type = typ2str( entry['type'] )
            if entry['nargs'] == '+': type += 's'
            if entry['qty'] is not None: type += ' (%d)' % entry['qty']
            default = 'None' if entry['default'] is None else remspace( entry['default'] )
            choices = '' if entry['choices'] is None else remspace( entry['choices'] )
            help = '' if entry['help'] is None else entry['help']

            if not has_choice: has_choice = entry['choices'] is not None
            if not has_help: has_help = entry['help'] is not None
            if entry['type'] == bool:
                default = default.replace('True','T').replace('False','F')
                choices = choices.replace('True','T').replace('False','F')
            items = [ name , type , default , choices , help ]
            for i in range( len( items ) ):
                if len( items[i] ) > maxs[i]: maxs[i] = len( items[i] )
            data.append( items )

        if not has_help: maxs[4] = 0
        if not has_choice: maxs[3] = 0
        lines , max = make_lines( data , maxs )
        if self.desc is not None:
            lentitle = len( title( self.desc ) )
            if lentitle > max:
                lines , dif , max = [] , lentitle - max , lentitle
                for i in range( len( maxs ) - 1 , 0 , -1 ):
                    if maxs[i] > 0: maxs[i] += dif ; break
                lines , max = make_lines( data , maxs )
        print_message( 'KALEIDO PARSER (HELP)' , self.desc , max , headers , maxs , lines )
        exit()

    ### TESTS
    def tests( self , args ):

        has_help = False
        data , headers = [] , [ 'ARGUMENT' , 'ERROR' , 'DESCRIPTION' ]
        maxs = [ len( headers[0] ) , len( headers[1] ) , len( headers[2] ) ]
        for entry in self.entries:

            val = vars( args )[entry['name']]
            cls , cho , typ = entry['type'] , entry['choices'] , typ2str( entry['type'] )

            is_required = wrong_type = not_choice = False
            not_argument = 'error' in entry
            if not not_argument:
                is_required = entry['required'] and val is None
                if not is_required:
                    val , wrong_type = convert( val , entry['nargs'] , cls )
                    if not wrong_type:
                        not_choice = cho is not None and val not in cho

            error = None
            if not_argument:
                error = entry['error']
            elif is_required:
                if entry['nargs'] == '+': typ += 's'
                error = 'REQUIRED ({}'.format( typ )
                if cho is not None: error += ' = {}'.format( remspace( cho ) )
                error += ')'
            elif wrong_type:
                if vartype( val ) == 'list':
                    for i in range( len( val ) ):
                        if not type(val[i]) == entry['type']:
                            lbl = '[{}] {} {}'.format( i , vartype( val[i] ) , val[i] )
                else:
                    lbl = '{} {}'.format( vartype( val ) , val )
                    if entry['nargs'] == '+': typ += 's'
                error = 'WRONG TYPE ({} != {})'.format( lbl , typ )
            elif not_choice:
                if entry['nargs'] == '+': typ += 's'
                error = 'NOT A CHOICE ({}'.format( typ )
                error += ' {} != {})'.format( remspace( val ) , remspace( cho ) )
            else:
                vars( args )[entry['name']] = val

            if error is not None:
                name = '--{}'.format( entry['name'] )
                help = '' if entry['help'] is None else entry['help']
                if not has_help: has_help = entry['help'] is not None
                if cls == bool: error = error.replace('True','T').replace('False','F')
                items = [ name , error , help ]
                for i in range( len( items ) ):
                    if len( items[i] ) > maxs[i]: maxs[i] = len( items[i] )
                data.append( items )

        if len( data ) > 0:
            if not has_help: maxs[2] = 0
            lines , max = make_lines( data , maxs )
            if self.desc is not None:
                lentitle = len( title( self.desc ) )
                if lentitle > max:
                    lines , dif , max = [] , lentitle - max , lentitle
                    for i in range( len( maxs ) - 1 , 0 , -1 ):
                        if maxs[i] > 0: maxs[i] += dif ; break
                    lines , max = make_lines( data , maxs )
            print_message( 'KALEIDO PARSER (ERROR)' , self.desc , max , headers , maxs , lines )
            exit()

    ### ARGS
    def args( self ):
        args = self.parse()
        if args.help: self.help()
        self.tests( args )
        if bool( self.info ):
            dict = vars( args )
            for key in self.info:
                if self.info[key] == 'tuple': dict[key] = tuple( dict[key] )
        return args

    ### ADD
    def add( self , n , t , a , d , c , q , h , r = False ):
        if is_tup( n ):
            if not is_tup( d ): d = [ d ] * len( n )
            if not is_tup( c ): c = [ c ] * len( n )
            if not is_tup( h ): h = [ h ] * len( n )
            if not is_tup( r ): r = [ r ] * len( n )
            if not is_tup( q ): q = [ q ] * len( n )
            for i in range( len( n ) ): self.add( n[i] , t , a , d[i] , c[i] , q[i] , h[i] , r[i] )
        else:
            self.entries.append( { 'name' : n , 'type' : t , 'nargs' : a , 'qty' : q ,
                                   'default' : d , 'choices' : c , 'help' : h , 'required' : r } )

    def add_int(   self , name , d = None , c = None , h = None ): self.add( name , int   , '?' , d , c , None , h , False )
    def add_flt(   self , name , d = None , c = None , h = None ): self.add( name , float , '?' , d , c , None , h , False )
    def add_str(   self , name , d = None , c = None , h = None ): self.add( name , str   , '?' , d , c , None , h , False )
    def add_bol(   self , name , d = None , c = None , h = None ): self.add( name , bool  , '?' , d , c , None , h , False )
    def add_rint(  self , name , d = None , c = None , h = None ): self.add( name , int   , '?' , d , c , None , h , True  )
    def add_rflt(  self , name , d = None , c = None , h = None ): self.add( name , float , '?' , d , c , None , h , True  )
    def add_rstr(  self , name , d = None , c = None , h = None ): self.add( name , str   , '?' , d , c , None , h , True  )
    def add_rbol(  self , name , d = None , c = None , h = None ): self.add( name , bool  , '?' , d , c , None , h , True  )

    def add_lint(  self , name , d = None , c = None , h = None , q = None ): self.add( name , int   , '+' , d , c , q , h , False )
    def add_lflt(  self , name , d = None , c = None , h = None , q = None ): self.add( name , float , '+' , d , c , q , h , False )
    def add_lstr(  self , name , d = None , c = None , h = None , q = None ): self.add( name , str   , '+' , d , c , q , h , False )
    def add_lbol(  self , name , d = None , c = None , h = None , q = None ): self.add( name , bool  , '+' , d , c , q , h , False )
    def add_rlint( self , name , d = None , c = None , h = None , q = None ): self.add( name , int   , '+' , d , c , q , h , True  )
    def add_rlflt( self , name , d = None , c = None , h = None , q = None ): self.add( name , float , '+' , d , c , q , h , True  )
    def add_rlstr( self , name , d = None , c = None , h = None , q = None ): self.add( name , str   , '+' , d , c , q , h , True  )
    def add_rlbol( self , name , d = None , c = None , h = None , q = None ): self.add( name , bool  , '+' , d , c , q , h , True  )

    def add_tint(  self , name , d = None , c = None , h = None , q = None ): self.add( name , int   , '+' , d , c , q , h , False ) ; self.info[name] = 'tuple'
    def add_tflt(  self , name , d = None , c = None , h = None , q = None ): self.add( name , float , '+' , d , c , q , h , False ) ; self.info[name] = 'tuple'
    def add_tstr(  self , name , d = None , c = None , h = None , q = None ): self.add( name , str   , '+' , d , c , q , h , False ) ; self.info[name] = 'tuple'
    def add_tbol(  self , name , d = None , c = None , h = None , q = None ): self.add( name , bool  , '+' , d , c , q , h , False ) ; self.info[name] = 'tuple'
    def add_rtint( self , name , d = None , c = None , h = None , q = None ): self.add( name , int   , '+' , d , c , q , h , True  ) ; self.info[name] = 'tuple'
    def add_rtflt( self , name , d = None , c = None , h = None , q = None ): self.add( name , float , '+' , d , c , q , h , True  ) ; self.info[name] = 'tuple'
    def add_rtstr( self , name , d = None , c = None , h = None , q = None ): self.add( name , str   , '+' , d , c , q , h , True  ) ; self.info[name] = 'tuple'
    def add_rtbol( self , name , d = None , c = None , h = None , q = None ): self.add( name , bool  , '+' , d , c , q , h , True  ) ; self.info[name] = 'tuple'

    def add_lst( self , name , t , d = None , c = None , h = None ): self.add( name, t , '+' , d , c , len(t) , h , False )

    def add_vrs_load_save_train_restart( self ):

        self.add_rstr( 'vrs'     , d = None  , h = 'Version to be used'          )
        self.add_str(  'load'    , d = None  , h = 'Path to logs folder to load' )
        self.add_str(  'save'    , d = None  , h = 'Path to logs folder to save' )
        self.add_bol(  'train'   , d = False , h = 'Flag for Training/Testing'   )
        self.add_bol(  'restart' , d = False , h = 'Flag for save model restart' )

    def add_num_epochs_eval_every( self ):

        self.add_int( 'num_epochs' , d = 100 , h = 'Number of training epochs'              )
        self.add_int( 'eval_every' , d =   5 , h = 'Interval between epochs for evaluation' )

    def add_num_epochs_eval_plot_every( self ):

        self.add_num_epochs_eval_every()
        self.add_int( 'plot_every' , d =  -1 , h = 'Interval between evaluations for plotting' )

