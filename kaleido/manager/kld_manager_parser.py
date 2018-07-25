
import argparse
import kaleido as kld

##### PARSER
class Parser:

    ### __INIT__
    def __init__( self ):
        self.parser = argparse.ArgumentParser()
        self.info = {}

    ### ARGS
    def args( self ):
        args = self.parser.parse_args()
        if bool( self.info ):
            dict = vars( args )
            for key in self.info:
                if self.info[key] == 'tuple': dict[key] = tuple( dict[key] )
        return args

    ### ADD
    def add( self , name , type , nargs , default , action = None ):
        if kld.chk.is_seq( name ):
            if not kld.chk.is_seq( default ): default = [ default ] * len( name )
            for i in range( len( name ) ): self.add( name[i] , type , nargs , default[i] , action )
        else:
            if type == bool:
                self.parser.add_argument( '--' + name , action = action , default = default )
            else:
                self.parser.add_argument( '--' + name , type = type , nargs = nargs , default = default )

    def add_int(     self , name , default = None ): self.add( name , int   , '?' ,  default )
    def add_float(   self , name , default = None ): self.add( name , float , '?' ,  default )
    def add_string(  self , name , default = None ): self.add( name , str   , '?' ,  default )
    def add_bool(    self , name , default = None ): self.add( name , bool  , '?' ,  default , 'store_false' if default else 'store_true' )
    def add_lint(    self , name , default = None ): self.add( name , int   , '+' ,  default )
    def add_lfloat(  self , name , default = None ): self.add( name , float , '+' ,  default )
    def add_tfloat(  self , name , default = None ): self.add( name , float , '+' ,  default ) ; self.info[name] = 'tuple'
    def add_lstring( self , name , default = None ): self.add( name , str   , '+' ,  default )

