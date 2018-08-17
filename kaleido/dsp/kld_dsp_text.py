
from tqdm import tqdm
from kaleido.aux import *

### COUNT
def count( name , num , max , post = '' ):
    mask = '{name}: {num:>{pad}d}{post}'
    return mask.format( name = name , num = num , pad = ndig( max ) , post = post )

### LOOPBAR
def loopBar( data , text , width = 100 , leave = True , enum = False ):
    bar = tqdm( data , '| ' + text + ' |' , ncols = width , leave = leave )
    return enumerate( bar ) if enum else bar

### PAD
def padc( val , pad , prev = '' , post = '' ):
    if pad == 0: return ''
    return '{}{:^{pad}}{}'.format( prev , val , post , pad = pad )
def padl( val , pad , prev = '' , post = '' ):
    if pad == 0: return ''
    return '{}{:<{pad}}{}'.format( prev , val , post , pad = pad )
def padr( val , pad , prev = '' , post = '' ):
    if pad == 0: return ''
    return '{}{:>{pad}}{}'.format( prev , val , post , pad = pad )

### TITLE
def title( desc = None , max = None , prev = '###' , post = '###' ):
    if desc is None: return ''
    pad = len( prev ) + len( post ) + 8
    if max is not None: pad = max - pad
    return '| {}  {:^{pad}}  {} |'.format( prev , desc , post , pad = pad )
def print_title( desc = None , max = None , prev = '###' , post = '###' ):
    if desc is None: return
    print( title( desc , max , prev , post ) )
    print( hline( max ) )

### HEADER
def header( desc , max ):
    return remspace( '|{:^{pad}}|'.format( desc , pad = max - 2 ) , '-' )
def print_header( desc , max ):
    print( hline( max ) )
    print( header( desc , max ) )
    print( hline( max ) )

### MAKE LINES
def make_lines( items , maxs ):
    lines , max = [] , 0
    for i in range( len( items ) ):
        line = hitem( items[i] , maxs )
        if len( line ) > max: max = len( line )
        lines.append( line )
    return lines , max

### HITEM
def hitem( values , maxs ):
    n , line = len( values ) , '|'
    for i in range( n ):
        if maxs[i] == 0: continue
        pad = padl if i == 0 else padc
        line += pad( values[i] , maxs[i] , '  ' , '  :' )
    return line[:-1] + '|'
def print_hitem( values , maxs ):
    print( hitem( values , maxs ) )

### LINE
def hline( max , s = '-' ):
    return '|' + s * ( max - 2 ) + '|'
def print_hline( max , s = '-' ):
    print( hline( max , s ) )

### MESSAGE
def print_message( text , desc , max , headers , maxs , lines ):
    print_header( text , max )
    print_title( desc , max )
    print_hitem( headers , maxs )
    print( hline( max ) )
    for line in lines: print( line )
    print( hline( max ) )


