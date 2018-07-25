
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import kaleido as kld
import numpy as np

### START
def start( num_fig = 1, width = 20.0 , height = 10.0 ):
    for i in range( num_fig ):
        plt.figure( num = i + 1 , figsize = ( width , height ) )
        plt.subplots_adjust( left   = 0.01 , bottom = 0.01 , right = 0.99 , top = 0.99 ,
                             wspace = 0.02 , hspace = 0.02 )

### NO TICKS
def no_ticks():
    plt.gca().xaxis.set_ticklabels( [] )
    plt.gca().yaxis.set_ticklabels( [] )

### XLIM
def xlim( min , max = None ):
    if max is None:
        if kld.chk.is_tuple( min ):
            mmin = np.min( [ np.min( m ) for m in min ] )
            mmax = np.max( [ np.max( m ) for m in min ] )
        else: mmin , mmax = np.min( min ) , np.max( min )
    plt.xlim( mmin , mmax )

### YLIM
def ylim( min , max = None ):
    if max is None:
        if kld.chk.is_tuple( min ):
            mmin = np.min( [ np.min( m ) for m in min ] )
            mmax = np.max( [ np.max( m ) for m in min ] )
        else: mmin , mmax = np.min( min ) , np.max( min )
    mdif = ( mmax - mmin ) / 10.0
    plt.ylim( mmin - mdif , mmax + mdif )

### GRID
def grid( flag = True ):
    plt.grid( 'on' if flag else 'off' )

### SETTINGS
def settings( grid = False , ticks = True , xlim = None , ylim = None ):
    if xlim is not None:
        if kld.chk.is_list( xlim ): kld.plot.xlim( xlim[0] , xlim[1] )
        else: kld.plot.xlim( xlim )
    if ylim is not None:
        if kld.chk.is_list( ylim ): kld.plot.xlim( ylim[0] , ylim[1] )
        else: kld.plot.ylim( ylim )
    if not ticks: kld.plot.no_ticks()
    kld.plot.grid( grid )

### BATCH IMAGES
def batch_images( data , d , rc , order ):
    plt.clf()
    for i , j in enumerate( order ):
        if j is not None:
            plt.subplot(rc[0],rc[1],i+1)
            if j == 0:
                img = data[0][d].copy() * 0.8
                add = data[1][d].copy()
                add[ np.where( add > 0 ) ] = 1.0
                img[:,:,0] = add
            else:
                img = data[j][d]
            plt.imshow( img , vmin = 0.0 , vmax = 1.0 )
            plt.axis('off')
    return plt

#### BATCH IMAGES
#def batch_images( data , d , rc , order ):
#    plt.clf()
#    for i , j in enumerate( order ):
#        if j is not None:
#            plt.subplot(rc[0],rc[1],i+1)
#            plt.imshow( data[j][d] , vmin = 0.0 , vmax = 1.0 )
#            plt.axis('off')
#    return plt

### SEQUENCE
def sequence( g , x , y , p , nr ):

    nc = int( p.shape[0] / nr )
    x = np.reshape( x , [ nr , nc , -1 , 1 ] )
    p = np.reshape( p , [ nr , nc , -1 , 1 ] )
    y = np.reshape( y , [ nr , nc , -1 , 1 ] )
    bck , fwd = x.shape[2] , y.shape[2]

    if p.shape[2] == y.shape[2]:

        plt.figure(1); plt.clf()
        for r in range( nr ):
            for c in range( nc ):
                plt.subplot(nr,nc,r*nc+c+1)
                plt.plot( g[c    :c+bck    ] ,   x[r,c, :,0]   , 'g.-' , linewidth = 2 )
                plt.plot( g[c+bck:c+bck+fwd] ,   y[r,c, :,0]   , 'r.-' , linewidth = 2 )
                plt.plot( g[bck+c-1:bck+c+1] , [ x[r,c,-1,0]   ,
                                                 y[r,c, 0,0] ] , 'r.-' , linewidth = 2 )
                plt.plot( g[c+bck:c+bck+fwd] ,   p[r,c, :,0]   , 'k-'  , linewidth = 2 )
                plt.plot( g[bck+c-1:bck+c+1] , [ x[r,c,-1,0]   ,
                                                 p[r,c, 0,0] ] , 'k-'  , linewidth = 2 )
                kld.plot.settings( xlim = g , ylim = ( x[r] , y[r] ) ,
                                   grid = True , ticks = False )

        plt.figure(2); plt.clf()
        ns , nc = y.shape[1] , y.shape[2]
        for r in range( nr ):
            for c in range( nc ):
                plt.subplot(nr,nc,r*nc+c+1)
                plt.plot( g[ns-1 :bck+ns-1] , x[r,-1,:,0] , 'b.-' , linewidth = 2 )
                plt.plot( g[     :bck     ] , x[r, 0,:,0] , 'g.-' , linewidth = 2 )
                plt.plot( g[bck+c:bck+ns+c] , y[r, :,c,0] , 'r.-' , linewidth = 2 )
                plt.plot( g[bck+c:bck+ns+c] , p[r, :,c,0] , 'k-'  , linewidth = 2 )
                kld.plot.settings( xlim = g , ylim = ( x[r] , y[r] ) ,
                                   grid = True , ticks = False )

    else:

        plt.figure(1); plt.clf()
        for r in range( nr ):
            for c in range( nc ):
                plt.subplot(nr,nc,r*nc+c+1)
                plt.plot( g[c    :c+bck    ] ,   x[r,c, :,0]   , 'g.-' , linewidth = 2 )
                plt.plot( g[c+bck:c+bck+fwd] ,   y[r,c, :,0]   , 'r.-' , linewidth = 2 )
                plt.plot( g[bck+c-1:bck+c+1] , [ x[r,c,-1,0]   ,
                                                 y[r,c, 0,0] ] , 'r.-' , linewidth = 2 )
                plt.plot( g[c+1  :c+bck+fwd] ,   p[r,c, :,0]   , 'k-'  , linewidth = 2 )
                kld.plot.settings( xlim = g , ylim = ( x[r] , y[r] ) ,
                                   grid = True , ticks = False )

        plt.figure(2); plt.clf()
        ns , nc = y.shape[1] , y.shape[2]
        for r in range( nr ):
            for c in range( nc ):
                plt.subplot(nr,nc,r*nc+c+1)
                plt.plot( g[ns-1 :bck+ns-1] , x[r,-1,:      ,0] , 'b.-' , linewidth = 2 )
                plt.plot( g[     :bck     ] , x[r, 0,:      ,0] , 'g.-' , linewidth = 2 )
                plt.plot( g[bck+c:bck+ns+c] , y[r, :,c      ,0] , 'r.-' , linewidth = 2 )
                plt.plot( g[bck+c:bck+ns+c] , p[r, :,c+bck-1,0] , 'k-'  , linewidth = 2 )
                kld.plot.settings( xlim = g , ylim = ( x[r] , y[r] ) ,
                                   grid = True , ticks = False )

    return plt
