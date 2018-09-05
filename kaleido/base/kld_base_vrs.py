
import tensorflow
import kaleido as kld

### VRSMOD
def modcls( file , saver , args , scope , *pargs , **kwargs ):
    return kld.pth.module( file , saver ).vrs( args , scope , *pargs , **kwargs )
def vrsmod( args , scope , vrs , saver ):
    free = scope[0] != '/'
    if not free: scope = scope[1:]
    file = scope.split('/')[-1]
    split = file.split('_')
    if len( split ) > 1: file = split[-2].lower()
    else: file = split[-1].lower()
    name = kld.pth.callernames(1)[0].split('_')[0]
    file = 'vrs_%s.%s_vrs_%s%s' % ( file , name , file , vrs )

    if scope.islower(): scope = ''

    if not free:
        with tensorflow.variable_scope( scope ) as scope:
            pass

    return kld.init( modcls , file , saver , args , scope )
