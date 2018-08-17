
import numpy as np
import sklearn.metrics
import kaleido as kld

### APRF
def aprf( labels , outputs , thr = 0.5 ):
    if kld.chk.is_seq( labels ):
        n = len( labels )
        acc = prec = recl = fmeas = 0.0
        for i in range( n ):
            acci , preci , recli , fmeasi = aprf( labels[i] , outputs[i] )
            acc += acci ; prec += preci ; recl += recli ; fmeas += fmeasi
        return [ acc / n , prec / n , recl / n , fmeas / n ]
    else:

        labels  = kld.aux.round( labels , thr )
        outputs = kld.aux.round( outputs , thr )

        TP = np.count_nonzero( ( outputs     ) * ( labels     ) )
        TN = np.count_nonzero( ( outputs - 1 ) * ( labels - 1 ) )
        FP = np.count_nonzero( ( outputs     ) * ( labels - 1 ) )
        FN = np.count_nonzero( ( outputs - 1 ) * ( labels     ) )

        prec  = TP / ( TP + FP + 1e-6 )
        recl  = TP / ( TP + FN + 1e-6 )
        fmeas = 2 * ( prec * recl / ( prec + recl + 1e-6 ) )
        acc   = ( TP + TN ) / ( TP + TN + FP + FN )

        return [ acc , prec , recl , fmeas ]
