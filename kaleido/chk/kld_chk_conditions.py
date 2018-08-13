
### ITER TO
def iter_to( i , n , tot = None ):
    if n == 0 or n is None: return False
    if i == 0: return True
    do = i % n == 0
    if not do and tot is not None:
        do = i >= tot
    return do

def for_hasbigger( xx , v = 0 ):
    for x in xx:
        if len( x ) > v: return True
    return False
