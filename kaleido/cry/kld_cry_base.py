
import random
from Crypto.Cipher import AES, ARC4
from kaleido.chk import *

### ENCRYPT
def encrypt( name ):

    KEY = '140B41B22A29DEB4061BDA6Fb6747E14'
    IV  = '0D79A874BE09C72F'

    is_data = is_str( name ) and name[-3:] == '.pb'

    if is_data:
        fin = open( name , 'rb' )
        data = fin.read()
        fin.close()
    else:
        data = name

    cipher = ARC4.new( KEY )
    data = cipher.encrypt( data )

    if is_data:
        fout = open( name[:-3] + '.cry' , 'wb' )
        fout.write( data )
        fout.close()

    return data

### DECRYPT
def decrypt( name ):

    KEY = '140B41B22A29DEB4061BDA6Fb6747E14'
    IV  = '0D79A874BE09C72F'

    is_data = is_str( name ) and name[-4:] == '.cry'

    if is_data:
        fin = open( name , 'rb' )
        data = fin.read()
        fin.close()
    else:
        data = name

    cipher = ARC4.new( KEY )
    data = cipher.decrypt( data )

    if is_data:
        fout = open( name[:-4] + '.pb' , 'wb' )
        fout.write( data )
        fout.close()

    return data
