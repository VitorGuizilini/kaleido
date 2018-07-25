
import os
import shutil

### RESTART
def restart( name ):
    if os.path.exists( name ):
        shutil.rmtree( name )

