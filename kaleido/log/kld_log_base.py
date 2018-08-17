
import os
import shutil

### RESTART
def restart( name ):
    if name is not None:
        if os.path.exists( name ):
            shutil.rmtree( name )

