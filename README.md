# kaleido
*by Vitor Campanholo Guizilini*

Python library that provides extra functionality to the Tensorflow library. 

# INSTALLATION

Add the following line to your ~/.bashrc file:

```
export PYTHONPATH=$PYTHONPATH:/path/to/where/this/library/is #(the root folder, NOT the kaleido folder)
```

followed by:

```
$ source ~/.bashrc
```

in a terminal, to refresh your environment variables.

# CONTENTS

- **kaleido:** the actual library
- **projects:** applications that use the library
- **download_datasets:** script to download the datasets for each project (*.tar.gz* files are downloaded to */datasets*, please extract after downloading). Usage:

```
./download_datasets nameofproject
```




