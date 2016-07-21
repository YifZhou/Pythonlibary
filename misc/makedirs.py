import os
from shutil import rmtree

def makedirs(dir, clear=False):
    """guarantee the aim dir exisits, if clear is True and the DIR exists,
    clear the aim DIR
    """
    if os.path.exists(dir):
        if clear:
            rmtree(dir)
        os.makedirs(dir)
    else:
        os.makedirs(dir)
