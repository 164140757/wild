import os 
from glob import glob

def hasSubdir(root):
    sub_dirs = set([os.path.dirname(p) for p in glob(root+"/*/*")])
    return len(sub_dirs)>0