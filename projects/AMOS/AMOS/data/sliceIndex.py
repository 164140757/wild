import numpy as np
from glob import glob
import os 
import json
import nibabel as nib

from multiprocessing import Pool
from tqdm import tqdm

# IMG_BNAME="/mnts2d/med_data1/haotian/AMOS/BTCV/preprocessed/Training/img/*.nii.gz"
SEG_BNAME="/mnts2d/med_data1/haotian/AMOS/BTCV/preprocessed/Training/label/*.nii.gz"
    
segs = glob(SEG_BNAME)

segs.sort()

classmap = {}
LABEL_NAME = ["spleen", "right kidney", "left kidney", "gallbladder", "esophagus",
              "liver", "stomach", "aorta", "inferior vena cava", "portal vein and splenic vein",
              "pancreas", "right adrenal gland", "left adrenal gland"]
MIN_TP = 1 # minimum number of positive label pixels to be recorded

fid = f'/mnts2d/med_data1/haotian/AMOS/BTCV/BTCVRawData/Training/label/classmap_{MIN_TP}.json' # name of the output file. 
for _lb in LABEL_NAME:
    classmap[_lb] = {}
    for _sid in segs:
        pid = os.path.split(_sid)[1].split("label")[-1].split(".nii.gz")[0]
        classmap[_lb][pid] = []

def check_slice(*args):
    slc = args[0][0]
    lb_vol = args[0][1]
    for cls in range(len(LABEL_NAME)):
        if cls in lb_vol.dataobj[..., slc]:
            if np.sum(lb_vol.dataobj[..., slc]) >= MIN_TP:
                classmap[LABEL_NAME[cls]][str(pid)].append(slc) 


for seg in segs:
    pid = os.path.split(seg)[1].split("label")[-1].split(".nii.gz")[0]
    lb_vol = nib.load(seg)
    n_slice = lb_vol.shape[2]
    with Pool(24) as p:
        r = list(tqdm(p.map(check_slice, [(slc, lb_vol) for slc in range(n_slice)]), total=n_slice))
    print(f'pid {str(pid)} finished!')    
    
with open(fid, 'w') as fopen:
    json.dump(classmap, fopen)
    fopen.close()  
    