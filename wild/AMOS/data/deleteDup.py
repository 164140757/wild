DATA_ROOT = r'F:\MIA\AMOS-CT-MR\processed\second_round\ct_nii\normal'
# REF_EXIST_DATA_ROOT = r'F:\MIA\AMOS-CT-MR\processed\first_round\valid'
REF_DF_DIRS = [r'F:\MIA\AMOS-CT-MR\raw\meta\first_round']

import os
from numpy.lib.polynomial import roots
from tqdm import tqdm
import pandas as pd

def walk_nii(root):
    total = []
    for root, dirs, files in os.walk(root):
        file_list=[]
        for file in files:
            file_list.append(root, file)
        total.extend(file_list)        
    pre_total_paths = [x for x in total if x.endswith('.nii.gz')]
    pre_num = len(pre_total_paths)
    total = set([os.path.split(x)[-1] for x in pre_total_paths])
    out = []
    print(f'Delete {len(total)-pre_num} duplicate files in {DATA_ROOT}.')
    for x in pre_total_paths:
        if os.path.split(x)[-1] not in total:
            os.remove(x)
        else:
            out.append(x)
    return out

def get_df(roots):
    total = []
    for root in roots:
        for root, dirs, files in os.walk(root):
            excels = []
            for file in files:
                excels.append(file)
            total.extend(excels)
    total = [x for x in total if x.endswith('.xlsx')]
    return pd.concat([df for df in [pd.read_excel(x) for x in total]])

data = walk_nii(DATA_ROOT)
# ref = walk_nii(REF_EXIST_DATA_ROOT)
ref_df = get_df(REF_DF_DIRS)
ref_df = ref_df[ref_df['ann_flag']==1 or ref_df['ann_flag']==2]
data_toRemove = [x for x in data if os.path.split(x)[-1].replace('.nii.gz', '') in ref_df['nii_file']]
print(f'Deleting duplicated {len(data_toRemove)} cases from data_root {DATA_ROOT}, as there are ann files.')
for x in tqdm(data_toRemove):
    os.remove(x)
# ref remove suffix _pred
# _ref = [os.path.split(x)[-1] for x in ref]
# dup = [x for x in data if os.path.split(x)[-1] in _ref]

# print(f'Deleting duplicated {len(dup)} cases from data_root {DATA_ROOT}, as there are same files from {REF_EXIST_DATA_ROOT}')
# for d in tqdm(dup):
#     os.remove(d)