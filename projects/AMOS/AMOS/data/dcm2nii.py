import os
from glob import glob
from multiprocessing import Pool

from tqdm import tqdm

data_root = "/mnts2d/med_data1/haotian/AMOS/second_round/CT/2021/202101/20210118-20210131"
out_dir = "/mnts2d/med_data1/haotian/AMOS/second_round/ct_nii"

def dcm2nii(dir):
    idx_s = dir.find('Images-result-deidentify')+len('Images-result-deidentify/')
    idx_end = idx_s+len('0b0548ed1ef4e06e4972fbf9c8946fda')
    out_path = os.path.join(out_dir, 'ct_nii_20210118-20210131', dir[idx_s:idx_end])
    os.makedirs(out_path, exist_ok=True)
    cmd='dcm2niix -f %f_%k_%j -z y -o \"{}\" \"{}\"'.format(out_path,dir)
    res=os.popen(cmd)
    output_str=res.read()
    return output_str

data_roots=glob(data_root+'/*/')
total_dir=[]
dir_list=[]
totolen=0
for data_root in data_roots:
    dir_list=[]
    for root, subdirs, files in os.walk(data_root):
        for subdir in subdirs:
            dir_list.append(os.path.join(root, subdir))
    totolen += len(dir_list)
    total_dir.extend(dir_list)
    print(data_root + "      : " + str(len(dir_list)))

print(f'Toal number of cases: {len(total_dir)}')
with Pool(8) as p:
    r=list(tqdm(p.map(dcm2nii, total_dir), total=len(total_dir)))
