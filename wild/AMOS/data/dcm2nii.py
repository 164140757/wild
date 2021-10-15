import os
from glob import glob
from multiprocessing import Pool

from tqdm import tqdm

data_root = r"F:\MIA\AMOS-CT-MR\raw\test"
out_dir = r"F:\MIA\AMOS-CT-MR\processed\test"

def hasSubdir(root):
    list_dir = [os.path.isdir(os.path.join(root, d)) for d in os.listdir(root)]
    return any(list_dir)

def dcm2niix(_dir, out_dir):
    check_id = os.path.split(_dir)[-1]
    out_path = os.path.join(out_dir, check_id)
    os.makedirs(out_path, exist_ok=True)
    cmd='dcm2niix -f %f_%k_%j -z y -o \"{}\" \"{}\"'.format(out_path, _dir)
    res=os.popen(cmd)
    output_str=res.read()
    return output_str


if __name__ == '__main__':
    data_roots=glob(data_root+'/*/')

    total_dir=[]

    totolen=0
    for data_root in data_roots:
        dir_list=[]
        for root, subdirs, _ in os.walk(data_root):
            for subdir in subdirs:
                dir_list.append(os.path.join(root, subdir))
        totolen += len(dir_list)
        total_dir.extend(dir_list)

    # clean sub-roots
    total_dir = [x for x in total_dir if not hasSubdir(x)]
    print(f'Total number of cases: {len(total_dir)}')

    with Pool(3) as p:
        tqdm(p.map(dcm2niix, total_dir), total=len(total_dir))
