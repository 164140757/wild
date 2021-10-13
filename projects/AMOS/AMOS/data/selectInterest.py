from glob import glob
import pandas as pd 
import os 
from wild.util import hasSubdir

df_path = '/mnts2d/med_data1/haotian/AMOS/second_round/CT/2021/202101/secondround_ct_data_meta_202101.xlsx'
DATA_ROOT = r'F:\MIA\AMOS-CT-MR\processed\second_round\ct_nii\ct_nii_20210118-20210131'

df = pd.read_excel(df_path, usecols=['complete_ab_flag', 'nii_file', '临床诊断', 'shape', 'Protocol Name', 'spacing', '检查时间'])

df['检查时间'] = pd.to_datetime(df['检查时间'], format='%Y%m%d')

df = df.loc[(df['检查时间'] >= '2021-01-18') & (df['检查时间'] <= '2021-01-31')] 
df = df[df['complete_ab_flag']!=1]
df = df.loc[(df['临床诊断'].str.contains('癌')) | (df['临床诊断'].str.contains('肿瘤'))]

# spacing * shape -> physical distance
spacing_z = df['spacing'].str.strip('[]').str.split(', ', expand=True).loc[:, 0].astype(float)
shape_z = df['shape'].str.strip('()').str.split(', ', expand=True).loc[:, 0].astype(float)
distance_z = spacing_z.multiply(shape_z, fill_value=0)*0.1
df.insert(0, 'd_z', distance_z)

df = df[df['d_z'] >= 40]
print(f'Patients in interest: {df.shape[0]}')

data_roots=glob(DATA_ROOT+'/*/')
total_dir=[]

totolen=0
for data_root in data_roots:
    dir_list=[]
    for root, subdirs, _ in os.walk(data_root):
        for subdir in subdirs:
            dir_list.append(os.path.join(root, subdir))
    totolen += len(dir_list)
    total_dir.extend(dir_list)

total_dir = [x for x in total_dir if not hasSubdir(x)]

