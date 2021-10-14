import os
import pandas as pd
import shutil

DATA_ROOT = r'F:\MIA\AMOS-CT-MR\processed\second_round\ct_nii\ct_nii_20210111-20210117'
META_PATH = r'D:\Development\focus\wild\data\AMOS\secondround_ct_data_meta_202101.xlsx'
OUT_ROOT = r'F:\MIA\AMOS-CT-MR\processed\second_round\ct_nii\ct_nii_20210111-20210117_'

df = pd.read_excel(META_PATH, usecols=['complete_ab_flag', 'nii_file', '病历号'])
df = df[df['complete_ab_flag']==1]
num_case = len(df)
unique = len(pd.unique(df['病历号'])) == len(df['病历号'])
df = df[[os.path.exists(os.path.join(DATA_ROOT, x.split('_')[0], x)+'.nii.gz') for x in df['nii_file'].tolist()]]
data_list_to_move = [os.path.join(DATA_ROOT, x.split('_')[0], x)+'.nii.gz' for x in df['nii_file'].tolist()]

print(f'Files in total: {num_case}')
print(f'Check Unique patients: {unique}')
if not unique:
    to_print = []
    print('Patients\' ID in common:')
    for x in (set(df['病历号'].tolist())-set(pd.unique(df['病历号'])).tolist()):
        to_print.append(x)
    print(to_print)
    
for dl in data_list_to_move:
    path_to = os.path.join(OUT_ROOT, dl.split(os.sep)[-2])
    os.makedirs(path_to, exist_ok=True)
    shutil.move(dl, os.path.join(path_to, os.path.split(dl)[-1]))

if unique: 
    shutil.rmtree(DATA_ROOT)    
    