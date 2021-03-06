import os, pandas as pd

# Use it after deleting files and update complete-auto-flag

DF_PATH = r'D:\Development\OneDrive - i.shu.edu.cn\AMOS\People_hospital\round_4\MR\2021\mr_data_meta_round_4.xlsx'
NII_ROOTS = [r'D:\Development\OneDrive - i.shu.edu.cn\AMOS\People_hospital\round_4\MR\2021\data', ]

df = pd.read_excel(DF_PATH)
df['complete_ab_flag'] = ''
total_nii=[]
# clean
for nii_root in NII_ROOTS:
    for root, dirs, files in os.walk(nii_root):
        file_list=[]
        for file in files:
            file_list.append(os.path.join(root, file))
        total_nii.extend(file_list)    
        
total_nii = [os.path.split(x)[-1].split('.nii.gz')[0] for x in total_nii]
df.loc[df['nii_file'].isin(total_nii), 'complete_ab_flag'] = 1
df.to_excel(os.path.join(DF_PATH), encoding='utf-8', index=False)