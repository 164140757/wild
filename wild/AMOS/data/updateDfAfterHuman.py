import os, pandas as pd

# Use it after deleting files and update complete-auto-flag

DF_PATH = r'F:\MIA\AMOS-CT-MR\raw\meta\second_round\secondround_ct_data_meta_202101.xlsx'
NII_ROOTS = [r'F:\MIA\AMOS-CT-MR\processed\second_round\ct_nii\interest\interest_202101', 
             r'F:\MIA\AMOS-CT-MR\processed\second_round\ct_nii\interest_new_diseases\interest_202101',
             r'F:\MIA\AMOS-CT-MR\processed\second_round\ct_nii\normal\interest_202101']

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