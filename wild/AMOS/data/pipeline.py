import itertools
from multiprocessing.pool import Pool
import pandas as pd 

import os 
from glob import glob
import shutil
from tqdm import tqdm
from functools import partial

from .dcm2nii import hasSubdir, dcm2niix
from .meta2csv import meta2csv
from .meta2report import mergeReportAndSeries


"""
PRE_FULL_REPORT_ROOT: All previous full reports to exclude from by patients' IDs.
DF_PATH: path that has all new patients full reports if it exists.
DATA_ROOT: DICOM ROOT or Nii ROOT.
OUT_DIR_NII: directory to output patients nii files of interest. 
PRE_NII_ROOT: previous nii files root to move to OUT_DIR_NII
"""

PRE_FULL_REPORT_ROOT = r'F:\MIA\AMOS-CT-MR\raw\meta'
DATA_ROOT = r'F:\MIA\AMOS-CT-MR\raw\second_round\CT\2021\202102'
OUT_DIR_NII_interest = r'F:\MIA\AMOS-CT-MR\processed\second_round\ct_nii\interest\interest_202102'
OUT_DIR_NII_tmp = r'F:\MIA\AMOS-CT-MR\processed\second_round\ct_nii\tmp_ct_nii_202102'
DF_PATH = PRE_FULL_REPORT_ROOT+'\second_round\secondround_ct_data_meta_202102.xlsx'
PRE_NII_ROOT = None

def select_nii_paths_with_interest(df_interest, nii_dir):
    """
    Args:
    - df_interest: dataframe after selection
    - nii_dir: nii files root.
    
    Return:
    Paths after interest selection.
    """
    total=[]
    # clean
    for root, dirs, files in os.walk(nii_dir):
        dir_list=[]
        for file in files:
            dir_list.append(os.path.join(root, file))
        total.extend(dir_list)        
    total = [x for x in total if x.endswith('.nii.gz')]
    total = [x for x in total if os.path.split(x)[-1].split('.nii.gz')[0] in df_interest['nii_file'].values]
    
    print(f'Get {len(total)} cases in the end and ready for moving it to {OUT_DIR_NII_interest}.')
    return total

def generate_dcm_dirs(df_interest, total_dcm_dirs=None):
    """
    Get dcm dirs input for dcm2niix 
    
    Args:
    - df_interest: dataframe after interest selection.
    - total_dcm_dirs: all dcm checks directories
    """
    if total_dcm_dirs is None:
        dir_list=[]
        print("Have not found dicom folders in variables. Start collecting dicom folders to convert to nii.")
        for root, dirs, files in os.walk(DATA_ROOT):
            dir_list=[]
            for _dir in dirs:
                dir_list.append(os.path.join(root, _dir))
            dir_list.extend(dir_list)
        print('Checking all dicom files paths.')
        total_dcm_dirs=[x for x in tqdm(dir_list) if not hasSubdir(x)]
        print(f'Found cases {len(total_dcm_dirs)} after checking.')
        
    total_paths = set()
    check_ids = df_interest['nii_file'].str.split('_', expand=True).loc[:, 0].values
    for _dir in total_dcm_dirs:
        if os.path.split(_dir)[-1] in check_ids:
            total_paths.add(_dir)
    return total_paths
        
def move(total_nii_paths):
    """
    - total_nii_paths: all nii_paths to move to OUT_DIR_NII
    """
    os.makedirs(OUT_DIR_NII_interest, exist_ok=True)
    if os.path.exists(os.path.join(DATA_ROOT, 'data.csv')):
        shutil.move(os.path.join(DATA_ROOT, 'data.csv'), os.path.join(OUT_DIR_NII_interest, 'data.csv'))

    for file in tqdm(total_nii_paths):
        _ = file.split(os.sep)
        dir_name = _[-2]
        file_name = _[-1]
        dir_out = os.path.join(OUT_DIR_NII_interest, dir_name)
        os.makedirs(dir_out, exist_ok=True)
        shutil.move(file, os.path.join(dir_out, file_name))    
        

def selectByDf(df=None):
    print('Start selecting patients of interst')
    if df is None:
        df = pd.read_excel(DF_PATH, usecols=['complete_ab_flag', 'nii_file', '临床诊断', 'shape', 'Protocol Name', 'spacing', '检查时间'])
    else:
        df = df[['complete_ab_flag', 'nii_file', '临床诊断', 'shape', 'Protocol Name', 'spacing', '检查时间']]
    
    df = df.loc[df['Protocol Name'].str.contains('Abdomen')]
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
    print(f'Patients in interest from df: {df.shape[0]}')
    return df

def dicom2FullReport(num_pool=8, save=True):
    total = []
    print('Start fetching all dicom files paths.')
    for root, dirs, files in os.walk(DATA_ROOT):
        dir_list=[]
        for _dir in dirs:
            dir_list.append(os.path.join(root, _dir))
        total.extend(dir_list)
        
    print('Checking all dicom files paths.')
    total = [x for x in tqdm(total) if not hasSubdir(x)]
    print(f'Found cases {len(total)} after checking.')
    
    print(f'Start collecting dicom info to the file {DF_PATH}.')
    with Pool(num_pool) as p:
        r = itertools.chain(*tqdm(p.map(meta2csv, total), total=len(total)))    
        
    pre_series_report_dirs = glob(PRE_FULL_REPORT_ROOT+'/*/*.xlsx')
    series_meta_df = pd.DataFrame(r)
    print('Merge reports and series\' meta...')
    full_df = mergeReportAndSeries(DATA_ROOT, pre_series_report_dirs, series_meta_df)
    
    out_dir = os.path.split(DF_PATH)[0]
    os.makedirs(out_dir, exist_ok=True)
    
    if save:
        print(f'Output the full report to the dir {DF_PATH}.')
        full_df.to_excel(os.path.join(DF_PATH, encoding='utf-8', index=False))
    return full_df, total

def dcm2niiFiles(total_dir, num_pool=8):
    """
    Output nii files of interest based on check ids from total_dir
    """
    print(f'Start dcm2niix and out to folder {OUT_DIR_NII_tmp}')
    with Pool(num_pool) as pool:
        tqdm(pool.map(partial(dcm2niix, out_dir=OUT_DIR_NII_tmp), total_dir), total=len(total_dir))

if __name__ == '__main__':
    # Please ensure all excels files including series meta, reports.csv are closed.
    # Or you may get access deny error.
    df = None
    if os.path.exists(DF_PATH):
        df = selectByDf()
    else:
        df, total_dcm_dirs = dicom2FullReport(save=True)
        df = selectByDf(df)

    if PRE_NII_ROOT is not None:
        paths = select_nii_paths_with_interest(df, PRE_NII_ROOT)
        move(paths)
    else:
        total = generate_dcm_dirs(df, total_dcm_dirs)
        dcm2niiFiles(total)
        paths = select_nii_paths_with_interest(df, OUT_DIR_NII_tmp)
        move(paths)
        
    


    