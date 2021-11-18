import itertools
from multiprocessing.pool import Pool
import pandas as pd

import os
from glob import glob
import shutil
from tqdm import tqdm
from functools import partial
import numpy as np

from dcm2nii import hasSubdir, dcm2niix
from meta2csv import meta2csv
from meta2report import mergeReportAndSeries


"""
PRE_FULL_REPORT_ROOT: All previous full reports to exclude from by patients' IDs.
DF_PATH: path that has all new patients full reports if it exists.
DATA_ROOT: DICOM ROOT or Nii ROOT.
OUT_DIR_NII: directory to output patients nii files of interest. 
PRE_NII_ROOT: previous nii files root to move to OUT_DIR_NII
KEYWORDS: Used to select by descriptions of patients
TYPE: Modality type (MR, CT)
SELECT: whether to select by interest
"""

PRE_FULL_REPORT_ROOT = r'D:\Development\OneDrive - i.shu.edu.cn\AMOS\People_hospital\meta'
# PRE_FULL_REPORT_ROOT = None
DATA_ROOT = r'E:\done\CT'
# DATA_ROOT = None
OUT_DIR_NII_interest = r'D:\Development\OneDrive - i.shu.edu.cn\AMOS\People_hospital\round_4\CT\data'
OUT_DIR_NII_tmp = r'D:\Development\OneDrive - i.shu.edu.cn\AMOS\People_hospital\round_4\CT\tmp_mr'
DF_PATH = r'D:\Development\OneDrive - i.shu.edu.cn\AMOS\People_hospital\round_4\CT\ct_data_meta_round_4.xlsx'
# PRE_NII_ROOT = r'D:\Development\OneDrive - i.shu.edu.cn\AMOS\People_hospital\round_4\MR\data'
TYPE = 'CT'
PRE_NII_ROOT=None

SELECT = True
# PRE_NII_ROOT = r'F:\MIA\AMOS-CT-MR\processed\second_round\ct_nii\ct_nii_raw_20210101_20210117'

# KEYWORDS = ['结石', '胆囊炎', '车祸伤', '胰腺炎', '切除', '骨折', '溃疡', '肾脏病', '腹水', '糜烂']
# KEYWORDS = ['瘤', '癌']
KEYWORDS = None
# KEYWORDS = ['痛', '体检', '发热', '贫血']
# KEYWORDS = ['晕', '呕吐', '感染', '糖尿病', '异常', '梗阻', '中毒', '白细胞', '高血压', '功能']


def select_nii_paths_with_interest(df_interest, nii_dir):
    """
    Args:
    - df_interest: dataframe after selection
    - nii_dir: nii files root.

    Return:
    Paths after interest selection.
    """
    total = []
    # clean
    for root, dirs, files in os.walk(nii_dir):
        file_list = []
        for file in files:
            file_list.append(os.path.join(root, file))
        total.extend(file_list)
    total = [x for x in total if x.endswith('.nii.gz')]
    total = [x for x in total if os.path.split(
        x)[-1].split('.nii.gz')[0] in df_interest['nii_file'].values]

    print(
        f'Get {len(total)} cases in the end and ready for moving it to {OUT_DIR_NII_interest}.')
    return total


def generate_dcm_dirs(df_interest, total_dcm_dirs=None):
    """
    Get dcm dirs input for dcm2niix 

    Args:
    - df_interest: dataframe after interest selection.
    - total_dcm_dirs: all dcm checks directories
    """
    if total_dcm_dirs is None:
        total = []
        print("Have not found dicom folders in variables. Start collecting dicom folders to convert to nii.")
        for root, dirs, files in os.walk(DATA_ROOT):
            dir_list = []
            for _dir in dirs:
                dir_list.append(os.path.join(root, _dir))
            total.extend(dir_list)
        print('Checking all dicom files paths.')
        total_dcm_dirs = [x for x in tqdm(total) if not hasSubdir(x)]
        print(f'Found cases {len(total_dcm_dirs)} after checking.')

    total_paths = set()
    check_ids = df_interest['nii_file'].str.split(
        '_', expand=True).loc[:, 0].values
    
    for _dir in total_dcm_dirs:
        if os.path.split(_dir)[-1] in check_ids:
            total_paths.add(_dir)
    return list(total_paths)


def move(total_nii_paths):
    """
    - total_nii_paths: all nii_paths to move to OUT_DIR_NII
    """
    os.makedirs(OUT_DIR_NII_interest, exist_ok=True)

    if PRE_NII_ROOT is not None:
        print(f'Copying nii_temp files in {PRE_NII_ROOT}')

    for file in tqdm(total_nii_paths):
        _ = file.split(os.sep)
        dir_name = _[-2]
        file_name = _[-1]
        dir_out = os.path.join(OUT_DIR_NII_interest, dir_name)
        os.makedirs(dir_out, exist_ok=True)
        if PRE_NII_ROOT is not None:
            shutil.copy(file, os.path.join(dir_out, file_name))
        else:
            shutil.move(file, os.path.join(dir_out, file_name))

    if os.path.exists(OUT_DIR_NII_tmp):
        print(f'Remove nii_temp files in {OUT_DIR_NII_tmp}')
        shutil.rmtree(OUT_DIR_NII_tmp)


def getDf(df=None):
    
    df_pre = None

    if df is None:
        df = pd.read_excel(DF_PATH)
        if not SELECT:
            print(f'Patients in interest from df: {df.shape[0]}')
            return df
        df_pre = df.copy(deep=True)
        df = df[['complete_ab_flag', 'nii_file', '结果描述',
                 'shape', 'Protocol Name', 'spacing', '检查时间']]
    else:
        if not SELECT:
            print(f'Patients in interest from df: {df.shape[0]}')
            return df
        df_pre = df.copy(deep=True)
        df = df[['complete_ab_flag', 'nii_file', '结果描述',
                 'shape', 'Protocol Name', 'spacing', '检查时间']]
        
        
    print('Start selecting patients of interst')
    # df = df.loc[df['Protocol Name'].str.contains('Abdomen', case=False, na=False)] if TYPE == 'CT' else df
    df['检查时间'] = pd.to_datetime(df['检查时间'], format='%Y%m%d')
    # df = df.loc[(df['检查时间'] >= '2021-01-18') & (df['检查时间'] <= '2021-01-31')]
    conditions = []
    conditions = [df['结果描述'].str.contains(
        key, na=False) for key in KEYWORDS] if KEYWORDS is not None else conditions
    # conditions.append(df['complete_ab_flag']!=1) 
    if len(conditions)!=0:
        df = df.loc[np.logical_or.reduce(conditions)]
    
    if df.shape[0] == 0:
        raise ValueError('The full report has no patients of interest.')

    # spacing * shape -> physical distance
    spacing_z = df['spacing'].astype(str).str.strip(
        '[]').str.split(', ', expand=True).loc[:, 0].astype(float)
    shape_z = df['shape'].astype(str).str.strip('()').str.split(
        ', ', expand=True).loc[:, 0].astype(float)
    distance_z = spacing_z.multiply(shape_z, fill_value=0)*0.1
    df.insert(0, 'd_z', distance_z)

    df = df[df['d_z'] >= 40] if TYPE=='CT' else df[df['d_z'] >= 70]
    print(f'Patients in interest from df: {df.shape[0]}')
    # annotate complete_ab_flag, but need check again
    if df.shape[0]!=df_pre[df_pre['complete_ab_flag']==1].shape[0]:
        df_pre['complete_ab_flag'] = ''
        df_pre.loc[df.index, 'complete_ab_flag'] = 1
        print(
            f'Add excel {DF_PATH} with complete_ab_flag that denotes targets of interest.')
        df_pre.to_excel(os.path.join(DF_PATH), encoding='utf-8', index=False)
    return df


def dicom2FullReport(num_pool=8, save=True):
    pre_series_report_dirs = []
    print('Start fetching all dicom files paths.')
    for root, dirs, files in os.walk(DATA_ROOT):
        dir_list = []
        for _dir in dirs:
            dir_list.append(os.path.join(root, _dir))
        pre_series_report_dirs.extend(dir_list)

    print('Checking all dicom files paths.')
    pre_series_report_dirs = [x for x in tqdm(pre_series_report_dirs) if not hasSubdir(x)]
    print(f'Found cases {len(pre_series_report_dirs)} after checking.')

    print(f'Start collecting dicom info to the file {DF_PATH}.')
    with Pool(num_pool) as p:
        r = itertools.chain(*tqdm(p.map(meta2csv, pre_series_report_dirs), total=len(pre_series_report_dirs)))

    if PRE_FULL_REPORT_ROOT is not None:
        pre_series_report_files = []
        # clean
        for root, dirs, files in os.walk(PRE_FULL_REPORT_ROOT):
            file_list = []
            for file in files:
                file_list.append(os.path.join(root, file))
            pre_series_report_files.extend(file_list)
        pre_series_report_files = [x for x in pre_series_report_files if x.endswith('.xlsx')]
    else:
        pre_series_report_files = None
    series_meta_df = pd.DataFrame(r)
    # series_meta_df = pd.read_excel('series_tmp.xlsx')
    print('Merge reports and series\'s meta...')
    full_df = mergeReportAndSeries(
        DATA_ROOT, pre_series_report_files, series_meta_df, DF_PATH)

    out_dir = os.path.split(DF_PATH)[0]
    os.makedirs(out_dir, exist_ok=True)

    if save:
        print(f'Output the full report to the dir {DF_PATH}.')
        full_df.to_excel(os.path.join(DF_PATH), encoding='utf-8', index=False)
    return full_df, pre_series_report_dirs


def dcm2niiFiles(total_dir, num_pool=8):
    """
    Output nii files of interest based on check ids from total_dir(DICOM directories)
    """
    print(f'Start dcm2niix and out to folder {OUT_DIR_NII_tmp}')
    total = len(total_dir)
    # spare resources for outcomes that exist
    total_dir = [x for x in total_dir if not os.path.exists(
        os.path.join(OUT_DIR_NII_interest, os.path.split(x)[-1]))]
    _total = len(total_dir)
    print(f'Skip {total - _total} cases that already exist.')

    with Pool(num_pool) as pool:
        tqdm(pool.map(partial(dcm2niix, out_dir=OUT_DIR_NII_tmp),
             total_dir), total=len(total_dir))


if __name__ == '__main__':
    # Please ensure all excels files including series meta, reports.csv are closed.
    # Or you may get access deny error.
    df = None
    total_dcm_dirs = None

    if os.path.exists(DF_PATH):
        df = getDf()
    else:
        df, total_dcm_dirs = dicom2FullReport(save=True)
        df = getDf(df)

    if PRE_NII_ROOT is not None:
        paths = select_nii_paths_with_interest(df, PRE_NII_ROOT)
        move(paths)
    else:
        total = generate_dcm_dirs(df, total_dcm_dirs)
        dcm2niiFiles(total)
        paths = select_nii_paths_with_interest(df, OUT_DIR_NII_tmp)
        move(paths)
