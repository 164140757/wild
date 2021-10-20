# environment settings
import os
import glob
import pydicom
import pandas as pd




def mergeReportAndSeries(data_root, pre_series_report_dirs, series_meta_df):
    """
    Args:
    - pre_series_report_dirs: previous full report dir list
    - data_root: new DICOM dir to find report csv
    - series_meta_df: new data series meta df 
    """
    csvs = glob.glob(data_root+'/*.csv', recursive=True)
    report_df = pd.concat([pd.read_csv(f) for f in csvs ])
    pre_df = None
    # 提取之前已标注的病人病历号
    if pre_series_report_dirs is not None:
        pre_df = pd.concat([pd.concat(pd.read_excel(dr, usecols=['病历号', 'complete_ab_flag'], sheet_name=None), ignore_index=True) for dr in pre_series_report_dirs])
        pre_df = pre_df[pre_df['complete_ab_flag'] == 1]

    report_df['检查时间'] = pd.to_datetime(report_df['检查时间'], format='%Y-%m-%d %H:%M:%S')
    report_df['检查时间'] = report_df['检查时间'].dt.strftime('%Y%m%d')

    #检查异常值，并丢掉，
    indexes = report_df.loc[report_df['检查时间'].isnull().values].index
    if len(indexes) !=0:
        report_df = report_df.drop(indexes)

    report_df.rename(columns={'检查号':'check_id'}, inplace=True)
    merge_res = series_meta_df.merge(report_df, on='check_id')
    targets = merge_res[~merge_res['病历号'].isin(pre_df['病历号'].values)].copy() if pre_df is not None else merge_res
    targets["nii_file"] = series_meta_df["check_id"].str.cat(series_meta_df["Study Instance UID"].str.cat(series_meta_df["Series Instance UID"],  sep='_'), sep='_')
    targets['complete_ab_flag'] = ''
    targets['ann_flag'] = ''
    
    # targets.dropna(how='all', axis=1, inplace=True)
    return targets
    
if __name__ == '__main__':
    data_root = r'F:\MIA\AMOS-CT-MR\raw\second_round\CT\2021\202102'
    pre_series_report_dirs = glob.glob(r'F:\MIA\AMOS-CT-MR\raw\meta'+'/*/*.xlsx')
    series_meta_df = pd.read_excel('series_tmp.xlsx')
    
    mergeReportAndSeries(data_root, pre_series_report_dirs, series_meta_df).to_excel('full_tmp.xlsx')
    