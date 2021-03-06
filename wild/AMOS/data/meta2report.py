# environment settings
import os
import glob
import pydicom
import pandas as pd




def mergeReportAndSeries(data_root, pre_series_report_files, series_meta_df, DF_PATH=''):
    """
    Args:
    - pre_series_report_dirs: previous full report file list
    - data_root: new DICOM dir to find report csv
    - series_meta_df: new data series meta df 
    - DF_PATH: Default: ''
    """
    # save series_meta_df in case of lossing all info
    if  os.path.exists(DF_PATH):
        series_meta_df = pd.read_excel(DF_PATH)
    elif series_meta_df is not None:
        series_meta_df.to_excel(DF_PATH)
        
    total_report = []
    # clean
    for root, dirs, files in os.walk(data_root):
        file_list = []
        for file in files:
            file_list.append(os.path.join(root, file))
        total_report.extend(file_list)
    total_report = [x for x in total_report if x.endswith('.csv')]
    report_df = pd.concat([pd.read_csv(f) for f in total_report ])
    pre_df = None
    
    if pre_series_report_files is not None:
        pre_df = pd.concat([pd.concat(pd.read_excel(dr, usecols=['check_id'], sheet_name=None), ignore_index=True) for dr in pre_series_report_files])

    report_df['检查时间'] = pd.to_datetime(report_df['检查时间'], format='%Y-%m-%d %H:%M:%S')
    report_df['检查时间'] = report_df['检查时间'].dt.strftime('%Y%m%d')

    #检查异常值，并丢掉，
    indexes = report_df.loc[report_df['检查时间'].isnull().values].index
    if len(indexes) !=0:
        report_df = report_df.drop(indexes)

    report_df.rename(columns={'检查号':'check_id'}, inplace=True)
    merge_res = series_meta_df.merge(report_df, on='check_id')
    targets = merge_res[~merge_res['check_id'].isin(pre_df['check_id'].values)].copy() if pre_df is not None else merge_res
    targets["nii_file"] = series_meta_df["check_id"].str.cat(series_meta_df["Study Instance UID"].str.cat(series_meta_df["Series Instance UID"],  sep='_'), sep='_')
    targets['complete_ab_flag'] = ''
    targets['ann_flag'] = ''
    
    # targets.dropna(how='all', axis=1, inplace=True)
    return targets
    
if __name__ == '__main__':
    data_root = r'E:\done'
    pre_series_report_dirs = glob.glob(r'F:\MIA\AMOS-CT-MR\raw\meta'+'/*/*.xlsx')
    series_meta_df = pd.read_excel('series_tmp.xlsx')
    
    mergeReportAndSeries(data_root, pre_series_report_dirs, series_meta_df).to_excel('full_tmp.xlsx')
    