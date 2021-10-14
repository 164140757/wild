# environment settings
import os
import glob
import pydicom
import pandas as pd

data_dir = "/mnts2d/med_data1/haotian/AMOS/second_round/CT/2021/202101"
root_dir = "/mnts2d/med_data1/haotian/AMOS/second_round/"
nii_dir = "/mnts2d/med_data1/haotian/AMOS/second_round/ct_nii"

series_meta_dirs = [
    '/mnts2d/med_data1/haotian/AMOS/second_round/ct_series_20210101-20210110.xlsx', 
    '/mnts2d/med_data1/haotian/AMOS/second_round/ct_series_20210111-20210117.xlsx',
    '/mnts2d/med_data1/haotian/AMOS/second_round/ct_series_20210118-20210131.xlsx']
# 与前面轮次id去重
pre_dirs = [
    "/mnts2d/med_data1/haotian/AMOS/first_round/firstround_data_meta-2020.xlsx",
    "/mnts2d/med_data1/haotian/AMOS/first_round/firstround_data_meta-2019.xlsx",
    "/mnts2d/med_data1/haotian/AMOS/first_round/firstround_data_meta-2018.xlsx",
]


def mergeReportAndSeries(data_root, pre_series_report_dirs, series_meta_df):
    """
    Args:
    - pre_series_report_dirs: previous full report dir list
    - data_root: new DICOM dir to find report csv
    - series_meta_df: new data series meta df 
    """
    csvs = glob.glob(data_root+'/*/*.csv', recursive=True)
    report_df = pd.concat([pd.read_csv(f) for f in csvs ])
    # 提取之前已标注的病人病历号
    pre_df = pd.concat([pd.concat(pd.read_excel(dr, usecols=['病历号', 'complete_ab_flag'], sheet_name=None), ignore_index=True) for dr in pre_series_report_dirs])
    pre_df = pre_df[pre_df['complete_ab_flag'] == 1]

    report_df['检查时间'] = pd.to_datetime(report_df['检查时间'], format='%Y-%m-%d %H:%M:%S')
    report_df['检查时间'] = report_df['检查时间'].dt.strftime('%Y%m%d')

    #检查异常值，并丢掉，
    indexes = report_df.loc[report_df['检查时间'].isnull().values]
    if len(indexes) !=0:
        report_df = report_df.drop(indexes)

    # 使用实际文件去与report df 还有 meta df做 匹配
    niigzs = []
    for root, dirs, files in os.walk(nii_dir):
        for file in files:
            base_name = os.path.basename(os.path.join(root, file))
            if not base_name.startswith(".") and base_name.endswith("nii.gz"):
                niigzs.append(os.path.join(root, file))

    over_res = []
    report_df.rename(columns={'检查号':'check_id'}, inplace=True)
    # ['033dffc420e476b47fd248c6d9ccce03', '1.2.840.113704.1.111.5736.1609647701.1', '1.3.46.670589.33.1.20427946432493030468.23285858122440928453', 'Tilt', 'Eq', '1']
    for item in niigzs:
        case_base_name = os.path.basename(item)
        case_base_name = case_base_name.split(".nii.gz")[0]
        ids = case_base_name.split("_")
        if len(ids)!=3:
            continue
        _check_id, _study_id, _series_id = case_base_name.split("_")
        meta_res = series_meta_df.loc[(series_meta_df["check_id"] == _check_id) & (series_meta_df["Study Instance UID"] == _study_id) & (series_meta_df["Series Instance UID"] == _series_id)]
        if len(meta_res) ==1:
            repo_res = report_df.loc[(report_df["check_id"] == _check_id)&(report_df["检查时间"] ==str(meta_res["Study Date"].values[0]))].copy()
            if repo_res['病历号'].values[0] not in pre_df['病历号']:
                repo_res["nii_file"] = case_base_name
                repo_res['complete_ab_flag'] = ''
                repo_res['ann_flag'] = ''
                res = pd.merge(meta_res, repo_res, on=["check_id"])
                over_res.append(res)

    over_res = pd.concat(over_res)
    over_res.dropna(how='all', axis=1, inplace=True)
    return over_res
    # over_res.to_excel(os.path.join(out_path, encoding='utf-8', index=False))