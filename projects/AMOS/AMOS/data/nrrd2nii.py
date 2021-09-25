




#!/usr/bin/env python
from functools import partial
import re
from typing import OrderedDict
import nrrd
import numpy as np
import argparse
import os
import nibabel as nib
import nrrd
import numpy as np
import copy



def alterLabelValueBySegName(inNrrd):
    dict_label_map = {}
    img = inNrrd[0]
    headers = inNrrd[1]
    
    headers_out = copy.deepcopy(headers)
    img_out = np.zeros(img.shape)

    for header in headers:
        if re.match('Segment[0-9]+_Name$',header) is not None:
            idx = header.split('_')[0].split('Segment')[-1]
            # print(f'index: {idx}')
            gt_labelVal = int(headers[f'Segment{idx}_Name'].split('_')[-1])
            pre = int(headers[f'Segment{idx}_LabelValue'])
            # print(f'pre: {pre}')
            # print(f'to: {gt_labelVal}')
            dict_label_map[int(headers[f'Segment{idx}_LabelValue'])] = gt_labelVal
            # print(f'map: {dict_label_map}')
            headers_out[f'Segment{idx}_LabelValue'] = gt_labelVal
            headers_out[f'Segment{idx}_Name'] = f'Segment_{gt_labelVal}'
            headers_out[f'Segment{idx}_ID'] = f'Segment_{gt_labelVal}'

    # print(dict_label_map)
    for pre, to in dict_label_map.items():
        img_out[img==pre] = to

    return img_out, headers_out

def paddingToOrigin(inNrrd):
    img = inNrrd[0]

    
    return img

def _space2ras(space):
    '''Find the diagonal transform required to transform space to RAS'''

    positive= space.split('-')

    xfrm=[ ]
    if positive[0][0].lower() == 'l': # 'left'
        xfrm.append(-1)
    else:
        xfrm.append(1)

    if positive[1][0].lower() == 'p': # 'posterior'
        xfrm.append(-1)
    else:
        xfrm.append(1)

    if positive[2][0].lower() == 'i': # 'inferior'
        xfrm.append(-1)
    else:
        xfrm.append(1)

    # return 4x4 diagonal matrix
    xfrm.append(1)
    return np.diag(xfrm)


def nifti_write(inImg, prefix= None):

    if prefix:
        prefix= os.path.abspath(prefix)
    else:
        prefix= os.path.abspath(inImg).split('.')[0]

    try:
        img= nrrd.read(inImg)
        # hdr= img[1]
        # data= img[0]

        # change headers context by ID
        data, hdr=alterLabelValueBySegName(img)

        SPACE_UNITS = 2
        TIME_UNITS = 0

        SPACE2RAS = _space2ras(hdr['space'])

        translation= hdr['space origin']
        
        if hdr['dimension']==4:
            axis_elements= hdr['kinds']
            for i in range(4):
                if axis_elements[i] == 'list' or axis_elements[i] == 'vector':
                    grad_axis= i
                    break
            
            volume_axes= [0,1,2,3]
            volume_axes.remove(grad_axis)
            rotation= hdr['space directions'][volume_axes,:3]
            
            xfrm_nhdr= np.matrix(np.vstack((np.hstack((rotation.T, np.reshape(translation,(3,1)))),[0,0,0,1])))

            # put the gradients along last axis
            if grad_axis!=3:
                data= np.moveaxis(data, grad_axis, 3)
            
            try:
                # DWMRI
                # write .bval and .bvec
                f_val= open(prefix+'.bval', 'w')
                f_vec= open(prefix+'.bvec', 'w')
                b_max = float(hdr['DWMRI_b-value'])

                mf= np.matrix(np.vstack((np.hstack((hdr['measurement frame'],
                                                    [[0],[0],[0]])),[0,0,0,1])))
                for ind in range(hdr['sizes'][grad_axis]):
                    bvec = [float(num) for num in hdr[f'DWMRI_gradient_{ind:04}'].split()]
                    L_2= np.linalg.norm(bvec[:3])
                    bval= round(L_2 ** 2 * b_max)

                    bvec.append(1)
                    # bvecINijk= RAS2IJK @ SPACE2RAS @ mf @ np.matrix(bvec).T
                    # simplified below
                    bvecINijk= xfrm_nhdr.T @ mf @ np.matrix(bvec).T

                    L_2= np.linalg.norm(bvecINijk[:3])
                    if L_2:
                        bvec_norm= bvecINijk[:3]/L_2
                    else:
                        bvec_norm= [0, 0, 0]

                    f_val.write(str(bval)+' ')
                    f_vec.write(('  ').join(str(x) for x in np.array(bvec_norm).flatten())+'\n')

                f_val.close()
                f_vec.close()
            
            except:
                # fMRI
                pass
            
            TIME_UNITS= 8
        
        else:
            rotation= hdr['space directions']
            xfrm_nhdr= np.matrix(np.vstack((np.hstack((rotation.T, np.reshape(translation,(3,1)))),[0,0,0,1])))


        xfrm_nifti= SPACE2RAS @ xfrm_nhdr
        # RAS2IJK= xfrm_nifti.I


        # automatically sets dim, data_type, pixdim, affine
        img_nifti= nib.nifti1.Nifti1Image(data, affine= xfrm_nifti)
        hdr_nifti= img_nifti.header

        # now set xyzt_units, sform_code= qform_code= 2 (aligned)
        # https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/xyzt_units.html
        # simplification assuming 'mm' and 'sec'
        hdr_nifti.set_xyzt_units(xyz= SPACE_UNITS, t= TIME_UNITS)
        hdr_nifti['qform_code'] = 2
        hdr_nifti['sform_code']= 2
        
        # append seg info 

        nib.save(img_nifti, prefix+'.nii.gz')
    except:
        print(f'{inImg} is error. ')
   


def main():
    parser = argparse.ArgumentParser(description='NRRD to NIFTI conversion tool')
    parser.add_argument('-i', '--input', type=str, required=True, help='input nrrd/nhdr file')
    parser.add_argument('-p', '--prefix', type=str,
                        help='output prefix for .nii.gz, .bval, and .bvec files (default: input prefix)')

    args = parser.parse_args()

    nifti_write(args.input, args.prefix)


def pipeline(img, outdir):
    file_ = os.path.split(img)[-1].split('_pred.nii.gz')[0] + '_pred'
    dr = img.split(os.sep)[-2]
    path = os.path.join(outdir, dr)
    os.makedirs(path, exist_ok=True)
    prefix = os.path.join(path, file_)
    nifti_write(img, prefix)
# import vtk


# def readnrrd(filename):
#     """Read image in nrrd format."""
#     reader = vtk.vtkNrrdReader()
#     reader.SetFileName(filename)
#     reader.Update()
#     info = reader.GetInformation()
#     return reader.GetOutput(), info

# def writenifti(image,filename, info):
#     """Write nifti file."""
#     writer = vtk.vtkNIFTIImageWriter()
#     writer.SetInputData(image)
#     writer.SetFileName(filename)
#     writer.SetInformation(info)
#     writer.Write()

# m, info = readnrrd('/media/neubias/b0c7dd3a-8b12-435e-8303-2c331d05b365/DATA/Henry_data/mri.nrrd')
# writenifti(m, '/media/neubias/b0c7dd3a-8b12-435e-8303-2c331d05b365/DATA/Henry_data/mri_prueba2.nii', info)
if __name__ == '__main__':
    from glob import glob
    from multiprocessing import Pool
    from tqdm import tqdm

    data_root = '/mnts2d/med_data1/haotian/AMOS/first_round/valid/ps_slicer_nrrd'
    out_root = '/mnts2d/med_data1/haotian/AMOS/first_round/valid/ps_'
    os.makedirs(out_root,exist_ok=True)

    data_list = glob(data_root+'/*/*.nrrd')
    data = '/home/baihaotian/programs/wild/data/AMOS/63aac6c0425d136aca3c83458da27279_1.2.840.113704.1.111.1036.1547534440.1_1.2.840.113704.1.111.1036.1547534533.19_pred.nii.gz.seg.nrrd'
    
    nii_data = '/mnts2d/med_data1/haotian/AMOS/first_round/valid/firstround_select/labels/0/63aac6c0425d136aca3c83458da27279_1.2.840.113704.1.111.1036.1547534440.1_1.2.840.113704.1.111.1036.1547534533.19_pred.nii.gz'
    # headers = nib.load(nii_data).header
    in_ = nrrd.read(data)
    # print(np.unique(in_[0]))
    # pipeline(data,
    #  '/home/baihaotian/programs/wild/data/')
    headers = in_[1]
    # print(headers['Segment1_Name'])
    # for header in headers:
    #     if re.match('Segment[0-9]+_ID$',header) is not None:
    #         idx = header.split('_')[0].split('Segment')[-1]
    #         print(f'header: {header}, num: {idx}')
    #         print(headers[header])

    # img = alterLabelValueBySegName(in_)[0]
    # print(np.unique(img))
    # headers = alterLabelValueBySegName(in_)[1]
    # print('---------------------------')
    for i in headers:
        print(f'{i}, {headers[i]}')
    # for data in data_list:
    #     pipeline(data, out_root)
    # with Pool(6) as p:
    #     r = list(tqdm(p.map(partial(pipeline, outdir=out_root), data_list), total=len(data_list)))



