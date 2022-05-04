#!/usr/bin/env python

# this script generates a cardiac imaging plane in low resolution and save into nii files


import function_list as ff
import os
import math
import numpy as np
import nibabel as nib 
import supplement
import pandas as pd
cg = supplement.Experiment()

WL = 500
WW = 800
plane_image_size = [160,160,1]
    
scale = [1,1,0.67]
zoom_factor = 1 # in case the background in the plane is too large 


# Define planes and batch
task_list = ['2C','3C','4C','BASAL']
task_num = [0,1,2,3]
batch_select_excel_file = os.path.join('/Data/McVeighLabSuper/projects/Zhennong','Cases_for_Cardiowise_list_batch_selection.csv')
if os.path.isfile(batch_select_excel_file) == 1:
    batch_pre_select = 1
    csv_file = pd.read_csv(batch_select_excel_file)
else:
    batch_pre_select = 0
    batch_pick = [3,4,4,4,4] #[seg,2C,3C,4C]

# define patient_list
patient_list = ff.find_all_target_files(['Abnormal/*','Normal/*'],cg.save_dir)

# make plane nii files for all kinds of planes
for patient in patient_list:
    patient_id = os.path.basename(patient)
    patient_class = os.path.basename(os.path.dirname(patient))
    print(patient_class,patient_id)

    if batch_pre_select == 1:
        case = csv_file[csv_file['Patient_ID'] == patient_id]
        assert case.shape[0] == 1
        if case.iloc[0]['2C'] == 'x' and case.iloc[0]['SAX'] == 'x':
            print('exclude')
            continue
        else:
            batch_pick = [int(case.iloc[0]['2C']),int(case.iloc[0]['3C']),int(case.iloc[0]['4C']),int(case.iloc[0]['SAX'])]
    print(batch_pick)

    

    save_folder = os.path.join(patient,'planes_pred_low_res_nii')
    ff.make_folder([os.path.dirname(save_folder),save_folder])


    for num in task_num:
        imaging_plane = task_list[num]
        print(imaging_plane)

        save_file = os.path.join(save_folder,'pred_'+imaging_plane+'.nii.gz')
        if os.path.isfile(save_file) == 1:
            print('already done')
            continue
    
        volume_file = os.path.join(cg.local_dir,patient_class,patient_id,'img-nii-1.5/0.nii.gz')
        volume_dim = nib.load(volume_file).shape
        volume_data = nib.load(volume_file).get_fdata()
        image_center = np.array([(volume_dim[0]-1)/2,(volume_dim[1]-1)/2,(volume_dim[-1]-1)/2]) 

        # load vectors
        vector = ff.get_predicted_vectors(os.path.join(patient,'vector-pred/batch_'+str(batch_pick[num]),'pred_'+imaging_plane+'_t.npy'),os.path.join(patient,'vector-pred/batch_'+str(batch_pick[num]),'pred_'+imaging_plane+'_r.npy'),scale,image_center)

        # get affine matrix
        volume_affine = ff.check_affine(os.path.join(cg.image_data_dir,patient_class,patient_id,'img-nii-1.5/0.nii.gz'))
        A = ff.get_affine_from_vectors(np.zeros(plane_image_size),volume_affine,vector)

        # reslice
        interpolation = ff.define_interpolation(volume_data,Fill_value = volume_data.min(),Method='linear')
        plane = ff.reslice_mpr(np.zeros(plane_image_size),image_center + vector['t'], ff.normalize(vector['x']), ff.normalize(vector['y']),scale[0],scale[1],interpolation)
        print(plane.shape)

        # save
        nii_array = nib.Nifti1Image(plane, A)
        nib.save(nii_array, save_file)
        


   