#!/usr/bin/env python

# In premier, we are going to convert our low-resolution DL plane into high-resolution DL plane.
# We do so not by simply up-sampling the plane image but by a series of actions to re-slice the same plane in high-resolution CT volume.
# default pixel_dim = 0.625, it's adequate for physicians to visualize the regional wall motion.

# the step 1 is to prepare the NifTi image of the low-res plane.

import function_list as ff
import os
import math
import numpy as np
import nibabel as nib 
import supplements
import pandas as pd
from Make_plane_movies import *
cg = supplements.Experiment()

WL = 500
WW = 800  
scale = [1,1,0.67] # default

######### Define patient_list
main_folder = os.path.join(cg.save_dir,'DL_prediction_low_res')
patient_list = ff.find_all_target_files(['Abnormal/CVC1803*'],main_folder)

######### Load the spreadsheet of model_set_selection
model_selection_done = 1
sheet_name = os.path.join(cg.save_dir,'model_set_selection_example.xlsx')
if os.path.isfile == 1:
    sheet = pd.read_excel(os.path.join(cg.save_dir,'model_set_selection_example.xlsx'))
else:
    model_selection_done = 0

######### Define planes you want to convert to NifTI
task_list = ['2C','3C','4C','BASAL']


# main script (usually no need to change):
for patient in patient_list:
    patient_id = os.path.basename(patient)
    patient_class = os.path.basename(os.path.dirname(patient))
    print(patient_class,patient_id)

    if model_selection_done == 1:
        info = sheet[sheet['Patient_ID'] == patient_id]
        assert info.shape[0] == 1
        model_set_pick = [int(info.iloc[0]['2CH']),int(info.iloc[0]['3CH']),int(info.iloc[0]['4CH']),int(info.iloc[0]['SAX'])]
    else:
        model_set_pick = [0,0,0,0] # default
    
    # define save folder
    save_folder = os.path.join(patient,'planes_pred_low_res_nii')
    ff.make_folder([save_folder])

    for num in range(0,len(task_list)):
        imaging_plane = task_list[num]

        save_file = os.path.join(save_folder,'pred_'+imaging_plane+'.nii.gz')
        if os.path.isfile(save_file) == 1:
            print('already done')
            continue
    
        volume_file = os.path.join(cg.image_data_dir,patient_class,patient_id,'img-nii-1.5/0.nii.gz') # only need the plane at one time frame
        volume_data = nib.load(volume_file).get_fdata()
        image_center = np.array([(volume_data.shape[0]-1)/2,(volume_data.shape[1]-1)/2,(volume_data.shape[-1]-1)/2]) 

        # load vectors
        vector = ff.get_predicted_vectors(os.path.join(patient,'vector-pred/batch_'+str(model_set_pick[num]),'pred_'+imaging_plane+'_t.npy'),os.path.join(patient,'vector-pred/batch_'+str(model_set_pick[num]),'pred_'+imaging_plane+'_r.npy'),scale,image_center)

        # get affine matrix
        volume_affine = ff.check_affine(os.path.join(cg.image_data_dir,patient_class,patient_id,'img-nii-1.5/0.nii.gz'))
        plane_affine = ff.get_affine_from_vectors(np.zeros(cg.low_res_dim),volume_affine,vector)

        # reslice
        interpolation = ff.define_interpolation(volume_data,Fill_value = volume_data.min(),Method='linear')
        plane = ff.reslice_mpr(np.zeros(cg.low_res_dim),image_center + vector['t'], ff.normalize(vector['x']), ff.normalize(vector['y']),scale[0],scale[1],interpolation)

        # save
        nii = nib.Nifti1Image(plane, plane_affine)
        nib.save(nii, save_file)
        


   