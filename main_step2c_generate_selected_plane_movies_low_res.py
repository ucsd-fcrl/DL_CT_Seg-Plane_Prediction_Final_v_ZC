#!/usr/bin/env python

# this step is optional. It can give you better DL product.

# this script generates adequate plane movies using the model set selection you defined in main_step2b.


import function_list as ff
import os
import numpy as np
import nibabel as nib 
import supplement
import pandas as pd
from PIL import Image
from Make_plane_movies import *
cg = supplement.Experiment()

WL = 500
WW = 800  

######### Define patient_list
main_folder = os.path.join(cg.save_dir,'DL_prediction_low_res')
patient_list = ff.find_all_target_files(['Abnormal/CVC1803*'],main_folder)

######### Load the spreadsheet 
sheet = pd.read_excel(os.path.join(cg.save_dir,'model_set_selection_example.xlsx'))



# main script (usually no need to change):
for patient in patient_list:
    patient_id = os.path.basename(patient)
    patient_class = os.path.basename(os.path.dirname(patient))
    print(patient_class,patient_id)

    # read model_set_selection spreadsheet info
    info = sheet[sheet['Patient_ID'] == patient_id]
    assert info.shape[0] == 1
    model_set_pick = [int(info.iloc[0]['seg (default = 3)']), int(info.iloc[0]['2CH']),int(info.iloc[0]['3CH']),int(info.iloc[0]['4CH']),int(info.iloc[0]['SAX'])]
    LAX_zoom = float(info.iloc[0]['Zoom_LAX (default = 1.2)'])
    SAX_zoom = float(info.iloc[0]['Zoom_SAX (default = 1.4)'])

    # define save folder
    save_folder = os.path.join(patient,'planes_pred_low_res_Final')
    ff.make_folder([save_folder, os.path.join(save_folder,'pngs')])

    # check whether already done
    if os.path.isfile(os.path.join(save_folder,patient_id+'_planes.mp4')) == 1:
        print('already done for this patient')
        continue

    # load plane vectors
    prepare = Prepare(main_folder,patient_class,patient_id, 0 ,cg.low_res_spacing,cg.low_res_dim)
    image_center, vector_2C, vector_3C, vector_4C, vector_SA = prepare.load_plane_vectors(batch_select = True, select_list = model_set_pick[1:])
    
    # get affine matrices
    volume_affine, A_2C, A_3C, A_4C = prepare.obtain_affine_matrix(vector_2C, vector_3C, vector_4C)

    # get the range of SAX stack using the LV segmentation, should already obtain this info as txt file in main_step2
    slice_num_info_file_name = os.path.join(patient,"slice_num_info_low_res_batch_" + str(model_set_pick[-1]) + ".txt")
    if os.path.isfile(slice_num_info_file_name) == 1:
        a, b = prepare.read_SAX_slice_num_file(slice_num_info_file_name)
    else: # txt file has not been made
        seg = nib.load(os.path.join(patient,'seg-pred/batch_'+str(model_set_pick[0]),'pred_s_0.nii.gz'))
        seg_LV = seg.get_fdata()
        a , b = prepare.define_SAX_range(vector_SA,image_center,seg_LV, model_set_pick[-1])
 
    # get a center list of 9-plane SAX stack
    _, center_list9, gap = prepare.define_SAX_planes_center_list(vector_SA, image_center, a, b)
   
    # get the image list
    img_list = ff.sort_timeframe(ff.find_all_target_files(['img-nii-1.5/*.nii.gz'],os.path.join(cg.image_data_dir,patient_class,patient_id)),2)

    # make the plane images for each time frame
    for img in img_list:
        volume_data = nib.load(img).get_fdata()
        time = ff.find_timeframe(img,2)
        save_path = os.path.join(save_folder,'pngs',str(time) +'.png')
        make = Make_Planes(main_folder, patient_class, patient_id, volume_data,[WL,WW], image_center,[vector_2C, vector_3C, vector_4C, vector_SA], volume_affine, center_list9 ,cg.low_res_spacing, cg.low_res_dim, zoom_factor_LAX = LAX_zoom, zoom_factor_SAX = SAX_zoom)
        make.plane_image(save_path, draw_lines = True) # draw_lines means use lines to represent SAX/LAX planes on LAX/SAX planes
        print('finish time frame ',time)

    # make cine movie
    pngs = ff.sort_timeframe(ff.find_all_target_files(['*.png'],os.path.join(save_folder,'pngs')),1)
    save_movie_path = os.path.join(save_folder,patient_id+'_planes.mp4')
    if len(pngs) == 16:
        fps = 15 # set 16 will cause bug
    elif len(pngs) > 20:
        fps = len(pngs)//2
    else:
        fps = len(pngs)
    ff.make_movies(save_movie_path,pngs,fps)





    
    