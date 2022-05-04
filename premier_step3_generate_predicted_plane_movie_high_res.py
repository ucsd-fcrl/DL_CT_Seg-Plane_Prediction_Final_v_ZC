#!/usr/bin/env python

# this script generates high resolution cine movie of cardiac imaging planes!
# see an example of this movie in example_files/Example_plane_cine_image.mp4 

import function_list as ff
import os
import numpy as np
import nibabel as nib 
import pandas as pd
import supplement
from Extract_vector_from_affines import *
from Make_plane_movies import *


cg = supplement.Experiment()
WL = 500
WW = 800

####### Define patient list
main_folder = os.path.join(cg.save_dir, 'DL_prediction_high_res')
patient_list = ff.find_all_target_files(['Abnormal/CVC1803*'],main_folder)

###### Load model_set_selection spreadsheet
model_selection_done = 1
sheet_name = os.path.join(cg.save_dir,'model_set_selection_example.xlsx')
if os.path.isfile == 1:
    sheet = pd.read_excel(os.path.join(cg.save_dir,'model_set_selection_example.xlsx'))
else:
    model_selection_done = 0



# main script (usually no need to change)
for patient in patient_list:
    patient_class = os.path.basename(os.path.dirname(patient))
    patient_id = os.path.basename(patient)
    print(patient_class,patient_id)

    # extract plane vectors from high-res plane NifTI images
    extract_vector = Vector_from_affine(main_folder,patient_class,patient_id,['2C','3C','4C','BASAL'])
    save_vector_folder = os.path.join(patient,'vector-pred-high-res-'+str(cg.high_res_spacing))
    ff.make_folder([save_vector_folder])
    extract_vector.save_vectors(save_vector_folder)

    # get model_set_selection info and zoom factors
    if model_selection_done == 1:
        info = sheet[sheet['Patient_ID'] == patient_id]
        assert info.shape[0] == 1
        model_select_SA = int(info.iloc[0]['SAX'])
        LAX_zoom = float(info.iloc[0]['Zoom_LAX (default = 1.2)'])
        SAX_zoom = float(info.iloc[0]['Zoom_SAX (default = 1.4)'])
    else: # default
        model_select_SA = 0
        LAX_zoom = 1.2
        SAX_zoom = 1.4

    # define save folder
    save_folder = os.path.join(patient,'planes_pred_high_res_Final')
    ff.make_folder([save_folder,os.path.join(save_folder,'pngs')])

    # check already done
    if os.path.isfile(os.path.join(save_folder,patient_id+'_planes.mp4')) == 1:
        print('already done')
        continue

    # load vectors
    prepare = Prepare_premier(main_folder,patient_class,patient_id, 0, cg.high_res_spacing,cg.high_res_dim)
    image_center, vector_2C, vector_3C, vector_4C, vector_SA, normal_vector_SA = prepare.load_plane_vectors_high_res(os.path.basename(save_vector_folder))

    # get affine matrix
    volume_affine,_, _, _ = prepare.obtain_affine_matrix(vector_2C, vector_3C, vector_4C)

    # define SAX range:
    slice_num_info_file_name = os.path.join(cg.save_dir,'DL_prediction_low_res',patient_class,patient_id,"slice_num_info_low_res_batch_" + str(model_select_SA) + ".txt")
    if os.path.isfile(slice_num_info_file_name) == 1:
        a, b = prepare.read_SAX_slice_num_file(slice_num_info_file_name)
    else:
        ValueError('no pre-saved slice num, you should have it in main_step2')

    # get a center list of 9-plane SAX stack
    _, center_list9, gap = prepare.define_SAX_planes_center_list(vector_SA, image_center, a, b, normal_vector_SA)

    # get the image list
    img_list = ff.sort_timeframe(ff.find_all_target_files(['img-nii-'+str(cg.high_res_spacing)+'/*.nii.gz'],os.path.join(cg.image_data_dir,patient_class,patient_id)),2)

    # make the plane images for each time frame
    for img in img_list:
        volume_data = nib.load(img).get_fdata()
        time = ff.find_timeframe(img,2)
        save_path = os.path.join(save_folder,'pngs',str(time) +'.png')
        make = Make_Planes(main_folder, patient_class, patient_id, volume_data,[WL,WW], image_center,[vector_2C, vector_3C, vector_4C, vector_SA], volume_affine, center_list9 ,cg.high_res_spacing, cg.high_res_dim, zoom_factor_LAX = LAX_zoom, zoom_factor_SAX = SAX_zoom)
        make.plane_image(save_path, draw_lines = True) # draw_lines means use lines to represent SAX/LAX planes on LAX/SAX planes
        print('finish time frame ',time)

    # make the movie
    pngs = ff.sort_timeframe(ff.find_all_target_files(['*.png'],os.path.join(save_folder,'pngs')),1)
    save_movie_path = os.path.join(save_folder, patient_id+'_planes.mp4')
    ff.make_movies(save_movie_path,pngs)