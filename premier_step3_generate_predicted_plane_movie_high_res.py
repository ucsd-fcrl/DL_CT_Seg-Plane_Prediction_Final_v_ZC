#!/usr/bin/env python

# this script generates plane product with LAX planes as lines plotted on SAX planes and same for SAX planes as well

# example of low and high res plane movie!!!!!!!!!!

import function_list as ff
import os
import numpy as np
import nibabel as nib 
import pandas as pd
import supplement
from PIL import Image
from Extract_vector_from_affines import *


cg = supplement.Experiment()



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


for patient in patient_list:
    patient_class = os.path.basename(os.path.dirname(patient))
    patient_id = os.path.basename(patient)
    print(patient_class,patient_id)

    # extract plane vectors from high-res plane NifTI images
    extract_vector = Vector_from_affine(main_folder,patient_class,patient_id,['2C','3C','4C','BASAL'])
    save_vector_folder = os.path.join(patient,'vector-pred-high-res-0.625')
    ff.make_folder([save_vector_folder])
    extract_vector.save_vectors(save_vector_folder)

    # get zoom factor
    if model_selection_done == 1:
        info = sheet[sheet['Patient_ID'] == patient_id]
        assert info.shape[0] == 1
        LAX_zoom = float(info.iloc[0]['Zoom_LAX (default = 1.2)'])
        SAX_zoom = float(info.iloc[0]['Zoom_SAX (default = 1.4)'])
    else:
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
    volume_dim = nib.load(os.path.join(cg.image_data_dir,patient_class,patient_id,'img-nii-0.625/0.nii.gz')).shape
    image_center = np.array([(volume_dim[0]-1)/2,(volume_dim[1]-1)/2,(volume_dim[-1]-1)/2]) 


    vector_2C = ff.get_ground_truth_vectors(os.path.join(main_folder,patient_class,patient_id,'vector-pred-high-res-0.625','pred_2C.npy'))
    vector_3C = ff.get_ground_truth_vectors(os.path.join(main_folder,patient_class,patient_id,'vector-pred-high-res-0.625','pred_3C.npy'))
    vector_4C = ff.get_ground_truth_vectors(os.path.join(main_folder,patient_class,patient_id,'vector-pred-high-res-0.625','pred_4C.npy'))
    vector_SA = ff.get_ground_truth_vectors(os.path.join(main_folder,patient_class,patient_id,'vector-pred-high-res-0.625','pred_BASAL.npy'))


        if sax_made == 'Horos':
            normal_vector_flip = 1
            normal_vector = -ff.normalize(np.cross(vector_SA['x'],vector_SA['y'])) 
        else:
            normal_vector_flip = 0
            normal_vector = ff.normalize(np.cross(vector_SA['x'],vector_SA['y'])) 

        # define plane num for SAX stack using low resolution segmentation
        slice_num_info_file_name = os.path.join(cg.save_dir,patient_class,patient_id,"slice_num_info_low_res_batch_" + str(batch_pick[-1]) + ".txt")
        if os.path.isfile(slice_num_info_file_name) == 1:
            slice_num_info_file = open(slice_num_info_file_name, 'r')
            Lines = slice_num_info_file.readlines()
            line1 = Lines[0];line2 = Lines[1]
            num1 = [i for i, e in enumerate(line1) if e == '='][-1]; a = int(line1[num1+2:len(line1)-1])
            num2 = [i for i, e in enumerate(line2) if e == '='][-1]; b = int(line2[num2+2:len(line2)])
            print(a,b)
        else:
            ValueError('no pre saved slice num')
        # else:   
        #     seg_dim = seg.shape
        #     seg_center = np.array([(seg_dim[0]-1)/2,(seg_dim[1]-1)/2,(seg_dim[-1]-1)/2]) 
        #     vector_SA_low_res = ff.get_predicted_vectors(os.path.join(cg.save_dir,patient_class,patient_id,'vector-pred/batch_'+str(batch_pick[-1]),'pred_BASAL_t.npy'),os.path.join(cg.save_dir,patient_class,patient_id,'vector-pred/batch_'+str(batch_pick[-1]),'pred_BASAL_r.npy'),[1,1,0.67], seg_center)
        #     a,b = ff.find_num_of_slices_in_SAX(np.zeros([160,160,1]),seg_center,vector_SA_low_res['t'],vector_SA_low_res['x'],vector_SA_low_res['y'],seg_data,normal_vector_flip,2.59 )
        #     print(a,b)
        #     slice_num_info_file = open(slice_num_info_file_name,"w+")
        #     slice_num_info_file.write("num of slices before basal = %d\nnum of slices after basal = %d" % (a, b))
        #     slice_num_info_file.close()
          

        # get affine matrix
        # volume_affine = ff.check_affine(os.path.join(cg.image_data_dir,patient_class,patient_id,'img-nii-1.5/0.nii.gz'))
        volume_affine = ff.check_affine(os.path.join(belong_path,'nii-images',patient_class,patient_id,'img-nii-1.5/0.nii.gz'))
        A_2C = ff.get_affine_from_vectors(np.zeros(plane_image_size),volume_affine,vector_2C,1.0)
        A_3C = ff.get_affine_from_vectors(np.zeros(plane_image_size),volume_affine,vector_3C,1.0)
        A_4C = ff.get_affine_from_vectors(np.zeros(plane_image_size),volume_affine,vector_4C,1.0)
      

        # get a center list of SAX stack
        # pix_dim = ff.get_voxel_size(os.path.join(cg.image_data_dir,patient_class,patient_id,'img-nii-1.5/0.nii.gz'))
        pix_dim = ff.get_voxel_size(os.path.join(belong_path,'nii-images',patient_class,patient_id,'img-nii-1.5/0.nii.gz'))
        pix_size = ff.length(pix_dim)
        center_list = ff.find_center_list_whole_stack(image_center + vector_SA['t'],normal_vector,a,b,8,pix_size)
        print('pix_size: ',pix_dim,pix_size)
        
        # get the index of each planes of 9-plane SAX stack (9 planes should start from MV and end with apex, convering the whole LV)
        index_list,center_list9,gap = ff.resample_SAX_stack_into_particular_num_of_planes(range(2,center_list.shape[0]),9,center_list)
        # if gap < 1:
        #     ValueError('no LV segmentation')

        
        # volume_list = ff.sort_timeframe(ff.find_all_target_files(['img-nii-1.5/*.nii.gz'],os.path.join(cg.image_data_dir,patient_class,patient_id)),2)
        volume_list = ff.sort_timeframe(ff.find_all_target_files(['img-nii-1.5/*.nii.gz'],os.path.join(belong_path,'nii-images',patient_class,patient_id)),2)
        

        for v in volume_list:
            volume = nib.load(v)
            volume_data = volume.get_fdata()
            if len(volume_data.shape) > 3:
                print('this data has more than 3 dimen')
                volume_data = volume_data[:,:,:,1]
                print(volume_data.shape)
                assert len(volume_data.shape) == 3
            elif len(volume_data.shape) < 3:
                print('this data has less than 3 dimen')
                continue
            else:
                aa = 1

            time = ff.find_timeframe(v,2)
            save_path = os.path.join(save_folder,'pngs',str(time) +'.png')
            ff.make_folder([os.path.dirname(save_path)])
            plane_image(save_path,volume_data,plane_image_size,WL,WW,image_center, vector_2C,vector_3C,vector_4C,vector_SA,A_2C,A_3C,A_4C,center_list9,zoom_factor_lax, zoom_factor_sax)
            print('finish time '+str(time))

        # make the movie
        pngs = ff.sort_timeframe(ff.find_all_target_files(['*.png'],os.path.join(save_folder,'pngs')),1)
        save_movie_path = os.path.join(save_folder, patient_id+'_planes.mp4')
        print(len(pngs),save_movie_path)
        if len(pngs) == 16:
            fps = 15 # set 16 will cause bug
        elif len(pngs) > 20:
            fps = len(pngs)//2
        else:
            fps = len(pngs)
        ff.make_movies(save_movie_path,pngs,fps)





    # zoom_factor_lax = 1.1
    # zoom_factor_save = 1.25
    
    # if batch_pre_select == 1:
    #     case = csv_file[(csv_file['Patient_Class'] == patient_class) & (csv_file['Patient_ID'] == patient_id)]
    #     assert case.shape[0] == 1
    #     if case.iloc[0]['2C'] == 'x' and case.iloc[0]['SAX'] == 'x':
    #         print('exclude')
    #         continue
    #     else:
    #         batch_pick = [int(case.iloc[0]['2C']),int(case.iloc[0]['3C']),int(case.iloc[0]['4C']),int(case.iloc[0]['SAX'])]
    #         zoom_factor_lax = float(case.iloc[0]['zoom_factor_lax'])
    #         zoom_factor_sax = float(case.iloc[0]['zoom_factor_sax'])
    # print(batch_pick,zoom_factor_lax,zoom_factor_sax)

    # save_folder = os.path.join(cg.final_dir,patient_class,patient_id)
    # ff.make_folder([os.path.dirname(save_folder),save_folder])



    # if os.path.isfile(os.path.join(save_folder,patient_id+'_planes.mp4')) == 1:
    #     print('already done')
    #     continue
    #     #os.remove(os.path.join(save_folder,patient_id+'_planes.mp4'))
    # else:
           
    #     # this part just for VR data
    #     if os.path.isdir(os.path.join(cg.main_data_dir,'nii-images',patient_class,patient_id)) == 1:
    #         belong_path = cg.main_data_dir
    #     elif os.path.isdir(os.path.join(cg.main_data_dir,'2020_after_Junes/nii-images',patient_class,patient_id)) == 1:
    #         belong_path = os.path.join(cg.main_data_dir,'2020_after_Junes')
    #     else:
    #         ValueError('no this case')


    #     # seg_file = os.path.join(cg.save_dir,patient_class,patient_id,'seg-pred','batch_'+str(batch_pick[0])+'/pred_s_0.nii.gz')
    #     # seg = nib.load(seg_file)
    #     # seg_data = seg.get_fdata()

    #     #volume_dim = nib.load(os.path.join(cg.image_data_dir,patient_class,patient_id,'img-nii-1.5/0.nii.gz')).shape
    #     volume_dim = nib.load(os.path.join(cg.local_dir,patient_class,patient_id,'img-nii-1.5/0.nii.gz')).shape
    #     image_center = np.array([(volume_dim[0]-1)/2,(volume_dim[1]-1)/2,(volume_dim[-1]-1)/2]) 

    #     # # load vectors
    #     vector_2C = ff.get_predicted_vectors(os.path.join(cg.save_dir,patient_class,patient_id,'vector-pred/batch_'+str(batch_pick[0]),'pred_2C_t.npy'),os.path.join(cg.save_dir,patient_class,patient_id,'vector-pred/batch_'+str(batch_pick[0]),'pred_2C_r.npy'),scale, image_center)
    #     vector_3C = ff.get_predicted_vectors(os.path.join(cg.save_dir,patient_class,patient_id,'vector-pred/batch_'+str(batch_pick[1]),'pred_3C_t.npy'),os.path.join(cg.save_dir,patient_class,patient_id,'vector-pred/batch_'+str(batch_pick[1]),'pred_3C_r.npy'),scale, image_center)
    #     vector_4C = ff.get_predicted_vectors(os.path.join(cg.save_dir,patient_class,patient_id,'vector-pred/batch_'+str(batch_pick[2]),'pred_4C_t.npy'),os.path.join(cg.save_dir,patient_class,patient_id, 'vector-pred/batch_'+str(batch_pick[2]),'pred_4C_r.npy'),scale, image_center)
    #     vector_SA = ff.get_predicted_vectors(os.path.join(cg.save_dir,patient_class,patient_id,'vector-pred/batch_'+str(batch_pick[3]),'pred_BASAL_t.npy'),os.path.join(cg.save_dir,patient_class,patient_id,'vector-pred/batch_'+str(batch_pick[3]),'pred_BASAL_r.npy'),scale, image_center)
    #     # vector_2C = ff.get_ground_truth_vectors(os.path.join(cg.save_dir,patient_class,patient_id,'vector-pred-high-res-0.625','pred_2C.npy'))
    #     # vector_3C = ff.get_ground_truth_vectors(os.path.join(cg.save_dir,patient_class,patient_id,'vector-pred-high-res-0.625','pred_3C.npy'))
    #     # vector_4C = ff.get_ground_truth_vectors(os.path.join(cg.save_dir,patient_class,patient_id,'vector-pred-high-res-0.625','pred_4C.npy'))
    #     # vector_SA = ff.get_ground_truth_vectors(os.path.join(cg.save_dir,patient_class,patient_id,'vector-pred-high-res-0.625','pred_BASAL.npy'))


    #     if sax_made == 'Horos':
    #         normal_vector_flip = 1
    #         normal_vector = -ff.normalize(np.cross(vector_SA['x'],vector_SA['y'])) 
    #     else:
    #         normal_vector_flip = 0
    #         normal_vector = ff.normalize(np.cross(vector_SA['x'],vector_SA['y'])) 

    #     # define plane num for SAX stack using low resolution segmentation
    #     slice_num_info_file_name = os.path.join(cg.save_dir,patient_class,patient_id,"slice_num_info_low_res_batch_" + str(batch_pick[-1]) + ".txt")
    #     if os.path.isfile(slice_num_info_file_name) == 1:
    #         slice_num_info_file = open(slice_num_info_file_name, 'r')
    #         Lines = slice_num_info_file.readlines()
    #         line1 = Lines[0];line2 = Lines[1]
    #         num1 = [i for i, e in enumerate(line1) if e == '='][-1]; a = int(line1[num1+2:len(line1)-1])
    #         num2 = [i for i, e in enumerate(line2) if e == '='][-1]; b = int(line2[num2+2:len(line2)])
    #         print(a,b)
    #     else:
    #         ValueError('no pre saved slice num')
    #     # else:   
    #     #     seg_dim = seg.shape
    #     #     seg_center = np.array([(seg_dim[0]-1)/2,(seg_dim[1]-1)/2,(seg_dim[-1]-1)/2]) 
    #     #     vector_SA_low_res = ff.get_predicted_vectors(os.path.join(cg.save_dir,patient_class,patient_id,'vector-pred/batch_'+str(batch_pick[-1]),'pred_BASAL_t.npy'),os.path.join(cg.save_dir,patient_class,patient_id,'vector-pred/batch_'+str(batch_pick[-1]),'pred_BASAL_r.npy'),[1,1,0.67], seg_center)
    #     #     a,b = ff.find_num_of_slices_in_SAX(np.zeros([160,160,1]),seg_center,vector_SA_low_res['t'],vector_SA_low_res['x'],vector_SA_low_res['y'],seg_data,normal_vector_flip,2.59 )
    #     #     print(a,b)
    #     #     slice_num_info_file = open(slice_num_info_file_name,"w+")
    #     #     slice_num_info_file.write("num of slices before basal = %d\nnum of slices after basal = %d" % (a, b))
    #     #     slice_num_info_file.close()
          

    #     # get affine matrix
    #     # volume_affine = ff.check_affine(os.path.join(cg.image_data_dir,patient_class,patient_id,'img-nii-1.5/0.nii.gz'))
    #     volume_affine = ff.check_affine(os.path.join(belong_path,'nii-images',patient_class,patient_id,'img-nii-1.5/0.nii.gz'))
    #     A_2C = ff.get_affine_from_vectors(np.zeros(plane_image_size),volume_affine,vector_2C,1.0)
    #     A_3C = ff.get_affine_from_vectors(np.zeros(plane_image_size),volume_affine,vector_3C,1.0)
    #     A_4C = ff.get_affine_from_vectors(np.zeros(plane_image_size),volume_affine,vector_4C,1.0)
      

    #     # get a center list of SAX stack
    #     # pix_dim = ff.get_voxel_size(os.path.join(cg.image_data_dir,patient_class,patient_id,'img-nii-1.5/0.nii.gz'))
    #     pix_dim = ff.get_voxel_size(os.path.join(belong_path,'nii-images',patient_class,patient_id,'img-nii-1.5/0.nii.gz'))
    #     pix_size = ff.length(pix_dim)
    #     center_list = ff.find_center_list_whole_stack(image_center + vector_SA['t'],normal_vector,a,b,8,pix_size)
    #     print('pix_size: ',pix_dim,pix_size)
        
    #     # get the index of each planes of 9-plane SAX stack (9 planes should start from MV and end with apex, convering the whole LV)
    #     index_list,center_list9,gap = ff.resample_SAX_stack_into_particular_num_of_planes(range(2,center_list.shape[0]),9,center_list)
    #     # if gap < 1:
    #     #     ValueError('no LV segmentation')

        
    #     # volume_list = ff.sort_timeframe(ff.find_all_target_files(['img-nii-1.5/*.nii.gz'],os.path.join(cg.image_data_dir,patient_class,patient_id)),2)
    #     volume_list = ff.sort_timeframe(ff.find_all_target_files(['img-nii-1.5/*.nii.gz'],os.path.join(belong_path,'nii-images',patient_class,patient_id)),2)
        

    #     for v in volume_list:
    #         volume = nib.load(v)
    #         volume_data = volume.get_fdata()
    #         if len(volume_data.shape) > 3:
    #             print('this data has more than 3 dimen')
    #             volume_data = volume_data[:,:,:,1]
    #             print(volume_data.shape)
    #             assert len(volume_data.shape) == 3
    #         elif len(volume_data.shape) < 3:
    #             print('this data has less than 3 dimen')
    #             continue
    #         else:
    #             aa = 1

    #         time = ff.find_timeframe(v,2)
    #         save_path = os.path.join(save_folder,'pngs',str(time) +'.png')
    #         ff.make_folder([os.path.dirname(save_path)])
    #         plane_image(save_path,volume_data,plane_image_size,WL,WW,image_center, vector_2C,vector_3C,vector_4C,vector_SA,A_2C,A_3C,A_4C,center_list9,zoom_factor_lax, zoom_factor_sax)
    #         print('finish time '+str(time))

    #     # make the movie
    #     pngs = ff.sort_timeframe(ff.find_all_target_files(['*.png'],os.path.join(save_folder,'pngs')),1)
    #     save_movie_path = os.path.join(save_folder, patient_id+'_planes.mp4')
    #     print(len(pngs),save_movie_path)
    #     if len(pngs) == 16:
    #         fps = 15 # set 16 will cause bug
    #     elif len(pngs) > 20:
    #         fps = len(pngs)//2
    #     else:
    #         fps = len(pngs)
    #     ff.make_movies(save_movie_path,pngs,fps)


    

  
    
