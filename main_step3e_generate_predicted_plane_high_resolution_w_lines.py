#!/usr/bin/env python

# this script generates plane product with LAX planes as lines plotted on SAX planes and same for SAX planes as well


import function_list as ff
import os
import math
import numpy as np
import nibabel as nib 
import supplement
from PIL import Image
import pandas as pd
cg = supplement.Experiment()

WL = 500
WW = 800 #####change the width!!!!!!!!!!!
plane_image_size = [160,160,1]#[480,480,1]
if plane_image_size[0] == 480:
    color_box_size = [30,60]
else:
    color_box_size = [10,20]
sax_made = 'matlab'
scale = [1,1,0.67]
#zoom_factor = 1 # in case the background in the plane is too large 


# function to make the image
def plane_image(save_path,volume_data,plane_image_size,WL,WW,image_center, vector_2C,vector_3C,vector_4C,vector_SA, A_2C, A_3C, A_4C ,center_list,zoom_factor_lax = 1.1, zoom_factor_sax = 1.25):
    
    # define interpolation matrix
    inter = ff.define_interpolation(volume_data,Fill_value=volume_data.min(),Method='linear')
    
    
    # reslice long axis
    twoc = ff.reslice_mpr(np.zeros(plane_image_size),image_center + vector_2C['t'],vector_2C['x'],vector_2C['y'],vector_2C['s'][0]/zoom_factor_lax,vector_2C['s'][1]/zoom_factor_lax,inter)
    threec = ff.reslice_mpr(np.zeros(plane_image_size),image_center + vector_3C['t'],vector_3C['x'],vector_3C['y'],vector_3C['s'][0]/zoom_factor_lax,vector_3C['s'][1]/zoom_factor_lax,inter)
    fourc = ff.reslice_mpr(np.zeros(plane_image_size),image_center + vector_4C['t'],vector_4C['x'],vector_4C['y'],vector_4C['s'][0]/zoom_factor_lax,vector_4C['s'][1]/zoom_factor_lax,inter)


    # reslice short axis
  
    sax_collection = []
    for i in range(0,9):
        sax_collection.append(ff.reslice_mpr(np.zeros(plane_image_size),center_list[i],vector_SA['x'],vector_SA['y'],vector_SA['s'][0]/zoom_factor_sax,vector_SA['s'][1]/zoom_factor_sax,inter))
    assert len(sax_collection) == 9


    # get updated affine_matrix for zoomed image
    A_2C_zoomed = ff.get_affine_from_vectors(np.zeros(plane_image_size),volume_affine,vector_2C,zoom_factor_lax)
    A_3C_zoomed = ff.get_affine_from_vectors(np.zeros(plane_image_size),volume_affine,vector_3C,zoom_factor_lax)
    A_4C_zoomed = ff.get_affine_from_vectors(np.zeros(plane_image_size),volume_affine,vector_4C,zoom_factor_lax)

    # draw intersection lines: LAX lines on SAX
    sax_w_2c = []; sax_w_3c = []; sax_w_4c = []
    for ii in range(0,9):
        line2c,_,_ = ff.draw_plane_intersection(sax_collection[ii],vector_2C['x'],vector_2C['y'],A_2C_zoomed,ff.get_affine_from_vectors(np.zeros(plane_image_size),volume_affine,ff.make_matrix_for_any_plane_in_SAX_stack(center_list[ii],image_center,vector_SA),zoom_factor_sax),volume_affine)
        line3c,_,_ = ff.draw_plane_intersection(sax_collection[ii],vector_3C['x'],vector_3C['y'],A_3C_zoomed,ff.get_affine_from_vectors(np.zeros(plane_image_size),volume_affine,ff.make_matrix_for_any_plane_in_SAX_stack(center_list[ii],image_center,vector_SA),zoom_factor_sax),volume_affine)
        line4c,_,_ = ff.draw_plane_intersection(sax_collection[ii],vector_4C['x'],vector_4C['y'],A_4C_zoomed,ff.get_affine_from_vectors(np.zeros(plane_image_size),volume_affine,ff.make_matrix_for_any_plane_in_SAX_stack(center_list[ii],image_center,vector_SA),zoom_factor_sax),volume_affine)
        sax_w_2c.append(line2c)
        sax_w_3c.append(line3c)
        sax_w_4c.append(line4c)
    
    # draw intersection lines: SAX lines on LAX
    twoc_line = np.copy(twoc); threec_line = np.copy(threec); fourc_line = np.copy(fourc)
    twoc_line,_,_ = ff.draw_plane_intersection(twoc_line,vector_SA['x'],vector_SA['y'],ff.get_affine_from_vectors(np.zeros(plane_image_size),volume_affine,ff.make_matrix_for_any_plane_in_SAX_stack(center_list[2],image_center,vector_SA),zoom_factor_sax),A_2C_zoomed,volume_affine)
    twoc_line,_,_ = ff.draw_plane_intersection(twoc_line,vector_SA['x'],vector_SA['y'],ff.get_affine_from_vectors(np.zeros(plane_image_size),volume_affine,ff.make_matrix_for_any_plane_in_SAX_stack(center_list[4],image_center,vector_SA),zoom_factor_sax),A_2C_zoomed,volume_affine)
    twoc_line,_,_ = ff.draw_plane_intersection(twoc_line,vector_SA['x'],vector_SA['y'],ff.get_affine_from_vectors(np.zeros(plane_image_size),volume_affine,ff.make_matrix_for_any_plane_in_SAX_stack(center_list[6],image_center,vector_SA),zoom_factor_sax),A_2C_zoomed,volume_affine)
    threec_line,_,_ = ff.draw_plane_intersection(threec_line,vector_SA['x'],vector_SA['y'],ff.get_affine_from_vectors(np.zeros(plane_image_size),volume_affine,ff.make_matrix_for_any_plane_in_SAX_stack(center_list[2],image_center,vector_SA),zoom_factor_sax),A_3C_zoomed,volume_affine)
    threec_line,_,_ = ff.draw_plane_intersection(threec_line,vector_SA['x'],vector_SA['y'],ff.get_affine_from_vectors(np.zeros(plane_image_size),volume_affine,ff.make_matrix_for_any_plane_in_SAX_stack(center_list[4],image_center,vector_SA),zoom_factor_sax),A_3C_zoomed,volume_affine)
    threec_line,_,_ = ff.draw_plane_intersection(threec_line,vector_SA['x'],vector_SA['y'],ff.get_affine_from_vectors(np.zeros(plane_image_size),volume_affine,ff.make_matrix_for_any_plane_in_SAX_stack(center_list[6],image_center,vector_SA),zoom_factor_sax),A_3C_zoomed,volume_affine)
    fourc_line,_,_ = ff.draw_plane_intersection(fourc_line,vector_SA['x'],vector_SA['y'],ff.get_affine_from_vectors(np.zeros(plane_image_size),volume_affine,ff.make_matrix_for_any_plane_in_SAX_stack(center_list[2],image_center,vector_SA),zoom_factor_sax),A_4C_zoomed,volume_affine)
    fourc_line,_,_ = ff.draw_plane_intersection(fourc_line,vector_SA['x'],vector_SA['y'],ff.get_affine_from_vectors(np.zeros(plane_image_size),volume_affine,ff.make_matrix_for_any_plane_in_SAX_stack(center_list[4],image_center,vector_SA),zoom_factor_sax),A_4C_zoomed,volume_affine)
    fourc_line,_,_ = ff.draw_plane_intersection(fourc_line,vector_SA['x'],vector_SA['y'],ff.get_affine_from_vectors(np.zeros(plane_image_size),volume_affine,ff.make_matrix_for_any_plane_in_SAX_stack(center_list[6],image_center,vector_SA),zoom_factor_sax),A_4C_zoomed,volume_affine)

    # normalize by WL and WW and then orient
    twoc_n = ff.set_window(twoc,WL,WW); twoc_n = np.flip(twoc_n.T,0)
    twoc_line_n = ff.set_window(twoc_line,WL,WW); twoc_line_n = np.flip(twoc_line_n.T,0)
    threec_n = ff.set_window(threec,WL,WW); threec_n = np.flip(threec_n.T,0)
    threec_line_n = ff.set_window(threec_line,WL,WW); threec_line_n = np.flip(threec_line_n.T,0)
    fourc_n = ff.set_window(fourc,WL,WW); fourc_n = np.flip(fourc_n.T,0)
    fourc_line_n = ff.set_window(fourc_line,WL,WW); fourc_line_n = np.flip(fourc_line_n.T,0)

    sax_w_2c_n = []; sax_w_3c_n = []; sax_w_4c_n = []
    for iii in range(0,9):
        line2c_n = ff.set_window(sax_w_2c[iii],WL,WW); line2c_n = line2c_n.T
        line3c_n = ff.set_window(sax_w_3c[iii],WL,WW); line3c_n = line3c_n.T
        line4c_n = ff.set_window(sax_w_4c[iii],WL,WW); line4c_n = line4c_n.T
        sax_w_2c_n.append(line2c_n)
        sax_w_3c_n.append(line3c_n)
        sax_w_4c_n.append(line4c_n)
    assert len(sax_w_4c_n) == 9
    

    # make image
    [h,w,d] = plane_image_size
    I = np.zeros((h*3,w*4,3))
    I[0:h,0:w,0] = ff.color_box(twoc_n,color_box_size[0],color_box_size[1]); I[0:h,0:w,1] = twoc_line_n; I[0:h,0:w,2] = twoc_n
    I[h:h*2,0:w,0] = threec_n; I[h:h*2,0:w,1] = ff.color_box(threec_line_n,color_box_size[0],color_box_size[1]); I[h:h*2,0:w,2] = threec_n
    I[h*2:h*3,0:w,0] = fourc_n; I[h*2:h*3,0:w,1] = fourc_line_n; I[h*2:h*3,0:w,2] = ff.color_box(fourc_n,color_box_size[0],color_box_size[1])
    I[0:h,w:w*2,0] = sax_w_2c_n[0]; I[0:h,w:w*2,1] = sax_w_3c_n[0]; I[0:h,w:w*2,2] = sax_w_4c_n[0]
    I[0:h,w*2:w*3,0] = sax_w_2c_n[1]; I[0:h,w*2:w*3,1] = sax_w_3c_n[1]; I[0:h,w*2:w*3,2] = sax_w_4c_n[1]
    I[0:h,w*3:w*4,0] = sax_w_2c_n[2]; I[0:h,w*3:w*4,1] = sax_w_3c_n[2]; I[0:h,w*3:w*4,2] = sax_w_4c_n[2]
    I[h:h*2,w:w*2,0] = sax_w_2c_n[3]; I[h:h*2,w:w*2,1] = sax_w_3c_n[3]; I[h:h*2,w:w*2,2] = sax_w_4c_n[3]
    I[h:h*2,w*2:w*3,0] = sax_w_2c_n[4]; I[h:h*2,w*2:w*3,1] = sax_w_3c_n[4]; I[h:h*2,w*2:w*3,2] = sax_w_4c_n[4]
    I[h:h*2,w*3:w*4,0] = sax_w_2c_n[5]; I[h:h*2,w*3:w*4,1] = sax_w_3c_n[5]; I[h:h*2,w*3:w*4,2] = sax_w_4c_n[5]
    I[h*2:h*3,w:w*2,0] = sax_w_2c_n[6]; I[h*2:h*3,w:w*2,1] = sax_w_3c_n[6]; I[h*2:h*3,w:w*2,2] = sax_w_4c_n[6]
    I[h*2:h*3,w*2:w*3,0] = sax_w_2c_n[7]; I[h*2:h*3,w*2:w*3,1] = sax_w_3c_n[7]; I[h*2:h*3,w*2:w*3,2] = sax_w_4c_n[7]
    I[h*2:h*3,w*3:w*4,0] = sax_w_2c_n[8]; I[h*2:h*3,w*3:w*4,1] = sax_w_3c_n[8]; I[h*2:h*3,w*3:w*4,2] = sax_w_4c_n[8]

    # save
    Image.fromarray((I * 255).astype('uint8')).save(save_path)


# main function for image

# get batch selection
batch_select_excel_file = os.path.join(cg.main_data_dir,'patient_plane_batch_selection.xlsx')
if os.path.isfile(batch_select_excel_file) == 1:
    print('yes there is batch selection file')
    batch_pre_select = 1
    csv_file = pd.read_excel(batch_select_excel_file)
else:
    batch_pre_select = 0
    batch_pick = [3,4,4,4,4]

# Get patient_list
patient_list = ff.find_all_target_files(['*/*'],cg.save_dir)
batch_seg = 3

for i in range(0,len(patient_list)):
    p = patient_list[i]
    patient_class = os.path.basename(os.path.dirname(p))
    patient_id = os.path.basename(p)
    print(patient_class,patient_id)

    zoom_factor_lax = 1.1
    zoom_factor_save = 1.25
    
    if batch_pre_select == 1:
        case = csv_file[(csv_file['Patient_Class'] == patient_class) & (csv_file['Patient_ID'] == patient_id)]
        assert case.shape[0] == 1
        if case.iloc[0]['2C'] == 'x' and case.iloc[0]['SAX'] == 'x':
            print('exclude')
            continue
        else:
            batch_pick = [int(case.iloc[0]['2C']),int(case.iloc[0]['3C']),int(case.iloc[0]['4C']),int(case.iloc[0]['SAX'])]
            zoom_factor_lax = float(case.iloc[0]['zoom_factor_lax'])
            zoom_factor_sax = float(case.iloc[0]['zoom_factor_sax'])
    print(batch_pick,zoom_factor_lax,zoom_factor_sax)

    save_folder = os.path.join(cg.final_dir,patient_class,patient_id)
    ff.make_folder([os.path.dirname(save_folder),save_folder])



    if os.path.isfile(os.path.join(save_folder,patient_id+'_planes.mp4')) == 1:
        print('already done')
        continue
        #os.remove(os.path.join(save_folder,patient_id+'_planes.mp4'))
    else:
           
        # this part just for VR data
        if os.path.isdir(os.path.join(cg.main_data_dir,'nii-images',patient_class,patient_id)) == 1:
            belong_path = cg.main_data_dir
        elif os.path.isdir(os.path.join(cg.main_data_dir,'2020_after_Junes/nii-images',patient_class,patient_id)) == 1:
            belong_path = os.path.join(cg.main_data_dir,'2020_after_Junes')
        else:
            ValueError('no this case')


        # seg_file = os.path.join(cg.save_dir,patient_class,patient_id,'seg-pred','batch_'+str(batch_pick[0])+'/pred_s_0.nii.gz')
        # seg = nib.load(seg_file)
        # seg_data = seg.get_fdata()

        #volume_dim = nib.load(os.path.join(cg.image_data_dir,patient_class,patient_id,'img-nii-1.5/0.nii.gz')).shape
        volume_dim = nib.load(os.path.join(cg.local_dir,patient_class,patient_id,'img-nii-1.5/0.nii.gz')).shape
        image_center = np.array([(volume_dim[0]-1)/2,(volume_dim[1]-1)/2,(volume_dim[-1]-1)/2]) 

        # # load vectors
        vector_2C = ff.get_predicted_vectors(os.path.join(cg.save_dir,patient_class,patient_id,'vector-pred/batch_'+str(batch_pick[0]),'pred_2C_t.npy'),os.path.join(cg.save_dir,patient_class,patient_id,'vector-pred/batch_'+str(batch_pick[0]),'pred_2C_r.npy'),scale, image_center)
        vector_3C = ff.get_predicted_vectors(os.path.join(cg.save_dir,patient_class,patient_id,'vector-pred/batch_'+str(batch_pick[1]),'pred_3C_t.npy'),os.path.join(cg.save_dir,patient_class,patient_id,'vector-pred/batch_'+str(batch_pick[1]),'pred_3C_r.npy'),scale, image_center)
        vector_4C = ff.get_predicted_vectors(os.path.join(cg.save_dir,patient_class,patient_id,'vector-pred/batch_'+str(batch_pick[2]),'pred_4C_t.npy'),os.path.join(cg.save_dir,patient_class,patient_id, 'vector-pred/batch_'+str(batch_pick[2]),'pred_4C_r.npy'),scale, image_center)
        vector_SA = ff.get_predicted_vectors(os.path.join(cg.save_dir,patient_class,patient_id,'vector-pred/batch_'+str(batch_pick[3]),'pred_BASAL_t.npy'),os.path.join(cg.save_dir,patient_class,patient_id,'vector-pred/batch_'+str(batch_pick[3]),'pred_BASAL_r.npy'),scale, image_center)
        # vector_2C = ff.get_ground_truth_vectors(os.path.join(cg.save_dir,patient_class,patient_id,'vector-pred-high-res-0.625','pred_2C.npy'))
        # vector_3C = ff.get_ground_truth_vectors(os.path.join(cg.save_dir,patient_class,patient_id,'vector-pred-high-res-0.625','pred_3C.npy'))
        # vector_4C = ff.get_ground_truth_vectors(os.path.join(cg.save_dir,patient_class,patient_id,'vector-pred-high-res-0.625','pred_4C.npy'))
        # vector_SA = ff.get_ground_truth_vectors(os.path.join(cg.save_dir,patient_class,patient_id,'vector-pred-high-res-0.625','pred_BASAL.npy'))


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


    

  
    
