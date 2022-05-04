#!/usr/bin/env python

## 
# this script can do file copy and removal
##

import os
import numpy as np
import function_list as ff
import shutil
import pandas as pd
import supplement

np.random.seed(0)
cg = supplement.Experiment()

# # delete
# path1 = "/Data/McVeighLabSuper/wip/zhennong/"
# path2 = "/Data/McVeighLabSuper/wip/zhennong/2020_after_Junes/"

# file_path = ff.find_all_target_files(['predicted_planes/*/*/planes_pred_low_res/*/*.png','predicted_planes/*/*/slice_num*'],cg.main_data_dir)
# print(len(file_path),file_path)
# for f in file_path:
#   os.remove(f)
    


# file transfer 
# patient_list = ff.find_all_target_files(['*/*'],os.path.join(cg.main_data_dir,'Plane_movies'))
# for p in patient_list:
#     patient_id = os.path.basename(p)
#     patient_class = os.path.basename(os.path.dirname(p))
#     ff.make_folder([os.path.join(cg.main_data_dir,'Plane_movies_only_movies',patient_class)])
#     ff.make_folder([os.path.join(cg.main_data_dir,'Plane_movies_only_movies',patient_class,patient_id)])
#     print(patient_class,patient_id)


#     movie_lists = ff.find_all_target_files(['*.mp4'],p)
#     assert len(movie_lists) == 2
#     for m in movie_lists:
#         if os.path.isfile(os.path.join(cg.main_data_dir,'Plane_movies_only_movies',patient_class,patient_id,os.path.basename(m))) ==0:
#             shutil.copy(m,os.path.join(cg.main_data_dir,'Plane_movies_only_movies',patient_class,patient_id,os.path.basename(m)))



#############################################  
# file transfer from octomore
# folder = ff.find_all_target_files(['Resample_MPR_2020_after_Junes'],'/Data/local_storage/Zhennong/')
# print(folder)
# for f in folder:
#     shutil.copytree(f,os.path.join("/Data/McVeighLabSuper/wip/zhennong/2020_after_Junes",'Resample_MPR'))

# # compress
# patient_list = ff.get_patient_list_from_csv(os.path.join(cg.spreadsheet_dir,'Final_patient_list_include.csv'))
# print(len(patient_list))
# for p in patient_list:
#     print(p[0],p[1])
#     f1 = os.path.join(cg.seg_data_dir,p[0],p[1],'seg-nii-1.5-upsample-retouch-adapted-LV')
#     f2 = os.path.join(cg.seg_data_dir,p[0],p[1],'seg-nii-1.5-upsample-retouch-adapted')
#     shutil.make_archive(os.path.join(cg.seg_data_dir,p[0],p[1],'seg-nii-1.5-upsample-retouch-adapted-LV'),'zip',f1)
#     shutil.make_archive(os.path.join(cg.seg_data_dir,p[0],p[1],'seg-nii-1.5-upsample-retouch-adapted'),'zip',f2)
#     shutil.rmtree(f1)
#     shutil.rmtree(f2)

# transfer file to octomore
# patient_list = ff.find_all_target_files(['*/*'],cg.image_data_dir)

excel_file = os.path.join('/Data/ContijochLab/projects/ct_gls','Andy_Khan_WMA_scores.xlsx')
excel_data = pd.read_excel(excel_file)
patient_list= []
data_path1 = cg.image_data_dir
data_path2 = os.path.join(cg.main_data_dir,'2020_after_Junes/nii-images')
for i in range(0,excel_data.shape[0]):
  case = excel_data.iloc[i]
  if os.path.isdir(os.path.join(data_path1,case['Patient_Class'],case['Patient_ID'])) == 1:
    belong_path = os.path.dirname(os.path.dirname(data_path1))
  elif os.path.isdir(os.path.join(data_path2,case['Patient_Class'],case['Patient_ID'])) == 1:
    belong_path = os.path.dirname(data_path2)
  else:
    ValueError('wrong!')
  patient_list.append([case['Patient_Class'],case['Patient_ID'],belong_path])
print(patient_list)

ff.make_folder([cg.local_dir])

for p in patient_list:
    # patient_id = os.path.basename(p)
    # patient_class = os.path.basename(os.path.dirname(p))
    patient_class = p[0]; patient_id = p[1]
    

    # image_file = ff.find_all_target_files(['img-nii-1.5/0.nii.gz'],p)
    image_file = ff.find_all_target_files(['img-nii-1.5/0.nii.gz'],os.path.join(p[2],'nii-images',p[0],p[1]))
    assert len(image_file) == 1

    if os.path.isdir(os.path.join(cg.local_dir,patient_class,patient_id,'img-nii-1.5')) == 0:
        print(patient_class,patient_id)
        ff.make_folder([os.path.join(cg.local_dir,patient_class),os.path.join(cg.local_dir,patient_class,patient_id),os.path.join(cg.local_dir,patient_class,patient_id,'img-nii-1.5')])
        shutil.copy(image_file[0],os.path.join(cg.local_dir,patient_class,patient_id,'img-nii-1.5',os.path.basename(image_file[0])))

# make a spreadsheet for patients
# patient_list = ff.find_all_target_files(['*/*'],os.path.join(cg.main_data_dir,'predicted_planes'))
# result = []
# for p in patient_list:
#     patient_id = os.path.basename(p)
#     patient_class = os.path.basename(os.path.dirname(p))
#     print(patient_class,patient_id)

#     result.append([patient_class,patient_id])

# df = pd.DataFrame(result,columns=['Patient_Class','Patient_ID'])
# df.to_excel(os.path.join(cg.main_data_dir,'patient_plane_batch_selection.xlsx'),index=False)
