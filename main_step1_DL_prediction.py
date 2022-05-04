#!/usr/bin/env python

# this script uses trained deep learning (DL) model to predict:
# (1) chamber segmentation (LV+LA) 
# (2) vectors (translation vector t + directional vectors r) to reslice the cardiac imaging planes 

# the model architecture is a modified 3D-Unet, and the model input is the downsampled (pixel_dim = 1.5mm) 3D CT volume.
# the model outputs, including segmentation and imaging plane vectors, are also for low resolution (pixel_dim = 1.5mm).

# We have prepared 5 sets of models trained on 5 different sets of data. Think them as 5 different human experts doing the same tasks.
# Thus, you can choose the number of sets (like the number of experts) you want to do the job.

# System
import os

# Third Party
import numpy as np
import nibabel as nb
import supplement
import supplement.utils as ut
import function_list as ff
from build_DL_model import *
cg = supplement.Experiment()

######### Define models 
build_model = Build_Model()
model = build_model.build # model architecture
MODEL_list = build_model.model_list # pre-trained model list (in total there are 5 sets of models)

######### in total there are 5 sets of trained model numbered from 0 to 4, define which sets you want to use.
model_sets = [0,1,2,3,4] # want to use all the sets

######### in total we have 9 tasks:
# chamber segmentation + predict (a) translation vecotr "t" and (b) directional vector "r" for 4 planes (2CH, 3CH, 4CH and SAX): 1 + 4 x 2 = 9 tasks
task_list = ['s','2C_t','2C_r','3C_t','3C_r','4C_t','4C_r','BASAL_t','BASAL_r'] 
# define which tasks you want to do:
task_num_list = [0,1,2,3,4,5,6,7,8]  # want to do all of them


######### Define patient list
patient_list = ff.find_all_target_files(['Abnormal/CVC1803*'],cg.image_data_dir)
print(patient_list.shape) 

######### Define save_folder
save_folder = os.path.join(cg.save_dir,'DL_prediction_low_res')
ff.make_folder([save_folder])


# Main script (usually no need to change):
for i in model_sets:
  model_set = build_model.select_one_model_set(i)

  # do tasks one by one
  for task_num in task_num_list:
    print('current task is: ', task_list[task_num])

    [view,vector] = build_model.generator_parameters(task_list[task_num])
 
    # load saved weights
    model_files = ff.find_all_target_files([model_set[task_num]],cg.model_dir)
    assert len(model_files) == 1
    model.load_weights(model_files[0],by_name = True)
    print('finish loading saved weights: ',model_files[0])

    # predict patietns one by one
    for p in patient_list:
      patient_class = os.path.basename(os.path.dirname(p)); patient_id = os.path.basename(p)
      print(patient_class, patient_id)
      
      # if already done:
      if task_list[task_num] == 's':
        if os.path.isfile(os.path.join(save_folder,patient_class,patient_id,'seg-pred/batch_'+str(i),'pred_s_0.nii.gz')) == 1:
          print('already done segmentation')
          continue
      else:
        if os.path.isfile(os.path.join(save_folder,patient_class,patient_id,'vector-pred/batch_' + str(i),'pred_'+task_list[task_num]+'.npy')) == 1:
          print('already done ', task_list[task_num])
          continue

      # find the input images for time frames:
      if task_list[task_num] == 's':
        img_list = ff.sort_timeframe(ff.find_all_target_files(['img-nii-' + str(cg.low_res_spacing) + '/*.nii.gz'],p),2) # predict segmentation for all time frames
      else:
        img_list = ff.find_all_target_files(['img-nii-' + str(cg.low_res_spacing) + '/0.nii.gz'],p) # only need one time frame to predict planes


      for img in img_list:
        # predict:
        valgen = build_model.image_generator()
        seg_pred,t_pred,x_pred,y_pred= model.predict_generator(valgen.predict_flow(np.asarray([img]),
            batch_size = 1,
            view = view,
            relabel_LVOT = cg.relabel_LVOT,
            input_adapter = ut.in_adapt,
            output_adapter = ut.out_adapt,
            shape = cg.dim,
            input_channels = 1,
            output_channels = cg.num_classes,),
            verbose = 1,
            steps = 1,)

        # save u_net segmentation
        if task_list[task_num] == 's':
          save_path = os.path.join(save_folder,patient_class,patient_id,'seg-pred/batch_'+str(i),'pred_s_'+os.path.basename(img))
          ff.make_folder([os.path.join(save_folder,patient_class), os.path.join(save_folder,patient_class,patient_id),os.path.dirname(os.path.dirname(save_path)), os.path.dirname(save_path)])
          build_model.save_segmentation(img,seg_pred,save_path)
      
      # save vectors
        if task_list[task_num] != 's':
          save_path = os.path.join(save_folder,patient_class,patient_id,'vector-pred/batch_'+str(i),'pred_'+task_list[task_num])
          ff.make_folder([os.path.join(save_folder,patient_class), os.path.join(save_folder,patient_class,patient_id),os.path.dirname(os.path.dirname(save_path)), os.path.dirname(save_path)])
          build_model.save_vector(t_pred,x_pred,y_pred,save_path)