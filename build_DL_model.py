# class to build our DL model architecture and load trained model weights

import numpy as np
import nibabel as nb
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.models import Model
from keras.layers import Input, \
                         Conv1D, Conv2D, Conv3D, \
                         MaxPooling1D, MaxPooling2D, MaxPooling3D, \
                         UpSampling1D, UpSampling2D, UpSampling3D, \
                         Reshape, Flatten, Dense
from keras.initializers import Orthogonal
from keras.regularizers import l2

import supplement
import supplement.utils as ut
import dvpy as dv
import dvpy.tf
import function_list as ff
cg = supplement.Experiment()


class Build_Model():

    def __init__(self):
        self.build = self.build_model()
        self.model_list = self.get_model_list()


    def build_model(self):
        shape = cg.dim + (1,)
        model_inputs = [Input(shape)]
        model_outputs=[]
        ds_layer, _, unet_output = dvpy.tf.get_unet(cg.dim,cg.num_classes,cg.conv_depth,layer_name='unet',
                                            dimension =len(cg.dim),unet_depth = cg.unet_depth,)(model_inputs[0])
        ds_flat = Flatten()(ds_layer)
        Loc= Dense(384,kernel_initializer=Orthogonal(gain=1.0),kernel_regularizer = l2(1e-4),
                                            activation='relu',name='Loc')(ds_flat)
        translation = Dense(3,kernel_initializer=Orthogonal(gain=1e-1),kernel_regularizer = l2(1e-4),
                                            name ='t')(Loc)
        x_direction = Dense(3,kernel_initializer=Orthogonal(gain=1e-1),kernel_regularizer = l2(1e-4),
                                            name ='x')(Loc)
        y_direction = Dense(3,kernel_initializer=Orthogonal(gain=1e-1),kernel_regularizer = l2(1e-4),
                                            name ='y')(Loc)
        model_outputs += [unet_output]
        model_outputs += [translation]
        model_outputs += [x_direction]
        model_outputs += [y_direction]
        model = Model(inputs = model_inputs,outputs = model_outputs)

        return model

    def get_model_list(self):
        # for each task, there are 5 trained models named batch 0 - batch 4 (like 5 trained human experts)
        # model to do chamber segmentation
        model_s = ['*batch0/model-U2_batch0_s-050-*', 
                '*batch1/model-U2_batch1_s2-053-*',
                '*batch2/model-U2_batch2_s-058-*',
                '*batch3/model-U2_batch3_s-059-*',
                '*batch4/model-U2_batch4_s-056-*']

        # model to predict translation vector of 2-chamber plane
        model_2C_t = ['*batch0/model-U2_batch0_2C_t3-005-*', 
                    '*batch1/model-U2_batch1_2C_t-026-*',
                    '*batch2/model-U2_batch2_2C_t-033-*',
                    '*batch3/model-U2_batch3_2C_t-036-*',
                    '*batch4/model-U2_batch4_2C_t-020-*']

        # model to predict directional vector of 2-chamber plane
        model_2C_r = ['*batch0/model-U2_batch0_2C_r-032-*', 
                    '*batch1/model-U2_batch1_2C_r-040-*',
                    '*batch2/model-U2_batch2_2C_r2-035-*',
                    '*batch3/model-U2_batch3_2C_r-035-*',
                    '*batch4/model-U2_batch4_2C_r2-034-*']

        # model to predict translation vector of 3-chamber plane
        model_3C_t = ['*batch0/model-U2_batch0_3C_t-037-*', 
                    '*batch1/model-U2_batch1_3C_t-039-*',
                    '*batch2/model-U2_batch2_3C_t-040-*',
                    '*batch3/model-U2_batch3_3C_t-034-*',
                    '*batch4/model-U2_batch4_3C_t-036-*']

        # model to predict directional vector of 3-chamber plane
        model_3C_r = ['*batch0/model-U2_batch0_3C_r-040-*', 
                    '*batch1/model-U2_batch1_3C_r2-035-*',
                    '*batch2/model-U2_batch2_3C_r-031-*',
                    '*batch3/model-U2_batch3_3C_r-036-*',
                    '*batch4/model-U2_batch4_3C_r-032-*']

        # model to predict translation vector of 4-chamber plane
        model_4C_t = ['*batch0/model-U2_batch0_4C_t-032-*', 
                    '*batch1/model-U2_batch1_4C_t-036-*',
                    '*batch2/model-U2_batch2_4C_t-039-*',
                    '*batch3/model-U2_batch3_4C_t-032-*',
                    '*batch4/model-U2_batch4_4C_t-031-*']

        # model to predict directional vector of 4-chamber plane
        model_4C_r = ['*batch0/model-U2_batch0_4C_r-018-*', 
                    '*batch1/model-U2_batch1_4C_r-017-*',
                    '*batch2/model-U2_batch2_4C_r2-031-*',
                    '*batch3/model-U2_batch3_4C_r-039-*',
                    '*batch4/model-U2_batch4_4C_r-040-*']

        # model to predict translation vector of short-axis plane
        model_BASAL_t = ['*batch0/model-U2_batch0_BASAL_t2-026-*', 
                    '*batch1/model-U2_batch1_BASAL_t-030-*',
                    '*batch2/model-U2_batch2_BASAL_t-032-*',
                    '*batch3/model-U2_batch3_BASAL_t2-017-*',
                    '*batch4/model-U2_batch4_BASAL_t-038-*']

        # model to predict directional vector of short-axis plane
        model_BASAL_r = ['*batch0/model-U2_batch0_BASAL_r-035-*', 
                    '*batch1/model-U2_batch1_BASAL_r-018-*',
                    '*batch2/model-U2_batch2_BASAL_r-039-*',
                    '*batch3/model-U2_batch3_BASAL_r-040-*',
                    '*batch4/model-U2_batch4_BASAL_r-025-*']

        MODEL_list = [model_s,model_2C_t,model_2C_r,model_3C_t,model_3C_r,model_4C_t,model_4C_r,model_BASAL_t,model_BASAL_r]

        return MODEL_list

    def select_one_model_set(self, i):
        MODEL_list = self.model_list
        model_set = []
        for j in range(0,len(MODEL_list)):
            model_set.append(MODEL_list[j][i])
        return model_set

    def image_generator(self):
        return dv.tf.ImageDataGenerator(3,input_layer_names=['input_1'],output_layer_names=['unet','t','x','y'],)

    def generator_parameters(self,task_name):
        if task_name == 's':
            view = '2C'
            return [view,'']
        else:
            view = task_name.split('_')[0]
            vector = task_name.split('_')[1]
            return [view,vector]

    def save_segmentation(self,img,seg_pred,save_path):
        u_gt_nii = nb.load(img) # get affine matrix from image
        u_pred = np.argmax(seg_pred[0], axis = -1).astype(np.uint8)
        u_pred = dv.crop_or_pad(u_pred, u_gt_nii.get_data().shape)
        u_pred[u_pred == 3] = 4 # relabel LVOT
        u_pred = nb.Nifti1Image(u_pred, u_gt_nii.affine)
        nb.save(u_pred, save_path)

    def save_vector(self,t_pred,x_pred,y_pred,save_path):
        x_n = ff.normalize(x_pred)
        y_n = ff.normalize(y_pred)
        matrix = np.concatenate((t_pred.reshape(1,3),x_n.reshape(1,3),y_n.reshape(1,3)))
        np.save(save_path,matrix)


    




