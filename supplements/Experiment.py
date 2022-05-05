# System
import os

class Experiment():

  def __init__(self):

    # Number of Classes (Including Background)
    self.num_classes = int(os.environ['CG_NUM_CLASSES'])
    # Whether relabel of LVOT is necessary
    if int(os.environ['CG_RELABEL_LVOT']) == 1:
      self.relabel_LVOT = True
    else:
      self.relabel_LVOT = False
 
    # SPACING and Dimension of DL product
    self.low_res_spacing = float(os.environ['CG_LOW_RES_SPACING'])
    self.low_res_dim = [int(os.environ['CG_LOW_RES_DIMENSION']), int(os.environ['CG_LOW_RES_DIMENSION']), 1]
    self.high_res_spacing = float(os.environ['CG_HIGH_RES_SPACING'])
    self.high_res_dim = [int(os.environ['CG_HIGH_RES_DIMENSION']) ,int(os.environ['CG_HIGH_RES_DIMENSION']) , 1]

    # Dimension of padded input, for 3DUnet
    self.dim = (int(os.environ['CG_CROP_X']), int(os.environ['CG_CROP_Y']), int(os.environ['CG_CROP_Z']))
  
    # UNet Depth
    self.unet_depth = 5
  
    # Depth of convolutional feature maps
    self.conv_depth_multiplier = int(os.environ['CG_CONV_DEPTH_MULTIPLIER'])
    self.ii = int(os.environ['CG_FEATURE_DEPTH'])
    self.conv_depth = [2**(self.ii-4),2**(self.ii-3),2**(self.ii-2),2**(self.ii-1),2**(self.ii),2**(self.ii),2**(self.ii-1),2**(self.ii-2),
                      2**(self.ii-3),2**(self.ii-4),2**(self.ii-4)]
    self.conv_depth = [self.conv_depth_multiplier*x for x in self.conv_depth]


    # folder
    self.main_data_dir = os.environ['CG_MAIN_DATA_DIR']
    self.image_data_dir = os.environ['CG_IMAGE_DATA_DIR']
    self.local_dir = os.environ['CG_LOCAL_DIR']
    self.model_dir = os.environ['CG_MODEL_DIR']
    self.save_dir = os.environ['CG_SAVE_DIR']

