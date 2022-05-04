## Define which GPU
export CUDA_VISIBLE_DEVICES="1"

## define num of classes for segmentation. 4 = LV, LA , LVOT and background
export CG_NUM_CLASSES=4 
export CG_RELABEL_LVOT=1 # DL uses class 3 for LVOT while in manual annotation we use class 4 for LVOT

## define the pixel dimension and image dimension for low resolution and high resolution planes
export CG_LOW_RES_SPACING=1.5
export CG_LOW_RES_DIMENSION=160 # plane dim = [160,160]
export CG_HIGH_RES_SPACING=0.625
export CG_HIGH_RES_DIMENSION=480 # plane dim = [480,480]

## define cropeed image size
export CG_CROP_X=160
export CG_CROP_Y=160
export CG_CROP_Z=96


## define some U-Net parameters
export CG_CONV_DEPTH_MULTIPLIER=1
export CG_FEATURE_DEPTH=8


# folders
export CG_MAIN_DATA_DIR="/Data/McVeighLabSuper/wip/zhennong/"
export CG_IMAGE_DATA_DIR=${CG_MAIN_DATA_DIR}nii-images/
export CG_MODEL_DIR="/Data/McVeighLabSuper/projects/Zhennong/AI/CNN/all-classes-all-phases-data-1.5/" 
export CG_SAVE_DIR="/Data/McVeighLabSuper/wip/zhennong/3D_UNet_prediction/"
export CG_LOCAL_DIR="/Data/local_storage/Zhennong/VR_Data_0.625/"





