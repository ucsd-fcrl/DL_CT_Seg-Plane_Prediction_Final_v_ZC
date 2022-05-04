#!/usr/bin/env bash

# this script upsample planes (in low res) nii file into high resolution (default pixel_dim = 0.625)
# run it by ./premier_step2_upsample_plane.sh

####### define patient list
PATIENT=(/Data/McVeighLabSuper/wip/zhennong/3D_UNet_prediction/DL_prediction_low_res/Abnormal/CVC1803*/)


####### define folder
IMAGE_DIR='/Data/McVeighLabSuper/wip/zhennong/nii-images/'
SAVE_DIR='/Data/McVeighLab/wip/zhennong/3D_UNet_prediction/DL_prediction_high_res/'
mkdir -p ${SAVE_DIR}

# Folder where the low-res plane nii sits 
mpr_fld=planes_pred_low_res_nii
# Folder where the high-res volumes to be resliced sit
input_fld=img-nii-0.625
# Folder where you want to save the resampled results
output_fld=planes_pred_high_res_0.625_nii




# main script (usually no need to change)
set -o nounset
set -o errexit
set -o pipefail

out_size=480;
out_spac=0.625;
out_value=-2047;

dv_utils_fld="/Experiment/Documents/Repos/dv-commandline-utils/bin/"

SLICE[0]=2C
SLICE[1]=3C
SLICE[2]=4C
SLICE[3]=BASAL

for p in ${PATIENT[*]};
do
    patient_id=$(basename ${p})
    patient_class=$(basename $(dirname ${p}))
    echo ${p}${mpr_fld}

    if  [ ! -d ${p}${mpr_fld} ];then
        echo "no mpr image"
        continue
    fi

    save_folder=${SAVE_DIR}${patient_class}/${patient_id}/${output_fld}
    mkdir -p ${SAVE_DIR}${patient_class}
    mkdir -p ${SAVE_DIR}${patient_class}/${patient_id}
    mkdir -p ${save_folder}

    IMGS=(${IMAGE_DIR}${patient_class}/${patient_id}/${input_fld}/0.nii.gz) 
    echo ${IMGS[0]}

    for i in $(seq 0 $(( ${#IMGS[*]} - 1 )));
    do
        for s in ${SLICE[*]};
        do

            REF=( ${p}${mpr_fld}/pred_${s}.nii.gz)

            img_name=$(basename ${IMGS[${i}]}  .nii.gz);
            output_file=${save_folder}/$(basename ${REF[0]})
        
            echo ${IMGS[${i}]}
            echo ${REF[0]}
            echo $output_file

            if [ -f ${output_file} ];then
                echo "already done"
                continue
            else
                ${dv_utils_fld}dv-resample-from-reference --input-image ${IMGS[${i}]} --reference-image ${REF[0]} --output-image $output_file --output-size $out_size --output-spacing $out_spac --outside-value $out_value
            fi
        done
    done
done