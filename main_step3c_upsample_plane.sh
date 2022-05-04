#!/usr/bin/env bash
########################################
# run at mcveighlab-octomore
# Currently can only run in octomore terminal, cd Developer, . .py35/bin/activate to activate the Virtual environment
# screen to have a new screen, screen -r to retrieve the previous
# ctrl +A +D to quit 
#######################################

# this script resample planes (in low res) nii file into high resolution (only works at isotropic resolution, e.g 0.625x0.625x0.625mm)

. ./main_step1_defaults.sh

set -o nounset
set -o errexit
set -o pipefail

out_size=480;
out_spac=0.625;
out_value=-2047;

dv_utils_fld="/home/cnn/Documents/Repos/dv-commandline-utils/bin/"


# patient list
PATIENT=(/media/ContijochLab/workspaces/zhennong/Cases_for_Cardiowise_Elliot/Predict/Normal/*/)
PATIENT+=(/media/ContijochLab/workspaces/zhennong/Cases_for_Cardiowise_Elliot/Predict/Abnormal/*/)
SAVE_DIR='/media/ContijochLab/workspaces/zhennong/Cases_for_Cardiowise_Elliot/Predict/'


SLICE[0]=2C
SLICE[1]=3C
SLICE[2]=4C
SLICE[3]=BASAL


# Folder where you want to put the reslice
fld_prefix=planes_pred_high_res_0.625_nii

# Folder where the volumes to be resliced reside
input_fld=img-nii-0.625

# Folder where the mpr nii sits with direction info
mpr_fld=planes_pred_low_res_nii

for p in ${PATIENT[*]};
do
    patient_id=$(basename ${p})
    patient_class=$(basename $(dirname ${p}))
    echo ${p}${mpr_fld}
    if  [ ! -d ${p}${mpr_fld} ];then
        echo "no mpr image"
        continue
    fi

    save_folder=${SAVE_DIR}${patient_class}/${patient_id}/${fld_prefix}
    #mkdir -p ${SAVE_DIR}${patient_class}
    #mkdir -p ${SAVE_DIR}${patient_class}/${patient_id}
    mkdir -p ${save_folder}

    IMGS=(/media/ContijochLab/workspaces/zhennong/Cases_for_Cardiowise_Elliot/nii-images/${patient_class}/${patient_id}/${input_fld}/0.nii.gz) 
    echo ${IMGS[0]}

    for i in $(seq 0 $(( ${#IMGS[*]} - 1 )));
    do

        # For each volume, lets look at each MPR slice

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