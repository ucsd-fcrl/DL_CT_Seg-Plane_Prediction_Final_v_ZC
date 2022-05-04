#!/usr/bin/env bash
# run in docker c3d

# Get a list of patients.
patients=(/Data/ContijochLab/workspaces/zhennong/AUH_data/nii-images/*/*/ )
#patients+=(/Data/ContijochLab/workspaces/zhennong/Cases_for_Cardiowise_Elliot/nii-images/Normal/*/ )

img_or_seg=1 # 1 is image, 0 is seg

if ((${img_or_seg} == 1))
then
img_folder="img-nii"
else
img_folder="seg-nii"
fi

for p in ${patients[*]};
do
    if ! [ -d ${p}${img_folder} ] ||  ! [ "$(ls -A  ${p}${img_folder})" ];then
        echo "no image/seg"
        continue
    fi

    # find out the patient id
    patient_id=$(basename ${p})
    echo ${patient_id}

    # set output folder
    if ((${img_or_seg} == 1));then
    o_dir=${p}img-nii-0.625
    else
    o_dir=${p}seg-nii-1.5
    fi
    mkdir -p ${o_dir}
    

    # find all the data with original resolution
    IMGS=(${p}${img_folder}/*.nii.gz)
    for i in $(seq 0 $(( ${#IMGS[*]} - 1 )));
    do
        i_file=${IMGS[${i}]}
        o_file=${o_dir}/$(basename ${i_file})
        echo ${o_file}
        if [ -f ${o_file} ];then
            echo "already done this file"
            continue
        else
            if ((${img_or_seg} == 1))
            then
                c3d ${i_file} -interpolation Cubic -resample-mm 0.625x0.625x0.625mm -o ${o_file}
            else
                c3d ${i_file} -interpolation NearestNeighbor -resample-mm 1.5x1.5x1.5mm -o ${o_file}
            fi
        fi      
    done
done 


