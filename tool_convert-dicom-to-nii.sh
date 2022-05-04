#!/usr/bin/env bash

# run in terminal of your own laptop, use this one for conversion

##############
## Settings ##
##############

set -o nounset
set -o errexit
set -o pipefail


# define the folder where dcm2niix function is saved
dcm2niix_fld="/Users/zhennongchen/Documents/GitHub/AI_reslice_orthogonal_view/dcm2niix_11-Apr-2019/"


# define patient lists (the folder where )
#readonly PATIENTS=(/Volumes/McVeighLab/wip/ucsd_lvad/to_be_reviewed/CVC1709271428/)
#readonly PATIENTS=(/Volumes/McVeighLab/projects/Zhennong/AUH_patients/Original_dicoms/1_post/)
#readonly PATIENTS=(/Users/zhennongchen/Documents/Zhennong_CT_Data/AUH/1/1_post/ )
#readonly PATIENTS=(/Users/zhennongchen/Downloads/test/tt/ )    

# define image folder
img_fld="img_sorted"

for p in ${PATIENTS[*]};
do

  echo ${p}
  
  if [ -d ${p}${img_fld} ];
  then

  #output="/Volumes/McVeighLab/projects/Zhennong/AUH_patients/nii_images" # different directory from raw data
  #output="/Users/zhennongchen/Documents/Zhennong_CT_Data/AUH/nii_images"
  output="/Users/zhennongchen/Downloads/test/nii_images"
  mkdir -p ${output}/$(basename ${p})/
  mkdir -p ${output}/$(basename ${p})/img-nii/

  #check whether already converted
  nii_folder=${output}/$(basename ${p})/img-nii
  if [ -d ${nii_folder} ] && [ "$(ls -A  ${nii_folder})" ];then
    echo "already done"
    continue
  fi

  IMGS=(${p}${img_fld}/*/)
  
  for i in $(seq 0 $(( ${#IMGS[*]} - 1 )));
      do

      echo ${IMGS[${i}]}
      if [ "$(ls -A ${IMGS[${i}]})" ]; then # dcm2niix doesn't work (ignore image), remove -i y
        
        ${dcm2niix_fld}dcm2niix -i y -m y -b n -o "${output}/$(basename ${p})/img-nii/" -f "${i}" -9 -z y "${IMGS[${i}]}"
      
      else
        echo "${IMGS[${i}]} is emtpy; Skipping"
        continue
      fi
      
    done

  else
    echo "${p} missing dicom image folder"
    continue
    
  fi
done
