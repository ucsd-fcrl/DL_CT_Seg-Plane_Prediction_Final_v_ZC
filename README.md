# Use Trained DL Model to Predict Segmentation and Cardiac Imaging Planes.

This is the GitHub repo based on a published paper: <br />
Automated Cardiac Volume Assessment and Cardiac Long- and Short-Axis Imaging Plane Prediction from ECG-gated CT Volumes Enabled By Deep Learning.<br />
Authors: Zhennong Chen, Davis Vigneault, Marzia Rogolli, Francisco Contijoch<br />
Citation: Zhennong Chen, Marzia Rigolli, Davis Marc Vigneault, Seth Kligerman, Lewis Hahn, Anna Narezkina, Amanda Craine, Katherine Lowe, Francisco Contijoch, Automated cardiac volume assessment and cardiac long- and short-axis imaging plane prediction from electrocardiogram-gated computed tomography volumes enabled by deep learning, European Heart Journal - Digital Health, Volume 2, Issue 2, June 2021, Pages 311–322, https://doi.org/10.1093/ehjdh/ztab033

Check how to train a DL model to do segmentation + imaging plane prediction here: https://github.com/ucsd-fcrl/AI_chamber_segmentation_plane_re-slicing

## Description
We have developed a DL model to provide automatic, accurate and fast chamber segmentation (Left ventricle and Left atrium) + cardiac imaging planes re-slicing (two-chamber, three-chamber, four-chamber planes + a short-axis stack) from cardiac CT images. 

The purpose of this GitHub repo is to use trained DL models to predict segmentation and planes on *new* CT cases.

This repo has two sets of scripts: **Main and Premier**.<br />
- **Main**: because the input of DL model has to be the under-sampled CT volumes (pixel_dim = 1.5mm), This set of script "Main" can return you the segmentations and planes in *low resolution*.<br />
    - You may have multiple trained models (trained on different dataset) acting like multiple human experts. Thus, the other important purpose of "Main" is to let you select which model gives you the adequate results.<br />
- **Premier**: turn the low resolution DL outputs into high resolution (pixel_dim = 0.625)


