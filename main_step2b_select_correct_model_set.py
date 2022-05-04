# this step is optional. It can give you better DL product.

'''
-  recall, we have 5 sets of trained model acting like 5 experienced human experts
-  after running main step 1 and 2, you have obtained the segmentation and plane movies for every model set
-  The performance may not always be perfect for every model set (some experts will make mistakes)

-  Thus, you want to visually investigate the predicted planes and segmentation made by each model set
    - usually no need to investigate segmentation due to consistently high performance
-  Then, you pick which model set give you the adequate 2CH, 3CH, 4CH and SAX plane correspondingly.
-  You can also adjust the zoom_factor for LAX and SAX plane if the LV looks too small/large.
-  Finally, You put your model_set selection along with the adjusted zoom factors into a spreadsheet.

-  an example spreadsheet can be found in example_files/model_set_selection.xlsx
'''

# prepare a spreadsheet
import function_list as ff
import os
import numpy as np
import supplement
import pandas as pd
cg = supplement.Experiment()

patient_list = ff.find_all_target_files(['Abnormal/CVC1803*'],os.path.join(cg.save_dir,'DL_prediction_low_res'))

df = []
for p in patient_list:
    patient_class = os.path.basename(os.path.dirname(p))
    patient_id = os.path.basename(p)

    df.append([patient_class, patient_id, 3, '' ,'' ,'' ,'' ,1.2 ,1.4])

df = pd.DataFrame(df,columns = ['Patient_Class', 'Patient_ID','seg (default = 3)', '2CH', '3CH', '4CH', 'SAX', 
                                'Zoom_LAX (default = 1.2)', 'Zoom_SAX (default = 1.4)'])
df.to_excel(os.path.join(cg.save_dir, 'model_set_selection_example.xlsx'), index = False)