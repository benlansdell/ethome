#!/bin/sh
conda activate ml_mabe

# Initial pose extraction from DLC
python extract_pose_from_dlc.py

# Format data
python 0_basic_data_formatting.py 

# Train DL 1D CNN model
CUDA_VISIBLE_DEVICES=1 python 1_deep_learning_stacking.py

# Create features for model stacking
python 2_feature_engineering_stacking.py 

# Train final ML model (XGB) with these features
python 3_machine_learning_cv.py mars_distr_stacked_w_1dcnn xgb

#OR: if this is an inference run
#python 4_inference.py [ARGS]

python postanalysis_videocreation.py