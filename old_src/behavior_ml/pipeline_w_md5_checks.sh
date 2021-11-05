#!/bin/sh

conda activate ml_mabe

#Initialization
#Data will be placed in ./data/

#Personal access token for gitlab.aicrowd: sXwaiXsx_3mgi4idejXS

#Change to your own key...
API_KEY="0ba231d61506b40a4ae00df011cf0cb9" aicrowd login --api-key $API_KEY
aicrowd dataset download --challenge mabe-task-1-classical-classification

mkdir data
mkdir data/intermediate
mkdir results
 
mv train.npy data/train.npy
mv test-release.npy data/test.npy
mv sample-submission.npy data/sample_submission.npy

#Step 0
# Format data
# ~5 minutes
python 0_basic_data_formatting.py 

#Step 1
# Train DL 1D CNN model
# ~90 minutes
python 1_deep_learning_stacking.py

#Step 2
# Create features for model stacking
# ~3-4 hours, note also that the test features csv file is ~100GB
python 2_feature_engineering_stacking.py 

#Step 3 
# Train final ML model (XGB) with these features
# ~20 minutes
python 3_machine_learning.py mars_distr_stacked_w_1dcnn xgb --test

#Submit:
aicrowd submission create -c mabe-task-1-classical-classification \
                    -f results/submission_mars_distr_stacked_w_1dcnn_ml_xgb_paramset_default.npy

################################
#Check the answers are the same#
################################

#Step 0 -- checks out
md5sum ./data/intermediate/test_df.csv
md5sum ../../mabe/mabetask1_ml/data/intermediate/test_df.csv

md5sum ./data/intermediate/train_df.csv
md5sum ../../mabe/mabetask1_ml/data/intermediate/train_df.csv

#Step 1 -- doesn't yet check out... reproducibility in Keras may be complicated... will leave for now.
md5sum ./data/intermediate/deep_learning_stacking_prediction_probabilities_baseline_test_run_distances.npy
md5sum /home/blansdel/projects/mabe/mabetask1/deep_learning_stacking_prediction_probabilities_baseline_test_run_distances.npy

md5sum ./data/intermediate/deep_learning_stacking_prediction_probabilities_test_baseline_test_run_distances.npy
md5sum /home/blansdel/projects/mabe/mabetask1/deep_learning_stacking_prediction_probabilities_test_baseline_test_run_distances.npy

md5sum ./data/intermediate/deep_learning_stacking_predictions_baseline_test_run_distances.npy
md5sum /home/blansdel/projects/mabe/mabetask1/deep_learning_stacking_predictions_baseline_test_run_distances.npy

#Step 2 -- doesn't check out

# Seems I didn't set the seeds on any of the ML methods... so these are unlikely to reproduce either :(

#Step 3
# Don't have these saved :( Pretty stupid...

# Instead, here are some other sanity checks:

# Check 