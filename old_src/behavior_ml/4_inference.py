import numpy as np
import os
import glob
import pandas as pd
import pickle 
import argparse
from collections import defaultdict
from joblib import load

import hashlib

from sklearn.svm import LinearSVC as SVC
from sklearn.linear_model import LogisticRegression

import xgboost as xgb
import ssm

xgb_model = 'xgb'
xgb_features = 'mars_distr_stacked_w_1dcnn'

data_dir = './data/dlc_improved/'
model_dir = './data/dlc_improved_aug31_dlcrnetms/'

reweighting = True
use_hmm = False

mapping = {
    'e3v813a-20211001T121054-121622': 'social_1',
    'e3v813a-20211001T122726-123223': 'social_2',
    'e3v813a-20211001T123846-124419': 'social_3',
    'e3v813a-20211001T125501-130007': 'social_4',
    'e3v813a-20211001T132205-132735': 'social_1_re',
    'e3v813a-20211001T133723-134310': 'social_2_re',
    'e3v813a-20211001T135129-135613': 'social_3_re',
    'e3v813a-20211001T140617-141120': 'social_4_re'
}

def infer_hmm(hmm, emissions_raw, preds_raw, C):
    emissions = np.hstack(((emissions_raw*(C-1)).astype(int), np.atleast_2d((preds_raw).astype(int)).T))
    return hmm.most_likely_states(emissions)

files_in = sorted(glob.glob(f'{data_dir}/*_improved.csv'))
fold_idx = 1

test_features = pd.read_csv(f'data/intermediate/test_features_{xgb_features}.csv')

vid_names = {hashlib.md5(fn_in.encode()).hexdigest()[:8]:fn_in for fn_in in files_in}

loaded_data = load(f'{model_dir}/final_model_fold_{fold_idx}.pkl')
model = loaded_data['model']
hmm = loaded_data['hmm']
w_star = loaded_data['w_star']

if reweighting is None:
    reweighting = loaded_data['reweighting']
if use_hmm is None:
    use_hmm = loaded_data['use_hmm']

with open(f'data/intermediate/test_map_features_{xgb_features}.pkl', 'rb') as handle:
    test_map = pickle.load(handle)

inverse_test_map = {test_map[k]:k for k in test_map.keys()}
X = test_features.drop(columns = ['seq_id'])
groups = test_features['seq_id']

X_test = X
groups_test = groups

print("Predicting fit model on test data")
predict_proba = model.predict_proba(X_test)
predict = model.predict(X_test)

if reweighting:
    print("Reweighting for optimal F1 score")
    test_pred_probs_reweighted = predict_proba*w_star
    reweighted_predictions = np.argmax(test_pred_probs_reweighted, axis = -1)

    if use_hmm:
        print("Applying HMM to reweighted model optimal F1 score")
        C = 11
        final_predictions = infer_hmm(hmm, np.array(test_pred_probs_reweighted), np.array(reweighted_predictions), C)
    else:
        final_predictions = reweighted_predictions
else:
    if use_hmm:
        print("Applying HMM to ML model output")
        C = 11
        final_predictions = infer_hmm(hmm, np.array(predict_proba), np.array(predict), C)
    else:
        final_predictions = predict

final_predictions[final_predictions == 0] = 1
final_predictions[final_predictions == 2] = 1

print("Preparing submission")
fn_out = f"{data_dir}/submission_{xgb_features}_ml_{xgb_model}_paramset_default_hmm.npy"
submission = defaultdict(list)
for idx in range(len(final_predictions)):
    submission[test_map[groups_test.iloc[idx]]].append(final_predictions[idx])
np.save(fn_out, submission)

#How many predictions of interact class?
print("No behavior label count:", np.sum(final_predictions == 3))
print("Interaction count:", np.sum(final_predictions != 3))

#Split into each file
final_predictions_df = pd.DataFrame(groups)
final_predictions_df['preds'] = final_predictions

#Prediction counts
interaction_times = final_predictions_df.value_counts().reset_index().sort_values('seq_id')
interaction_times['seq_hash'] = interaction_times['seq_id'].apply(lambda x: test_map[x])
interaction_times['vid_name'] = interaction_times['seq_hash'].apply(lambda x: vid_names[x].split('/')[3].split('DLC')[0])
interaction_times['vid_title'] = interaction_times['vid_name'].apply(lambda x: mapping[x])
interaction_times = interaction_times.sort_values('vid_title')
interaction_times