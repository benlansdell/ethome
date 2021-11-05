import numpy as np
import os
import glob
import pandas as pd
import pickle 
import argparse
from collections import defaultdict
from joblib import load, dump

import hashlib
import json

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

#from sklearn.svm import SVC
from sklearn.svm import LinearSVC as SVC
from sklearn.linear_model import LogisticRegression

import xgboost as xgb
import ssm

rate = 1/30

xgb_model = 'xgb'
xgb_features = 'mars_distr_stacked_w_1dcnn'

use_hmm = True
reweighting = True
modes = ['refit_model',
        'train_from_scratch_model',
        'base_model']

mode = 'train_from_scratch_model'

pilot_2_durations = np.array([360+26.00,
                        360+59.97,
                        360+19.03,
                        300+29.03,
                        360+23.00])

from lib.utils import seed_everything

seed_everything()

def logit(p):
    return np.log(p / (1 - p))

#Idea 2. HMM.
def fit_hmm(gt, emissions_raw, preds_raw, D, C):

    #Fit empirical transition matrix
    transition_matrix = np.ones((D,D))

    N = len(gt)
    gt = gt.astype(int)

    for idx in range(N):
        if idx == 0: continue
        transition_matrix[gt[idx-1], gt[idx]] += 1
        
    for j in range(D):
        transition_matrix[j] /= np.sum(transition_matrix[j])
        
    #Adding the actual predicted category from the RF model (in addition to the probabilities)
    #helped improve performance -- increase the precision a bit
    
    #Turn traces into categories
    emissions = np.hstack(((emissions_raw*(C-1)).astype(int), np.atleast_2d((preds_raw).astype(int)).T))

    #print(emissions.shape)
    
    #Fit empirical emission probabilities
    emission_dist = np.ones((D, D+1, C))
    for i in range(D):
        for j in range(D+1):
            for k in range(C):
                ct = np.sum(emissions[(gt == i),j] == k)
                emission_dist[i, j, k] = max(1, ct)
            emission_dist[i,j,:] /= np.sum(emission_dist[i,j,:])

    true_hmm = ssm.HMM(D, D+1, observations="categorical", observation_kwargs = {'C': C})

    #Set params to empirical ones
    true_hmm.transitions.params = [np.log(transition_matrix)]

    #true_hmm.init_state_distn.params stay as is (uniform)

    #Emission probs, stored as logits
    true_hmm.observations.params = logit(emission_dist)
            
    return true_hmm

def infer_hmm(hmm, emissions_raw, preds_raw, C):
    emissions = np.hstack(((emissions_raw*(C-1)).astype(int), np.atleast_2d((preds_raw).astype(int)).T))
    return hmm.most_likely_states(emissions)

def sample_prob_simplex(n=4):
    x = sorted(np.append(np.random.uniform(size = n-1), [0,1]))
    y = np.diff(np.array(x))
    return y

def optimize_weights(train_labels, train_pred_prob, N = 1000):
    f = lambda w: f1_score(train_labels, np.argmax((train_pred_prob*w), axis = -1), average = 'macro', labels = [0,1,2])

    w_star = np.ones(4)/4
    f_star = 0

    for idx in range(N):
        w = sample_prob_simplex()
        f_curr = f(w)
        if f_curr > f_star:
            w_star = w
            f_star = f_curr

    #Reweight and then apply HMM
    train_pred_probs_reweighted = train_pred_prob*w_star
    train_pred_reweighted = np.argmax(train_pred_probs_reweighted, axis = -1)

    return (w_star, f_star, train_pred_probs_reweighted, train_pred_reweighted)

def format_labels(boris_in, length):
    boris_labels = pd.read_csv(boris_in, skiprows = 15)
    boris_labels['index'] = (boris_labels.index//2)
    boris_labels = boris_labels.pivot_table(index = 'index', columns = 'Status', values = 'Time').reset_index()
    boris_labels = list(np.array(boris_labels[['START', 'STOP']]))
    boris_labels = [list(i) for i in boris_labels]
    ground_truth = np.zeros(length)
    for start, end in boris_labels:
        ground_truth[int(start/rate):int(end/rate)] = 1
    return ground_truth

boris_files = ['./data/boris/DLC1.csv', #e3v813a-20210610T120637-121213
                './data/boris/DLC2.csv', #e3v813a-20210610T121558-122141
                './data/boris/DLC3.csv', #e3v813a-20210610T122332-122642
                './data/boris/DLC4.csv', #e3v813a-20210610T122758-123309
                './data/boris/DLC5.csv', #e3v813a-20210610T123521-124106
                './data/boris/DLC_Set2_1_reoffset.csv',
                './data/boris/DLC_Set2_2_reoffset.csv', 
                './data/boris/DLC_Set2_3_reoffset.csv', 
                './data/boris/DLC_Set2_4_reoffset.csv',
                './data/boris/DLC_Set2_5_reoffset.csv']

#Load in first pilot study data lengths
boris_project_file = './data/boris/pilot_study.boris'
f = open(boris_project_file,)  
project_data = json.load(f)

lengths = {}
for idx_b, fn in enumerate(boris_files):
    name = os.path.basename(fn)[:-4]
    try:
        key = list(project_data['observations'][name]['media_info']['length'].keys())[0]
        lengths[fn] = int(project_data['observations'][name]['media_info']['length'][key]/rate)
    except KeyError:
        lengths[fn] = round(pilot_2_durations[idx_b-5]/rate)

files_in = sorted(glob.glob('./data/dlc_improved/*_improved.csv'))

hashes = [hashlib.md5(fn_in.encode()).hexdigest()[:8] for fn_in in files_in]

boris_to_file = {i:j for i,j in zip(boris_files, files_in)}
files_to_hash = {i:j for i,j in zip(files_in, hashes)}
boris_to_hash = {i:files_to_hash[boris_to_file[i]] for i in boris_files}

#Only first set
full_set = ['./data/boris/DLC1.csv', #e3v813a-20210610T122758-123309
               './data/boris/DLC2.csv', #e3v813a-20210610T120637-121213
               './data/boris/DLC4.csv', #e3v813a-20210610T121558-122141
               './data/boris/DLC3.csv', #e3v813a-20210610T122332-122642
               './data/boris/DLC5.csv'] #e3v813a-20210610T123521-124106

#All recordings
#full_set = boris_files

def main(train_set, test_set, fold_idx):

    print('Training on', train_set)
    print('Testing on', test_set)
    test_set_hash = [boris_to_hash[i] for i in test_set]
    train_set_hash = [boris_to_hash[i] for i in train_set]

    test_features = pd.read_csv(f'data/intermediate/test_features_{xgb_features}.csv')
    model = load('./results/level_1_model_xgb.joblib')
    hmm = load('./results/level_1_hmm_model.joblib')
    with open(f'data/intermediate/test_map_features_{xgb_features}.pkl', 'rb') as handle:
        test_map = pickle.load(handle)

    inverse_test_map = {test_map[k]:k for k in test_map.keys()}
    X = test_features.drop(columns = ['seq_id'])
    groups = test_features['seq_id']

    #Load in features from BORIS
    X_test = X[groups == inverse_test_map[test_set_hash[0]]]
    groups_test = groups[groups == inverse_test_map[test_set_hash[0]]]

    #Use the rest as training to tweak the final model...
    y_train = []
    X_train = []
    for idx in range(len(train_set)):
        fn = train_set[idx]
        group = inverse_test_map[train_set_hash[idx]]
        X_train.append(X[groups == group])
        y_train += list(format_labels(fn, lengths[fn]))

    X_train = pd.concat(X_train, axis = 0)
    y_train = np.array(y_train)
    y_train[y_train == 0] = 3
    y_train[0] = 0
    y_train[1] = 2

    cols_when_model_builds = model.get_booster().feature_names
    X_train_ = X_train[cols_when_model_builds]

    #Do a final training on this set of data, too
    if mode == 'refit_model':
        #model.fit(X_train_, y_train, xgb_model = model)
        print("Refitting XGB w new labels")
        model.fit(X_train_, y_train)
    elif mode == 'train_from_scratch_model':
        print("Fitting XGB")
        #model = LogisticRegression()
        model = xgb.XGBClassifier()
        model.fit(X_train_, y_train)
    else:
        pass

    print("Predicting fit model on test data")
    predict_proba = model.predict_proba(X_test)
    predict = model.predict(X_test)

    predict_proba_train = model.predict_proba(X_train)
    predict_train = model.predict(X_train)

    models_to_save = {'model': model, 'hmm': None, 'w_star': None, 'use_hmm': use_hmm, 'reweighting': reweighting}

    if reweighting:
        print("Reweighting for optimal F1 score")
        (w_star, _, train_pred_probs_reweighted, train_pred_reweighted) = \
                                        optimize_weights(y_train, predict_proba_train)

        test_pred_probs_reweighted = predict_proba*w_star
        reweighted_predictions = np.argmax(test_pred_probs_reweighted, axis = -1)

        models_to_save['w_star'] = w_star

        if use_hmm:
            print("Applying HMM to reweighted model optimal F1 score")
            D = predict_proba_train.shape[1]
            C = 11
            hmm_reweighted = fit_hmm(np.array(y_train), np.array(train_pred_probs_reweighted), np.array(train_pred_reweighted), D, C)
            models_to_save['hmm'] = hmm_reweighted
            final_predictions = infer_hmm(hmm_reweighted, np.array(test_pred_probs_reweighted), np.array(reweighted_predictions), C)
        else:
            final_predictions = reweighted_predictions
    else:
        if use_hmm:
            print("Applying HMM to ML model output")
            D = predict_proba_train.shape[1]
            C = 11
            hmm = fit_hmm(np.array(y_train), np.array(predict_proba_train), np.array(predict_train), D, C)
            models_to_save['hmm'] = hmm
            final_predictions = infer_hmm(hmm, np.array(predict_proba), np.array(predict), C)
        else:
            final_predictions = predict

    final_predictions[final_predictions == 0] = 1
    final_predictions[final_predictions == 2] = 1
    
    true_labels = format_labels(test_set[0], lengths[test_set[0]])
    true_labels[true_labels == 0] = 3

    final_model_out = f'./data/dlc_improved/final_model_fold_{fold_idx}.pkl'
    dump(models_to_save, final_model_out)

    print("Preparing submission")
    fn_out = f"results/submission_{xgb_features}_ml_{xgb_model}_paramset_default_hmm.npy"
    submission = defaultdict(list)
    for idx in range(len(final_predictions)):
        submission[test_map[groups_test.iloc[idx]]].append(final_predictions[idx])
    np.save(fn_out, submission)

    #How many predictions of interact class?
    print("No behavior label count:", np.sum(final_predictions == 3))
    print("Interaction count:", np.sum(final_predictions != 3))
    print("Accuracy:", accuracy_score(true_labels, final_predictions))

    print("F1 score:", f1_score(true_labels, final_predictions, labels = [1]))
    print("Recall:", recall_score(true_labels, final_predictions, labels = [1]))
    print("Precision:", precision_score(true_labels, final_predictions, labels = [1]))

    #For each fold, save the model, weights, etc

if __name__ == "__main__":

    for idx in range(len(full_set)):
        test_set = [full_set[idx]]
        train_set = full_set[:idx] + full_set[(idx+1):]
        main(train_set, test_set, idx)

    # idx = 0
    # test_set = [full_set[idx]]
    # train_set = full_set[:idx] + full_set[(idx+1):]
    # main(train_set, test_set, idx)