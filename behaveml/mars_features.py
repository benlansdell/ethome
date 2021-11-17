import pandas as pd 
import numpy as np
import pickle 

from joblib import load

import sklearn.metrics 
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler

from behaveml.dl.feature_engineering import make_features_mars_w_1dcnn_features, \
                                            make_features_mars_w_1dcnn_features_test, \
                                            make_features_mars_distr

def compute_mars_features(df : pd.DataFrame, raw_col_names : list, animal_setup : dict, **kwargs) -> pd.DataFrame:
    features_df, _, _ = make_features_mars_distr(df[raw_col_names], animal_setup) 
    return features_df

def make_stacked_features(train_df, test_df): # pragma: no cover

    mars_features_df, reversemap, _ = make_features_mars_w_1dcnn_features(train_df)
    mars_features_df_test, reversemap_test, _ = make_features_mars_w_1dcnn_features_test(test_df)

    features_name = 'features_mars_distr_stacked_w_1dcnn'
    
    n_folds = 5

    X = mars_features_df.drop(columns = ['annotation', 'seq_id'])
    y = mars_features_df['annotation']
    groups = mars_features_df['seq_id']

    X_test = mars_features_df_test.drop(columns = 'seq_id')
    groups_test = mars_features_df_test['seq_id']

    # Setup the CV folds
    cv_groups = GroupKFold(n_splits = n_folds)
    scorer = sklearn.metrics.make_scorer(f1_score, labels = [0,1,2], average = 'macro')

    all_base_models = []

    validation_groups_by_fold = []

    for fold_idx, (train_index, val_index) in enumerate(cv_groups.split(X, y, groups)):

        #fold_idx = 0; train_index, val_index = next(cv_groups.split(X, y, groups))

        X_train = np.array(X)[train_index]
        y_train = np.array(y)[train_index]
        groups_train = groups[train_index]

        y_val = np.array(y)[val_index]
        X_val = np.array(X)[val_index]
        groups_val = groups[val_index]

        validation_groups_by_fold.append(groups_val)

        # Define a bunch of models for the stacking, one for each fold
        level0 = list()
        level0.append(['bayes', Pipeline([('scaler', StandardScaler()), ('nb', GaussianNB())]), [], []])
        level0.append(['et_entropy', ExtraTreesClassifier(n_estimators = 100, criterion='entropy', n_jobs = 5, verbose = 1, random_state = 42), [], []])
        level0.append(['et_gini', ExtraTreesClassifier(n_estimators = 100, criterion='gini', n_jobs = 5, verbose = 1, random_state = 42), [], []])
        level0.append(['rf_entropy', RandomForestClassifier(n_estimators = 20, criterion='entropy', n_jobs = 5, random_state = 42), [], []])
        level0.append(['rf_gini', RandomForestClassifier(n_estimators = 20, criterion='gini', n_jobs = 5, random_state = 42), [], []])

        for idx in range(len(level0)):
            name, model, _, _ = level0[idx]
            print(f"Fitting {name} in fold {fold_idx}")
            #Train the models on each fold, make prediction on held-out for each fold
#            model.fit(X_train, y_train)

            #Save the models too
            model = load(f'./results/level_0_model_{idx}_fold_{fold_idx}.joblib')

            predict_proba_val = model.predict_proba(X_val)
            predict_proba_test = model.predict_proba(X_test)
            #Save this
            level0[idx][2] = predict_proba_val
            level0[idx][3] = predict_proba_test

        all_base_models.append(level0)

    # Once done, concat predictions to make train data for XGB. 
    # And run the inference with the mean of each model to make the test data

    all_val_preds = []
    all_test_preds = {}
    for fold_idx in range(n_folds):
        n_row = validation_groups_by_fold[fold_idx].shape[0]
        base_models = all_base_models[fold_idx]
        col_names = []
        val_preds = np.zeros((n_row, 4*len(level0)))
        for model_idx, model in enumerate(base_models):


            name = model[0]
            classifier = model[1]
            this_val_pred = model[2]
            this_test_pred = model[3]

            if name in all_test_preds:
                all_test_preds[name] += this_test_pred
            else:
                all_test_preds[name] = this_test_pred

            col_names += [f'{name}_{i}' for i in range(4)]
            val_preds[:,(model_idx*4):((model_idx+1)*4)] = this_val_pred

        val_preds = pd.DataFrame(val_preds, columns = col_names)
        val_preds['seq_id'] = validation_groups_by_fold[fold_idx].reset_index()['seq_id']
        all_val_preds.append(val_preds)

    for name in all_test_preds:
        all_test_preds[name] /= n_folds

    all_test_data = np.hstack([v for v in all_test_preds.values()])
    all_test_df = pd.DataFrame(all_test_data, columns = col_names)
    all_test_df['seq_id'] = groups_test

    #Concat all the validation predictions, along with the group names for that fold
    all_val_preds = pd.concat(all_val_preds)

    # Append this to the mars_distr_w_1dcnn file so we have everything together to train the XGB model on
    mars_features_df[col_names] = np.nan
    seq_ids = pd.unique(mars_features_df['seq_id'])
    for seq_id in seq_ids:
        mars_features_df.loc[mars_features_df['seq_id'] == seq_id, col_names] = np.array(all_val_preds.loc[all_val_preds['seq_id'] == seq_id, col_names])

    # Append this to the mars_distr_w_1dcnn test files so we have everything together to test the model
    mars_features_df_test[col_names] = np.nan
    seq_ids = pd.unique(mars_features_df_test['seq_id'])
    for seq_id in seq_ids:
        mars_features_df_test.loc[mars_features_df_test['seq_id'] == seq_id, col_names] = np.array(all_test_df.loc[all_test_df['seq_id'] == seq_id, col_names])

    # Save the feature files
    mars_features_df.to_csv(f'./data/intermediate/train_{features_name}.csv', index = False)
    with open(f'./data/intermediate/test_map_{features_name}.pkl', 'wb') as handle:
        pickle.dump(reversemap_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Saving test feature file (may take a while)')
    mars_features_df_test.to_csv(f'./data/intermediate/test_{features_name}.csv', index = False)

    return mars_features_df, reversemap, name, mars_features_df_test, reversemap_test

def compute_mars_features_(df : pd.DataFrame, raw_col_names : list, animal_setup : dict): # pragma: no cover

    train_df = pd.read_csv('./data/intermediate/train_df.csv')
    test_df = pd.read_csv('./data/intermediate/test_df.csv')

    #Make features
    train_features, _, name, test_features, test_map = make_stacked_features(train_df, test_df)

    return pd.DataFrame(df) 
