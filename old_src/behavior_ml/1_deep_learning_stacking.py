#!/usr/bin/env python3
import numpy as np
import os
import argparse
import time 
import json 
import pickle 

from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
import keras.layers as layers
import keras.models as models
import tensorflow_addons as tfa
from keras.callbacks import LearningRateScheduler

import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score
from copy import deepcopy
import tqdm

from lib.utils import seed_everything, validate_submission
from lib.helper import API_KEY
from fscores import macroF1

###############
## DL models ##
###############

from dl_models import build_baseline_model

###################
## DL generators ##
###################

from dl_generators import MABe_Generator, features_mars_distr, features_identity

###################
## Grid searches ##
###################

from grid_searches import sweeps_baseline, feature_spaces

path = './'

seed_everything()
  
#Implement cross validation, instead of a single validation set
#Product predictions on the held out data, all merged into one frame
class Trainer:
    def __init__(self, *,
               pose_dictionary,
               splitter,
               anno_df,
               feature_dim, 
               batch_size, 
               num_classes,
               test_data = None,
               augment=False,
               class_to_number=None,
               past_frames=0, 
               future_frames=0,
               frame_gap=1, 
               use_conv=False, 
               build_model = build_baseline_model,
               Generator = MABe_Generator,
               use_callbacks = False,
               learning_decay_freq = 10,
               featurizer = features_identity):

        flat_dim = np.prod(feature_dim)
        if use_conv:
            input_dim = ((past_frames + future_frames + 1), flat_dim,)
        else:
            input_dim = (flat_dim * (past_frames + future_frames + 1),)

        self.learning_decay_freq = learning_decay_freq
        self.input_dim = input_dim
        self.use_conv=use_conv
        self.num_classes=num_classes
        self.build_model = build_model 
        self.use_callbacks = use_callbacks

        c2n = {'other': 0,'investigation': 1,
                    'attack' : 2, 'mount' : 3}
        self.class_to_number = class_to_number or c2n

        #Create a generator for each fold:
        self.train_generators = []
        self.val_generators_train = []
        self.val_generators_predict = []
        self.val_indices = []

        #Make the base generators for retraining on all data:
        self.train_generator_all = Generator(pose_dictionary, 
                                        batch_size=batch_size, 
                                        dim=input_dim,
                                        num_classes=num_classes, 
                                        past_frames=past_frames, 
                                        future_frames=future_frames,
                                        class_to_number=self.class_to_number,
                                        use_conv=use_conv,
                                        frame_gap=frame_gap,
                                        augment=augment,
                                        shuffle=True,
                                        mode='fit',
                                        featurize = featurizer) 

        if test_data is not None:
            self.test_generator = Generator(test_data, 
                                    batch_size=8192, 
                                    dim=input_dim, 
                                    num_classes=num_classes, 
                                    past_frames=past_frames,
                                    future_frames=future_frames,
                                    use_conv=use_conv,
                                    class_to_number=self.class_to_number,
                                    frame_gap=frame_gap,
                                    augment=False,
                                    shuffle=False,
                                    mode='predict',
                                    featurize = featurizer)

        #CV folds to define training and validation generators
        for idx_train, idx_val in splitter.split(anno_df.index):

            #Need to save these 
            self.val_indices.append(idx_val)
            #Create appropriate split (Using sklearn function?)
            train_data = {k : pose_dictionary[k] for k in anno_df.index[idx_train]}
            val_data = {k : pose_dictionary[k] for k in anno_df.index[idx_val]}

            #Append a generator with that data
            self.train_generators.append(
                                        Generator(train_data, 
                                            batch_size=batch_size, 
                                            dim=input_dim,
                                            num_classes=num_classes, 
                                            past_frames=past_frames, 
                                            future_frames=future_frames,
                                            class_to_number=self.class_to_number,
                                            use_conv=use_conv,
                                            frame_gap=frame_gap,
                                            augment=augment,
                                            shuffle=True,
                                            mode='fit',
                                            featurize = featurizer)
                                        )   

            self.val_generators_train.append(
                                        Generator(val_data, 
                                            batch_size=batch_size, 
                                            dim=input_dim, 
                                            num_classes=num_classes, 
                                            past_frames=past_frames,
                                            future_frames=future_frames,
                                            use_conv=use_conv,
                                            class_to_number=self.class_to_number,
                                            frame_gap=frame_gap,
                                            augment=False,
                                            shuffle=False,
                                            mode='fit',
                                            featurize = featurizer)
                                    )
  
            self.val_generators_predict.append(
                                        Generator(val_data, 
                                            batch_size=batch_size, 
                                            dim=input_dim, 
                                            num_classes=num_classes, 
                                            past_frames=past_frames,
                                            future_frames=future_frames,
                                            use_conv=use_conv,
                                            class_to_number=self.class_to_number,
                                            frame_gap=frame_gap,
                                            augment=False,
                                            shuffle=False,
                                            mode='predict',
                                            featurize = featurizer)
                                    )

    def delete_model(self):
        self.model = None

    def initialize_model(self, **kwargs):
        self.model = self.build_model(input_dim = self.input_dim, **kwargs)

    def _set_model(self, model):
        """ Set an external, provide initialized and compiled keras model """
        self.model = model

    def train(self, model_params, epochs=20, class_weight=None, steps_per_epoch = None, tune_callbacks = True):

        kwargs = {}
        if class_weight is not None:
            if type(class_weight) is dict:
                kwargs['class_weight'] = class_weight

        callbacks = [LearningRateScheduler(lambda x,y: lrs(x,y,self.learning_decay_freq))]

        if self.use_callbacks:
            callbacks += [macroF1(self.model, self.val_inputs, self.val_targets)]

        num_classes = 4
        n_folds = len(self.train_generators)

        all_val_preds = {}
        all_val_pred_probs = {}
        all_test_pred_probs = {}

        count = 0
        #Repeat this fit process:
        for tg, vg_train, vg_predict in zip(self.train_generators, self.val_generators_train, self.val_generators_predict):

            #Reinit model to start training again
            self.initialize_model(**model_params)

            #Instead of training, just load the model
            #loaded_model = keras.models.load_model(f'./results/task1_trained_model_fold_{count}.h5')
            #keras.models.load_model(f'./results/task1_model_fold_{count}/the_model')
            loaded_model = self.model
            loaded_model.load_weights(f'./results/task1_model_fold_{count}/the_model/variables/variables')
            self.model = loaded_model 

            count += 1

            #Once trained we can do the inference on the test data:
            test_pred_probs = self.get_test_prediction_probabilities()

            #Add this to the all_test_pred_prob
            for k in test_pred_probs:
                if k in all_test_pred_probs:
                    all_test_pred_probs[k] += test_pred_probs[k]
                else:
                    all_test_pred_probs[k] = test_pred_probs[k]

        #Once we've done that
        #Take the average over all these predictions:
        for k in all_test_pred_probs:
            all_test_pred_probs[k] /= n_folds

        #Save the test probabilities, averaged over the k models
        fn_test_out = f'{path}/data/intermediate/deep_learning_stacking_prediction_probabilities_test_baseline_test_run_distances.npy'
        np.save(fn_test_out, all_test_pred_probs)        

    def get_test_prediction_probabilities(self):
        all_test_preds = {}
        for vkey in self.test_generator.video_keys:
            nframes = self.test_generator.seq_lengths[vkey]
            all_test_preds[vkey] = np.zeros((nframes,4), dtype=np.float32)

        for X, vkey_fi_list in tqdm.tqdm(self.test_generator):
            test_pred = self.model.predict(X)
            #test_pred = np.argmax(test_pred, axis=-1)

            if len(test_pred.shape) > 2:
                test_pred = self.get_final_val_probabilities(test_pred)

            for p, (vkey, fi) in zip(test_pred, vkey_fi_list):
                all_test_preds[vkey][fi] = p
        return all_test_preds

def normalize_data(orig_pose_dictionary):
    for key in orig_pose_dictionary:
        X = orig_pose_dictionary[key]['keypoints']
        X = X.transpose((0,1,3,2)) #last axis is x, y coordinates
        #How should we normalize now????
        #X[..., 0] = X[..., 0]/1300
        #X[..., 1] = X[..., 1]/1200

        #How should we normalize now???? This will keep the aspect ratio similar to the MARS dataset...
        #X[..., 0] = X[..., 0]/2400
        #X[..., 1] = X[..., 1]/1200

        #There are other options... but we'll go with this for now

        #Original normalization
        X[..., 0] = X[..., 0]/1024
        X[..., 1] = X[..., 1]/570

        orig_pose_dictionary[key]['keypoints'] = X
    return orig_pose_dictionary

def split_validation_cv(orig_pose_dictionary, vocabulary, number_to_class, seed=2021, 
                       n_folds = 5):

    def num_to_text(anno_list):
        return np.vectorize(number_to_class.get)(anno_list)

    def get_percentage(sequence_key):
        anno_seq = num_to_text(orig_pose_dictionary[sequence_key]['annotations'])
        counts = {k: np.mean(np.array(anno_seq) == k) for k in vocabulary}
        return counts

    anno_percentages = {k: get_percentage(k) for k in orig_pose_dictionary}
    anno_perc_df = pd.DataFrame(anno_percentages).T
    rng_state = np.random.RandomState(seed)

    splitter = KFold(n_splits=n_folds, random_state=rng_state, shuffle = True)

    return splitter, anno_perc_df

def run_task(results_dir, dataset, vocabulary, test_data, config_name, number_to_class,
              build_model, augment=False, epochs=15, skip_test_prediction=False, seed=2021,
              Generator = MABe_Generator, use_callbacks = False, params = None, use_conv = True):

    if params is None:
        if config_name is None:
            raise ValueError("Provide one of params dictionary or config_name with path to json file")
        with open(config_name, 'r') as fp:
            params = json.load(fp)

    normalize = params["normalize"]
    params["seed"] = seed
    seed_everything(seed)

    if "steps_per_epoch" in params:
        steps_per_epoch = params["steps_per_epoch"]
        if steps_per_epoch == -1:
            steps_per_epoch = None
    else:
        steps_per_epoch = None

    features = params['features']
    feature_dim = feature_spaces[features][1]
    featurizer = feature_spaces[features][0]

    if normalize:
        dataset = normalize_data(deepcopy(dataset))
        if not skip_test_prediction:
            test_data = normalize_data(deepcopy(test_data))
        else:
            test_data = None

    splitter, anno_perc_df = split_validation_cv(dataset, 
                                    seed=seed,
                                    number_to_class = number_to_class,
                                    vocabulary=vocabulary,
                                    n_folds = 5)                               

    num_classes = len(anno_perc_df.keys())

    epochs = params["epochs"]

    class_to_number = vocabulary

    trainer = Trainer(pose_dictionary = dataset,
                    test_data = test_data,
                    splitter=splitter,
                    anno_df=anno_perc_df,
                    feature_dim=feature_dim, 
                    batch_size=params['batch_size'], 
                    num_classes=num_classes,
                    augment=augment,
                    class_to_number=class_to_number,
                    past_frames=params['past_frames'], 
                    future_frames=params['future_frames'],
                    frame_gap=params['frame_gap'],
                    use_conv=use_conv,
                    build_model = build_model,
                    Generator = Generator,
                    use_callbacks = use_callbacks,
                    learning_decay_freq=params['learning_decay_freq'],
                    featurizer = featurizer)

    #Extract model param dictionary
    model_params = {}
    for k in params:
        if 'model_param' in k:
            k_ = k.split('__')[1]
            model_params[k_] = params[k]

    class_weight_lambda = 1

    if params['reweight_loss'] is True:
        if len(trainer.train_generator.class_weights.shape) == 1:
            class_weight = {k:class_weight_lambda/(v+class_weight_lambda) for k,v in enumerate(trainer.train_generator.class_weights)}
        else:
            class_weight = 1/trainer.train_generator.class_weights
    else:
        class_weight = None

    model_params['class_weight'] = class_weight

    trainer.train(epochs=epochs, steps_per_epoch = steps_per_epoch, class_weight = class_weight, model_params = model_params)

def lrs(epoch, lr, freq = 10):
    if (epoch % freq) == 0 and epoch > 0:
        lr /= 3 
    return lr

def main():

    parametersweep = 'test_run_distances'
    config_name = 'dl_baseline_settings.json'
    build_model = build_baseline_model
    Generator = MABe_Generator
    use_callbacks = False
    sweeps = sweeps_baseline

    #Load default config
    with open(config_name, 'r') as fp:
        config = json.load(fp)
    if 'model_param__layer_channels' in config:
        config['model_param__layer_channels'] = tuple(config['model_param__layer_channels'])

    #Modify to setup parameter sweep    
    for k in sweeps[parametersweep][1]:
        config[k] = sweeps[parametersweep][1][k]

    arguments = {'config_name': config_name,
            'build_model': build_model,
            'Generator': Generator,
            'use_callbacks': use_callbacks}

    config['future_frames'] = 50
    config['past_frames'] = 50
    config['model_param__learning_rate'] = 0.0001

    train = np.load(path + 'data/train.npy',allow_pickle=True).item()
    test = np.load(path + 'data/test_inference.npy',allow_pickle=True).item()
            
    number_to_class = {i: s for i, s in enumerate(train['vocabulary'])}

    results_dir = './results/'
    run_task(results_dir,
                        dataset=train['sequences'], 
                        vocabulary=train['vocabulary'],
                        test_data=test['sequences'],
                        config_name = None,
                        build_model = arguments['build_model'],
                        number_to_class = number_to_class,
                        seed=2021,
                        Generator = arguments['Generator'],
                        use_callbacks = arguments['use_callbacks'],
                        params = config)

if __name__ == "__main__":
    main()