import numpy as np
import json 
import pandas as pd
import tqdm
import os
from copy import deepcopy

from behaveml.dl.dl_models import build_baseline_model
from behaveml.dl.dl_generators import MABe_Generator, features_identity
from behaveml.dl.grid_searches import sweeps_baseline, feature_spaces

THIS_FILE_DIR = os.path.dirname(os.path.realpath(__file__))

def seed_everything(seed = 2012):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything()

#Implement cross validation, instead of a single validation set
#Product predictions on the held out data, all merged into one frame
class Trainer(object):
    def __init__(self, *,
               feature_dim, 
               num_classes,
               test_data = None,
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

        c2n = {'other': 0,'investigation': 1, 'attack' : 2, 'mount' : 3}
        self.class_to_number = class_to_number or c2n

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

    def delete_model(self):
        self.model = None

    def initialize_model(self, **kwargs):
        self.model = self.build_model(input_dim = self.input_dim, **kwargs)

    def _set_model(self, model):
        """ Set an external, provide initialized and compiled keras model """
        self.model = model

    def train(self, model_params, class_weight=None, n_folds = 5):

        kwargs = {}
        if class_weight is not None:
            if type(class_weight) is dict:
                kwargs['class_weight'] = class_weight

        all_test_pred_probs = {}

        #Repeat this fit process:
        for count in range(n_folds):

            #Reinit model to start training again
            self.initialize_model(**model_params)

            #Instead of training, just load the model
            model_path = os.path.join(THIS_FILE_DIR, 
                                      'pretrained_models', 
                                      f'task1_model_fold_{count}/the_model/variables/variables')
            loaded_model = self.model
            loaded_model.load_weights(model_path).expect_partial()
            self.model = loaded_model 

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

        return all_test_pred_probs

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
        #Original MARS normalization
        X[..., 0] = X[..., 0]/1024
        X[..., 1] = X[..., 1]/570
        orig_pose_dictionary[key]['keypoints'] = X
    return orig_pose_dictionary

def run_task(vocabulary, test_data, config_name,
              build_model, skip_test_prediction=False, seed=2021,
              Generator = MABe_Generator, use_callbacks = False, params = None, 
              use_conv = True):

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
        if not skip_test_prediction:
            test_data = normalize_data(deepcopy(test_data))
        else:
            test_data = None

    num_classes = 4

    class_to_number = vocabulary

    trainer = Trainer(feature_dim=feature_dim, 
                      num_classes=num_classes,
                      test_data = test_data,
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

    if params['reweight_loss'] is True:
        class_weight = 1/trainer.train_generator.class_weights
    else:
        class_weight = None

    model_params['class_weight'] = class_weight
    all_test_probs = trainer.train(class_weight = class_weight, model_params = model_params)
    return all_test_probs

def lrs(epoch, lr, freq = 10):
    if (epoch % freq) == 0 and epoch > 0:
        lr /= 3 
    return lr

def convert_to_mars_format(df, colnames, animal_setup):
    n_animals = len(animal_setup['mouse_ids'])
    n_body_parts = len(animal_setup['bodypart_ids'])
    pose_dict = {}
    videos = np.unique(df.filename)
    for vid in videos:
        pose_dict[vid] = {'annotator_id': 0}
        keypoints = df.loc[df.filename == vid, colnames]
        n_rows = keypoints.shape[0]
        pose_dict[vid]['keypoints'] = keypoints.to_numpy().reshape((n_rows, n_animals, 2, n_body_parts))
    return pose_dict

#Basically, undo the change above
def convert_to_pandas_df(data, colnames = None):
    dfs = []
    for vid in data:
        df = pd.DataFrame(data[vid], columns = colnames)
        dfs.append(df)
    final_df = pd.concat(dfs, axis = 0)
    return final_df

def compute_dl_probability_features(df : pd.DataFrame, raw_col_names : list, animal_setup : dict, **kwargs):

    test_data = convert_to_mars_format(df, raw_col_names, animal_setup)
    parametersweep = 'test_run_distances'
    config_name = os.path.join(THIS_FILE_DIR, 'dl_baseline_settings.json')
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

    vocab = {'attack': 0, 'investigation': 1, 'mount': 2, 'other': 3}
                            
    all_test_probs = run_task(vocabulary=vocab,
                                test_data=test_data,
                                config_name = None,
                                build_model = arguments['build_model'],
                                seed=2021,
                                Generator = arguments['Generator'],
                                use_callbacks = arguments['use_callbacks'],
                                params = config)

    colnames = [f'prob_{k}' for k in vocab.keys()]
    test_df = convert_to_pandas_df(all_test_probs, colnames = colnames)
    return test_df