from .dl_generators import *

feature_spaces = {'distances': [features_distances, [91]], 
                    'identity': [features_identity, [2,7,2]],
                    'distances_normalized': [features_distances_normalized, [91]],
                    'mars': [features_mars, []],
                    'mars_distr': [features_mars_distr, []],
                }
                    # This one was removed....I think it's not needed
                    # 'mars_no_shift': [features_mars_no_shift, [160]]

sweeps_baseline = {
    'test_run_distances':
        [{'num_samples': 1},
        {
            "model_param__learning_rate": 0.0001,
            "epochs": 15,
            "features": "distances",
            "model_param__dropout_rate": 0.5
        }]
}