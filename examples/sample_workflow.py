"""Demo workflow showing a simple building of behavior classifier.

#######################
## General todo list ##
#######################

#TODO

# * Clean up the MARS code...
# * Stacking example: I think this can all be done in sklearn... no behaveml code is needed.
# * Make a cleaner feature creation interface? One that can support any animal config
#   Better way of getting parameters to feature creation step...like framewidth may be useful, for instance?
# * Tests for F1 optimizer and HMM. Doesn't seem to improve performance much right now...
# * Plots of the DLC tracks
# * Make it use the units! This must be supported before 'releasing'

#### This will mark the end of the first 'release' version

## LATER FEATURES TO ADD

# * Support for additional input types
# * Export back to DLC format (if only interested in the interpolation functions, e.g.)
# * GUI in napari?

#WORKING ON

# * TSNE embeddings...colored by prediction label? Also can make it colored by predictor (attack/mount/investigate)
#   With clustering...
#   Make a plots.py for plotting helper functions

#DONE

# * Add save/load functionality
# * Make some requirements 'optional'... they aren't specified as required in the package spec, but add tests that
#   they are installed on the system before trying to use them. Add errors if the system doesn't support them. This way the package
#   stays light weight. 
#   Current list of unchecked optionals: tensorflow, matplotlib
#   List of checked optionals: ssm, ffmpeg
#  
#   Another way to do this is to use pip install options. e.g. I would have a pip install behaveml[all] option
#   I think these can work alongside each other actually. I implement both...
# * Tests for feature adder and removed by RE
# * Add option to add features by pattern matching (regular expressions?)
#   E.g. we want to add the features whose names start with 'likelihood'
# * Add HMM on top of all this jazz
# * Add F1 score optimizer on top of all this jazz
#   Add these as extra sk-learn models
# * Movie making with predictions. For QC... to inspect quality of predictions
# * Check that row order is preserved by DL model. Otherwise it's useless. How do we do this?
#   I have a good sign that it is: I now get 73% F1 score -- when I had the bug that swapped the videos
#   I had around 13% F1 score -- so 13% is what I should expect for scrambled, out of order, annotation rows.
#   But... I should check the performance of the DL features alone, and of the MARS features alone, to see
#   what is contributing to the performance. Perhaps only the MARS features are getting me to 73%.
# * Tests for interpolation code
# * Add DLC filtering code
# * Read in probabilities from DLC
# * Documentation
# * Add a way to enforce, for each feature creation function, that it has the columns it needs.
#   Add a 'req columns' field somewhere. The names should matter
#   Perhaps there should be a 'Feature' class... that has property req features
# * Write a bunch of tests... make sure it's behaving as it should
# * Move read BORIS function to io.py
# * Make it so that raw tracking columns are not added as features by default
# * Add a column renamer -- say you labeled your columns differently in DLC, this will 
#   name them as expected by the 'req_columns' field for the features you want.
# * Get rid of warning messages when load CNN pre-trained parameters
#   expect_partial() on the load status object, e.g. tf.train.Checkpoint.restore(...).expect_partial()

"""

###########################
## Example analysis code ##
###########################

#More reliable to not use GPU here. It's only doing inference with a small net, doesn't take long:
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ''

from glob import glob 
from behaveml import VideosetDataFrame, clone_metadata
from behaveml import mars_feature_maker, cnn_probability_feature_maker, interpolate_lowconf_points

#A list of DLC tracking files
tracking_files = sorted(glob('./tests/data/dlc/*.csv'))

#A list of BORIS labeled files, ordered to correspond with DLC tracking file list
boris_files = sorted(glob('./tests/data/boris/*.csv'))

#A list of video files, ordered to correspond with DLC tracking file list
video_files = sorted(glob('./tests/data/videos/*.avi'))

frame_length = None              # (float) length of entire horizontal shot
units = None                     # (str) units frame_length is given in
fps = 30                         # (int) frames per second
resolution = (1200, 1600)        # (tuple) HxW in pixels

#Metadata is a dictionary that attaches each of the above parameters to the video/behavior annotations
metadata = clone_metadata(tracking_files, 
                          label_files = boris_files, 
                          video_files = video_files,
                          frame_length = frame_length, 
                          fps = fps, 
                          units = units, 
                          resolution = resolution)

dataset = VideosetDataFrame(metadata)

#Filter out low-confidence DLC tracks and interpolate those points instead
print("Interpolating low-confidence tracking points")
interpolate_lowconf_points(dataset)

#Now create features on this dataset
print("Calculating MARS features")
dataset.add_features(mars_feature_maker, 
                     featureset_name = 'MARS', 
                     add_to_features = True)

#Note: by default this keras code will try to use CUDA. 
print("Calculating 1D CNN pretrained network features")
dataset.add_features(cnn_probability_feature_maker, 
                     featureset_name = '1dcnn', 
                     add_to_features = True)

#print("Adding likelihood columns")
dataset.activate_features_by_name('likelihood')

####################
# Works up to here # 
####################

# dataset.add_features(compute_stacked_features, 
#                      featureset_name = 'stacked', 
#                      add_to_features = True)

#You can also make your own 'feature maker':
#The feature creation functions take:
# * Pandas data frame (dataset.data)
# * A list of column names that indicate which columns in the data frame 
#   the features will be computed from. For basic feature, this will be the 'raw' pose-tracking column names
#   For 'model stacking' models, this can be derivative column sets
# * A dictionary, animal_setup, that contains details about the animals in the experiment
# They return 
# * Pandas data frame with the new features, in the same order as input data

#dataset.add_features(create_custom_features, 
#                     featureset_name = 'custom', 
#                     add_to_feature_col = True)

#Now all the work of the package is done, more or less, and 
# we can do ML on this object in e.g. sklearn by using the following attributes:
# dataset.features #The feature matrix setup for learning w
# dataset.label    #The labels for supervision
# dataset.group    #Used for group-level cross validation 
#                   (by default, groups are set to filename, so this implements video-level CV)

######################
## Machine learning ##
######################

## Sample ML model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from behaveml.models import F1Optimizer, HMMSklearn, ModelTransformer

splitter = GroupKFold(n_splits = dataset.n_videos)
model = ModelTransformer(RandomForestClassifier)
#model = XGBClassifier()
#model = LogisticRegression(solver = 'liblinear')
#model = KNeighborsClassifier(metric = 'manhattan')

# pipeline = Pipeline([
#                      ("rf", model),
#                      ("f1max", F1Optimizer(N = 10)),
#                      ("hmm", HMMSklearn(D = 2))
#                     ])

pipeline = Pipeline([
                     ("rf", model),
#                     ("f1max", F1Optimizer(N = 1000)),
                     ("hmm", HMMSklearn(D = 2))
                    ])


# dataset.feature_cols = dataset.data.columns[28:42]
# model = F1Optimizer(N = 10)
# f1_optimized = model.fit_transform(dataset.features, dataset.labels)

print("Fitting ML model with (group) LOO CV")
predictions = cross_val_predict(XGBClassifier(), 
                                dataset.features, 
                                dataset.labels, 
                                groups = dataset.group, 
                                cv = splitter,
                                verbose = 1,
                                n_jobs = 5)

#Append these for later use
dataset.data['prediction'] = predictions
acc = accuracy_score(dataset.labels, predictions)
f1 = f1_score(dataset.labels, predictions)
pr = precision_score(dataset.labels, predictions)
re = recall_score(dataset.labels, predictions)
print("Acc", acc, "F1", f1, 'precision', pr, 'recall', re)

#Now we have our model we can make a video of its predictions. 
#Provide the column names whose state we're going to overlay on the video, along
#with the directory to output the videos
dataset.make_movie(['label', 'prediction'], '.')