"""Demo workflow showing a simple building of behavior classifier.

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
from behaveml.io import get_sample_data_paths

#A list of DLC tracking files
#tracking_files = sorted(glob('./tests/data/dlc/*.csv'))

#A list of BORIS labeled files, ordered to correspond with DLC tracking file list
#boris_files = sorted(glob('./tests/data/boris/*.csv'))

tracking_files, boris_files = get_sample_data_paths()

#A list of video files, ordered to correspond with DLC tracking file list
video_files = sorted(glob('./tests/data/videos/*.avi'))

#Get test data:


frame_length = None              # (float) length of entire horizontal shot
units = None                     # (str) units frame_length is given in
fps = 30                         # (int) frames per second
resolution = (1200, 1600)        # (tuple) HxW in pixels

#Metadata is a dictionary that attaches each of the above parameters to the video/behavior annotations
#                          video_files = video_files,
metadata = clone_metadata(tracking_files, 
                          label_files = boris_files, 
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

#########################
## Supervised learning ##
#########################

## Sample ML model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

#from behaveml.models import F1Optimizer, HMMSklearn, ModelTransformer
from behaveml.models import F1Optimizer, ModelTransformer

splitter = GroupKFold(n_splits = dataset.n_videos)
model = ModelTransformer(RandomForestClassifier)

pipeline = Pipeline([
                     ("rf", model),
                     ("hmm", HMMSklearn(D = 2))
                    ])

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

#####################
## Post processing ##
#####################

#Now we have our model we can make a video of its predictions. 
#Provide the column names whose state we're going to overlay on the video, along
#with the directory to output the videos
dataset.make_movie(['label', 'prediction'], '.')
