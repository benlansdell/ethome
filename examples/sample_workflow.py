"""Demo workflow showing a simple building of behavior classifier.

"""

#More reliable to not use GPU here. It's only doing inference with a small net, doesn't take long:
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ''

from glob import glob 
from ethome import create_dataset, create_metadata, interpolate_lowconf_points
from ethome.io import get_sample_data_paths

#Gather the DLC and BORIS tracking and annotation files
tracking_files, boris_files = get_sample_data_paths()

#Setup some parameters
frame_width = 20                 # (float) length of entire horizontal shot
frame_width_units = 'in'         # (str) units frame_width is given in
fps = 30                         # (int) frames per second
resolution = (1200, 1600)        # (tuple) HxW in pixels

#Create a parameter object and video dataset
metadata = create_metadata(tracking_files, 
                          labels = boris_files, 
                          frame_width = frame_width, 
                          fps = fps, 
                          frame_width_units = frame_width_units, 
                          resolution = resolution)

animal_renamer = {'adult': 'resident', 'juvenile':'intruder'}

dataset = create_dataset(metadata, animal_renamer=animal_renamer)

interpolate_lowconf_points(dataset)

#Now create features on this dataset. Can use pre-built featuresets, or make your own. Here are two that work with a mouse resident-intruder setup:
dataset.features.add('cnn1d_prob')
dataset.features.add('mars')

#########################
## Supervised learning ##
#########################

## Sample ML model
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

splitter = GroupKFold(n_splits = dataset.metadata.n_videos)

print("Fitting ML model with (group) LOO CV")
predictions = cross_val_predict(XGBClassifier(), 
                                dataset.ml.features, 
                                dataset.ml.labels, 
                                groups = dataset.ml.group, 
                                cv = splitter,
                                verbose = 1,
                                n_jobs = 1)

#Append these for later use
dataset['prediction'] = predictions
acc = accuracy_score(dataset.ml.labels, predictions)
f1 = f1_score(dataset.ml.labels, predictions)
pr = precision_score(dataset.ml.labels, predictions)
re = recall_score(dataset.ml.labels, predictions)
print("Acc", acc, "F1", f1, 'precision', pr, 'recall', re)

#####################
## Post processing ##
#####################

#Now we have our model we can make a video of its predictions. 
#Provide the column names whose state we're going to overlay on the video, along
#with the directory to output the videos
dataset.io.make_movie(['label', 'prediction'], '.')

#NOTE: need to have provided 'video' column in the metadata to make movies.