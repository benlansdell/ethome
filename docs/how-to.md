
# How To guide

This guide covers the all of the tasks you can perform with this package, roughly in the order you'd want to do them. A very basic outline is also in the Quick Start section of the readme. 

This guide covers basic usage -- it doesn't comprehensively describe how to use every function or feature in `ethome`; you can consult the API docs for complete information on usage. After installation, cut and past the code samples below to follow along.

## 1 Getting started

`ethome` makes it easy to perform common machine learning analyses on pose-tracking data, perhaps in combination with behavioral annotations. The key thing you need to get started, then, is pose tracking data. At present, data from DeepLabCut or pose data stored in NWB files is supported (via the `ndx-pose` extension). 

### 1a Loading NWB files

The first task is to load the data into a form useful for machine learning. The basic object of the package is an extended pandas `DataFrame`, which provides associated support functions that are suited for behavior analysis. The `DataFrame` object will house data from one or more video's worth of pose data, along with associated metadata for each video. 

The NeurodataWithoutBorders format can store both pose tracking data and behavioral annotations, along with associated metadata. If all of your data is stored in this format, then it's easy to import it into `ethome`:
```python
from ethome import create_dataset
from ethome.io import get_sample_nwb_paths
fn_in = get_sample_nwb_paths()
recordings = create_dataset(fn_in)
```

You can provide multiple recordings, just provide a list of paths instead. Each separate file is assumed to represent a different session/experiment/time period. I.e., they're *not* meant to represent the same session from different cameras, or the same session for different animals.  

### 1b Loading your own metadata/loading DLC files

If your data is stored in DeepLabCut `csv`s or `h5` files, perhaps with accompanying behavioral annotations from [BORIS](https://www.boris.unito.it/), then you'll have to associate these with each other, and provide relevant metadata yourself. Sections 1b -> 1f outline how to do this. Data stored in NWB files have already addressed each of these steps and you can skip these sections. 

To import the data, you'll need to provide metadata for each video you want to analyze. For this, you create a `metadata` dictionary housing this information. This is a dictionary whose keys are paths to pose-tracking DLC `.csv`s -- this is how each video is identified. The value of each entry is a dictionary that provides details about that video. For instance, you may have:
```python
tracking_csv = './dlc_tracking_file.csv'
metadata = {tracking_csv : {'fps': 30, 'resolution': (1200, 1600)}}
```

NOTE: Beyond providing the `fps` for each video, all other fields are optional. 

### 1c Helper function for making metadata dictionary

Often you'll have many videos that have the same metadata, in that case you can easily create an appropriate dictionary with the helper function `create_metadata`. Say you now have two tracking files, each with the same FPS and resolution. You can make the corresponding metadata dictionary with:
```python
from ethome import create_metadata
tracking_csvs = ['./dlc_tracking_file_1.csv', './dlc_tracking_file_2.csv']
fps = 30
resolution = (1200, 1600)
metadata = create_metadata(tracking_csvs, fps = fps, resolution = resolution)
```
The `metadata` dictionary now has two items, one for each video, each listing the same FPS and resolution. 

NOTE: Any keyword that is an iterable of the same length as the tracking files is zipped with the tracking files accordingly. That is, if you also have behavioral annotations provided by BORIS for each of the videos, then you should prepare a list `labeled_data` and provide that to `create_metadata`:
```python
tracking_csvs = ['./dlc_tracking_file_1.csv', './dlc_tracking_file_2.csv']
labeled_data = ['./boris_tracking_file_1.csv', './boris_tracking_file_2.csv']
fps = 30
resolution = (1200, 1600)
metadata = create_metadata(tracking_csvs, labels = labeled_data, fps = fps, resolution = resolution)
```
Rather than assigning the same value (e.g. `fps = 30`) to all videos, the entry `labeled_data[i]` would then be associated with `tracking_csvs[i]`. These lists, therefore, must be sorted appropriately.

### 1d Special fields

When making this metadata dictionary, keep in mind:
* The `labels` field is special. If it is provided, then it is treated as housing the paths to behavioral annotations exported from a corresponding BORIS project. The package loads these behavior annotations and adds them to the data frame with the field `label`.
* The `video` field is also special. You should use it to provide a path to the corresponding video that was tracked. If available, this will be used by some of the visualization functions.
* For each video, the `fps` field must be provided, so that frame numbers can be converted into times.

### 1e Scaling pose data

There is some support for scaling the data to get it into desired units, consistent across all recordings. 

If the tracking is in pixels and you do want to rescale it to some physical distance, you should provide `frame_width`, `frame_width_units` and `resolution` for all videos. This ensures the entire dataset is using the same units. The package will use these values for each video to rescale the (presumed) pixel coordinates to physical coordinates.

`resolution` is a tuple (H,W) in pixels of the videos and `frame_width` is the width of the image, in units `frame_width_units`.

By default, all coordinates are converted to 'mm'. The pair 'units':'mm' is added to the metadata dictionary for each video. You can specify the units by providing the `units` key yourself. Supported units include: 'mm', 'cm', 'm', 'in', 'ft'.

If the DLC/tracking files are already in desired units, either in physical distances, or pixels, then do *not* provide all of the fields `frame_width`, `resolution`, and `frame_width_units`. If you want to keep track of the units, you can add a `units` key to the metadata. This could be `pixels`, `cm`, etc, as appropriate.

### 1f Making the data frame

Once you have the metadata dictionary prepared, you can easily create a `DataFrame` as:
```python
recordings = create_dataset(metadata)
```

This creates a pandas dataframe, `recordings`, that contains pose data, and perhaps behavior annotations, from all the videos listed in `metadata`.

### 1g Renaming things

If your tracking project named the animals some way, but you want them named another way in this dataframe, you can provide an `animal_renamer` dictionary as an argument to the constructor:
```python
recordings = create_dataset(metadata, animal_renamer={'adult': 'resident', 'juvenile':'intruder'})
```
Similarly with the body parts -- you can provide a `part_renamer` dictionary.

### 1h Metadata

When `recordings` is created, additional metadata is computed and accessible via:
* `recordings.metadata` houses the following attributes:
    * `details`: the metadata dictionary given to create_dataset
    * `videos`: list of videos given in `metadata`
    * `n_videos`: number of videos in DataFrame
    * `label_key`: associates numbers with text labels for each behavior
    * `reverse_label_key`: associates text labels for each behavior number
* `recordings.pose`, houses pose information:
    * `body_parts`: list of body parts loaded from the DLC file(s)
    * `animals`: list of animals loaded from the DLC files(s)
    * `animal_setup`: dictionary detailing animal parts and names
    * `raw_track_columns`: all original columns names loaded from DLC

## 2 Interpolate low-confidence pose tracking

Some simple support for interpolating low-confidence tracks in DLC is provided. Often predicted locations below a given confidence level are noisy and unreliable, and better tracks may be obtained by removing these predictions and interpolating from more confident predictions on either side of the uncertain prediction. 

You can achieve this with
```python
from ethome import interpolate_lowconf_points
interpolate_lowconf_points(recordings)
```

## 3 Generate features

To do machine learning you'll want to create features from the pose tracking data. `ethome` can help you do this in a few different ways. You can either use one of the feature-making functions provided or create a custom feature-making function, or custom class. 

To use the inbuilt functions you can reference them by identifying string, or provide the function itself to the `features.add` function. For instance, to compute the distances between all body parts (within and between animals), you could do:
```python
recordings.features.add('distances')
```
This will compute and add the distances between all body parts of all animals.

### 3a In-built support for resident-intruder setup

First, if your setup is a social mouse study, involving two mice, similar enough to the standard resident-intruder setup, then you can use some pre-designed feature sets. The body parts that are tracked must be those from the MARS dataset (See figure). You will have to have labeled and tracked your mice in DLC in the same way. (with the same animal and body part names -- `ethome`'s `create_dataset` function can rename them appropriately)

![Resident-intruder keypoints](assets/mars_keypoints.png)

The animals must be named `resident` and `intruder`, and the body parts must be: `nose`, `leftear`, `rightear`, `neck`, `lefthip`, `righthip`, and `tail`.

The `cnn1d_prob`, `mars`, `mars_reduced` and `social` functions can be used to make features for this setup. 

* `cnn1d_prob` runs a 1D CNN and outputs prediction probabilities of three behaviors (attack, mount, and investigation). Even if you're not interested in these exact behaviors, they may still be useful for predicting the occurance of other behaviors, as part of an ensemble model. 
* `mars` computes a long list of features as used in the MARS paper. You can refer to that paper for more details. 
* `mars_reduced` is a reduced version of the MARS features
* `social` is a set of features that only involve measures of one animal in relation to the other.

### 3b Generic features

You can generate more generic features using the following functions:
* `centroid` the centroid of each animal's body parts
* `centroid_velocity` the velocity of the centroids
* `centroid_interanimal` the distances between the centroids of all the animals
* `centroid_interanimal_speed` the rate of change of `centroid_interanimal`
* `intrabodypartspeeds` the speeds of all body parts
* `intrabodypartdistances` the distances between all animals body parts (inter- and intra-animal)
* `distances` is an alias for `intrabodypartdistances`

These classes work for any animal setup, not just resident-intruder with specific body parts, as assumed for the `mars` features.

### 3c Add your own features

There are two ways to add your own feature sets to your DataFrame. 

The first is to create a function that takes a pandas DataFrame, and returns a new DataFrame with the features you want to add. For example:
```python
def diff_cols(df, required_columns = []):
    return df[required_columns].diff()

recordings.features.add(diff_cols, required_columns = ['resident_neck_x', 'resident_neck_y'])
```

The second is to create a class that has, at the least, the method `transform`. 
```python
class BodyPartDiff:
    def __init__(self, required_columns):
        self.required_columns = required_columns

    def transform(self, df):
        return df[self.required_columns].diff()

head_diff = BodyPartDiff(['resident_neck_x', 'resident_neck_y'])
recordings.features.add(head_diff)
```
This is more verbose than the above, but has the advantage that the it can be re-used. E.g. you may want to fit the instance to training data and apply it to test data, similar to an sklearn model.

### 3d Features manipulation

By default, when new features are added to the dataframe, they are considered 'active'. Active features can be accessed through
```python
recordings.ml.features
```
You can pass this to any ML method for further processing. This `.ml.features` is just a convenience for managing the long list of features you will have created in the steps above. You can always just treat `recordings` like a pandas DataFrame and do ML how you would normally. 

To activate features you can use `recordings.features.activate`, and to deactivate features you can use `recordings.features.deactivate`. Deactivating keeps them in the DataTable, but just no longer includes those features in the `recordings.ml.features` view.

## 4 Fit a model for behavior classification

Ok! The hard work is done, so now you can easily train a behavior classifier based on the features you've computed and the labels provided. 

E.g.
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut

cv = LeaveOneGroupOut()
model = RandomForestClassifier()
cross_val_score(model, recording.ml.features, recording.ml.labels, recordings.ml.group, cv = cv)
```

A convenience function that essentially runs the above lines is provided, 
`add_randomforest_predictions`:
```python
from ethome import add_randomforest_predictions
add_randomforest_predictions(recording)
```
which can be used as a starting point for developing behavior classifiers. 

## 5 Make output movies

Now we have our model we can make a video of its predictions. Provide the column names whose state we're going to overlay on the video, along with the directory to output the videos:

```python
dataset.io.save_movie(['label', 'prediction'], '.')
```
The video field in the `metadata`, specifying the path to the underlying video, has to be present for each recording for this to work. 

## 6 Save your data

You can save your data as a pickled DataFrame with
```python
recordings.io.save('outfile.pkl')
```
(and can be loaded again with:)
```python
recordings = pd.DataFrame.io.load('outfile.pkl')
```

NOTE: By importing `ethome` you extend the functionality of the pandas DataFrame, hence can access things like `.io.load`

## 7 Summary and reference list of added functionality by `ethome`

For reference, the metadata and added functions added to the dataframe are:
* `recordings.metadata`, which houses
    * `details`: the metadata dictionary given to create_dataset
    * `videos`: list of videos given in `metadata`
    * `n_videos`: number of videos in DataFrame
    * `label_key`: associates numbers with text labels for each behavior
    * `reverse_label_key`: associates text labels for each behavior number
* `recordings.pose`, houses pose information:
    * `body_parts`: list of body parts loaded from the DLC file(s)
    * `animals`: list of animals loaded from the DLC files(s)
    * `animal_setup`: dictionary detailing animal parts and names
    * `raw_track_columns`: all original columns names loaded from DLC
* `recordings.features`, feature creation and manipulation
    * `activate`: activate columns by name
    * `deactivate`: deactivate columns by name
    * `regex`: select column names based on regex
    * `add`: create new features
* `recordings.ml`, machine learning conveniences
    * `features`: a 'active' set of features
    * `labels`: if loaded from BORIS, behavior labels. from text to label with `recordings.metadata.label_key`
    * `group`: the corresponding video filename for all rows in the table -- can be used for GroupKFold CV, or similar
* `recordings.io`, I/O functions
    * `save`: save DataFrame as pickle file
    * `to_dlc_csv`: save original tracking data back into csv -- if you interpolated or otherwise manipulated the data
    * `load`: load DataFrame from pickle file
    * `save_movie`: create a movie with some feature column you indicate overlaid

See the API docs for usage details.