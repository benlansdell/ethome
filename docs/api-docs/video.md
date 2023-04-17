<!-- markdownlint-disable -->

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `video`
Basic video tracking and behavior class that houses data 

**Global Variables**
---------------
- **global_config**
- **FEATURE_MAKERS**
- **UNIT_DICT**

---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `create_metadata`

```python
create_metadata(tracking_files: list, **kwargs) → dict
```

Prepare a metadata dictionary for defining a ExperimentDataFrame.  

Only required argument is list of pose tracking file names.  

Any other keyword argument must be either a non-iterable object (e.g. a scalar parameter, like FPS) that will be copied and tagged to each of the pose tracking files, or an iterable object of the same length of the list of pose tracking files. Each element in the iterable will be tagged with the corresponding pose-tracking file. 



**Args:**
 
 - <b>`tracking_files`</b>:  List of pose tracking files 
 - <b>`**kwargs`</b>:  described as above 



**Returns:**
 Dictionary whose keys are pose-tracking file names, and contains a dictionary with key,values containing the metadata provided 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L609"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `create_dataset`

```python
create_dataset(
    input: dict = None,
    label_key: dict = None,
    part_renamer: dict = None,
    animal_renamer: dict = None,
    video: list = None,
    labels: list = None,
    **kwargs
) → DataFrame
```

Creates DataFrame that houses pose-tracking data and behavior annotations, along with relevant metadata, features and behavior annotation labels. 



**Args:**
 
 - <b>`input`</b>:  String OR list of strings with path(s) to tracking file(s).   OR Dictionary whose keys are pose tracking files, and value is a dictionary of associated metadata  for that video (see `create_metadata` if using this construction option) 
 - <b>`label_key`</b>:  Default None. Dictionary whose keys are positive integers and values are behavior labels. If none, then this is inferred from the behavior annotation files provided.   
 - <b>`part_renamer`</b>:  Default None. Dictionary that can rename body parts from tracking files if needed (for feature creation, e.g.) 
 - <b>`animal_renamer`</b>:  Default None. Dictionary that can rename animals from tracking files if needed 
 - <b>`**kwargs`</b>:  Any other data to associate with each of the tracking files. This includes label files, and other metadata.   Any list-like arguments of appropriate length are zipped (associated) with each tracking file. See How To guide for more information. 



**Returns:**
 DataFrame object. This is a pandas DataFrame with additional metadata and methods. 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L784"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_experiment`

```python
load_experiment(fn_in: str) → DataFrame
```

Load DataFrame from file. 



**Args:**
 
 - <b>`fn_in`</b>:  path to file to load 



**Returns:**
 DataFrame object from pickle file 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L799"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_sample_openfield_data`

```python
get_sample_openfield_data()
```

Load a sample dataset of 1 mouse in openfield setup. The video is the sample that comes with DLC. 



**Returns:**
  DataFrame with the corresponding tracking and behavior annotation files 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L838"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `add_randomforest_predictions`

```python
add_randomforest_predictions(df: DataFrame)
```

Perform cross validation of a RandomForestClassifier to predict behavior based on  activated features. Can be useful to assess model performance, and if you have enough data. 



**Args:**
 
 - <b>`df`</b>:  Dataframe housing features and labels to perform classification.  Will perform leave-one-video-out cross validation hence dataframe needs at least two videos to run. 



**Returns:**
 None. Modifies df in place, adding column 'prediction' with model predictions in it. 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L164"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `EthologyMetadataAccessor`




<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L165"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(pandas_obj)
```






---

#### <kbd>property</kbd> details





---

#### <kbd>property</kbd> label_key





---

#### <kbd>property</kbd> n_videos





---

#### <kbd>property</kbd> reverse_label_key





---

#### <kbd>property</kbd> videos








---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L201"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `EthologyFeaturesAccessor`




<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L202"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(pandas_obj)
```






---

#### <kbd>property</kbd> active







---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L215"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `activate`

```python
activate(name: str) → list
```

Add already present columns in data frame to the feature set.  



**Args:**
 
 - <b>`name`</b>:  string for pattern matching -- any feature that starts with this string will be added 



**Returns:**
 List of matched columns (may include columns that were already activated). 

---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L262"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add`

```python
add(
    feature_maker,
    featureset_name: str = None,
    add_to_features=True,
    required_columns=[],
    **kwargs
) → list
```

Compute features to dataframe using Feature object. 'featureset_name' will be prepended to new columns, followed by a double underscore.  



**Args:**
 
 - <b>`featuremaker`</b>:  A Feature object that houses the feature-making function to be executed and a list of required columns that must in the dataframe for this to work 
 - <b>`featureset_name`</b>:  Name to prepend to the added features  
 - <b>`add_to_features`</b>:  Whether to add to list of active features (i.e. will be returned by the .features property) 



**Returns:**
 List of new columns that are computed 

---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L248"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `deactivate`

```python
deactivate(name: str) → list
```

Remove columns from the feature set.  



**Args:**
 
 - <b>`name`</b>:  string for pattern matching -- any feature that starts with this string will be removed 



**Returns:**
 List of removed columns. 

---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L323"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `deactivate_cols`

```python
deactivate_cols(col_names: list) → list
```

Remove provided columns from set of feature columns. 



**Args:**
 
 - <b>`col_names`</b>:  list of column names 



**Returns:**
 The columns that were removed from those designated as features. 

---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L232"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `regex`

```python
regex(pattern: str) → list
```

Return a list of column names that match the provided regex pattern. 



**Args:**
 
 - <b>`pattern`</b>:  a regex pattern to match column names to 



**Returns:**
 list of column names 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L338"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `EthologyPoseAccessor`




<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L339"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(pandas_obj)
```






---

#### <kbd>property</kbd> animal_setup





---

#### <kbd>property</kbd> animals





---

#### <kbd>property</kbd> body_parts





---

#### <kbd>property</kbd> raw_track_columns








---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L387"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `EthologyMLAccessor`




<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L388"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(pandas_obj)
```






---

#### <kbd>property</kbd> features





---

#### <kbd>property</kbd> fold_cols





---

#### <kbd>property</kbd> folds





---

#### <kbd>property</kbd> group





---

#### <kbd>property</kbd> label_cols





---

#### <kbd>property</kbd> labels





---

#### <kbd>property</kbd> splitter








---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L453"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `EthologyIOAccessor`




<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L454"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(pandas_obj)
```








---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L506"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load`

```python
load(fn_in: str) → DataFrame
```

Load ExperimentDataFrame object from pickle file. 



**Args:**
 
 - <b>`fn_in`</b>:  path to load pickle file from.  



**Returns:**
 None. Data in this object is populated with contents of file. 

---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L457"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(fn_out: str) → None
```

Save ExperimentDataFrame object with pickle. 



**Args:**
 
 - <b>`fn_out`</b>:  location to write pickle file to 



**Returns:**
 None. File is saved to path. 

---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L517"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_movie`

```python
save_movie(label_columns, path_out: str, video_filenames=None) → None
```

Given columns indicating behavior predictions or whatever else, make a video with these predictions overlaid.  

ExperimentDataFrame metadata must have the keys 'video_file', so that the video associated with each set of pose tracks is known. 



**Args:**
 
 - <b>`label_columns`</b>:  list or dict of columns whose values to overlay on top of video. If dict, keys are the columns and values are the print-friendly version. 
 - <b>`path_out`</b>:  the directory to output the videos too 
 - <b>`video_filenames`</b>:  list or string. The set of videos to use. If not provided, then use all videos as given in the metadata. 



**Returns:**
 None. Videos are saved to 'path_out' 

---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L471"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dlc_csv`

```python
to_dlc_csv(base_dir: str, save_h5_too=False) → None
```

Save ExperimentDataFrame tracking files to DLC csv format. 

Only save tracking data, not other computed features. 



**Args:**
 
 - <b>`base_dir`</b>:  base_dir to write DLC csv files to 
 - <b>`save_h5_too`</b>:  if True, also save the data as an h5 file 



**Returns:**
 None. Files are saved to path. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
