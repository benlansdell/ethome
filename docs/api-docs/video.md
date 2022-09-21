<!-- markdownlint-disable -->

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `video`
Basic video tracking and behavior class that houses data.  

Basic object is the ExperimentDataFrame class. 

## A note on unit conversions 

For the unit rescaling, if the dlc/tracking file is already in desired units, either in physical distances, or pixels, then don't provide all of 'frame_width', 'resolution', and 'frame_width_units'. If you want to keep track of the units, you can add a 'units' key to the metadata. This could be 'pixels', or 'cm', as appropriate. 

If the tracking is in pixels and you do want to rescale it to some physical distance, you should provide 'frame_width', 'frame_width_units' and 'resolution' for all videos. This ensures the entire dataset is using the same units. The package will use these values for each video to rescale the (presumed) pixel coordinates to physical coordinates.  

Resolution is a tuple (H,W) in pixels of the videos. 'frame_width' is the width of the image, in units 'frame_width_units' 

When this is done, all coordinates are converted to 'mm'. The pair 'units':'mm' is added to the metadata dictionary for each video 

If any of the provided parameters are provided, but are not the right format, or some values are missing, a warning is given and the rescaling is not performed. 

**Global Variables**
---------------
- **global_config**
- **FEATURE_MAKERS**
- **UNIT_DICT**

---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L47"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `clone_metadata`

```python
clone_metadata(tracking_files: list, **kwargs) → dict
```

Prepare a metadata dictionary for defining a ExperimentDataFrame.  

Only required argument is list of DLC tracking file names.  

Any other keyword argument must be either a non-iterable object (e.g. a scalar parameter, like FPS) that will be copied and tagged to each of the DLC tracking files, or an iterable object of the same length of the list of DLC tracking files. Each element in the iterable will be tagged with the corresponding DLC file. 



**Args:**
 
 - <b>`tracking_files`</b>:  List of DLC tracking .csvs 
 - <b>`**kwargs`</b>:  described as above 



**Returns:**
 Dictionary whose keys are DLC tracking file names, and contains a dictionary with key,values containing the metadata provided 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L459"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `create_experiment`

```python
create_experiment(
    metadata: dict,
    label_key: dict = None,
    part_renamer: dict = None,
    animal_renamer: dict = None
) → DataFrame
```

Houses DLC tracking data and behavior annotations in pandas DataFrame for ML, along with relevant metadata, features and behavior annotation labels. 



**Args:**
 
 - <b>`metadata`</b>:  Dictionary whose keys are DLC tracking csvs, and value is a dictionary of associated metadata  for that video. Most easiest to create with 'clone_metadata'.  
 - <b>`Required keys are`</b>:  ['fps'] 
 - <b>`label_key`</b>:  Default None. Dictionary whose keys are positive integers and values are behavior labels. If none, then this is inferred from the behavior annotation files provided.   
 - <b>`part_renamer`</b>:  Default None. Dictionary that can rename body parts from tracking files if needed (for feature creation, e.g.) 
 - <b>`animal_renamer`</b>:  Default None. Dictionary that can rename animals from tracking files if needed 



**Returns:**
 DataFrame object. This is a pandas DataFrame with additional metadata and methods. 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L576"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L591"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_sample_openfield_data`

```python
get_sample_openfield_data()
```

Load a sample dataset of 1 mouse in openfield setup. The video is the sample that comes with DLC. 



**Returns:**
  DataFrame with the corresponding tracking and behavior annotation files 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L147"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `EthologyMetadataAccessor`




<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L148"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(pandas_obj)
```






---

#### <kbd>property</kbd> n_videos





---

#### <kbd>property</kbd> reverse_label_key





---

#### <kbd>property</kbd> videos








---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L166"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `EthologyFeaturesAccessor`




<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(pandas_obj)
```








---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L171"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L218"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L204"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L269"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L188"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L284"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `EthologyPoseAccessor`




<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L285"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(pandas_obj)
```









---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L293"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `EthologyMLAccessor`




<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L294"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(pandas_obj)
```






---

#### <kbd>property</kbd> features





---

#### <kbd>property</kbd> folds





---

#### <kbd>property</kbd> group





---

#### <kbd>property</kbd> labels





---

#### <kbd>property</kbd> splitter








---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L339"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `EthologyIOAccessor`




<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L340"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(pandas_obj)
```








---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L392"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L343"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L403"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_movie`

```python
save_movie(label_columns, path_out: str, video_filenames=None) → None
```

Given columns indicating behavior predictions or whatever else, make a video with these predictions overlaid.  

ExperimentDataFrame metadata must have the keys 'video_file', so that the video associated with each set of DLC tracks is known. 



**Args:**
 
 - <b>`label_columns`</b>:  list or dict of columns whose values to overlay on top of video. If dict, keys are the columns and values are the print-friendly version. 
 - <b>`path_out`</b>:  the directory to output the videos too 
 - <b>`video_filenames`</b>:  list or string. The set of videos to use. If not provided, then use all videos as given in the metadata. 



**Returns:**
 None. Videos are saved to 'path_out' 

---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/video.py#L357"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
