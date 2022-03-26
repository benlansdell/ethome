<!-- markdownlint-disable -->

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/video.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `video`
Basic video tracking and behavior class that houses data  

**Global Variables**
---------------
- **global_config**
- **UNIT_DICT**

---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/video.py#L110"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `clone_metadata`

```python
clone_metadata(tracking_files: list, **kwargs) → dict
```

Prepare a metadata dictionary for defining a VideosetDataFrame.  

Only required argument is list of DLC tracking file names.  

Any other keyword argument must be either a non-iterable object (e.g. a scalar parameter, like FPS) that will be copied and tagged to each of the DLC tracking files, or an iterable object of the same length of the list of DLC tracking files. Each element in the iterable will be tagged with the corresponding DLC file. 



**Args:**
 
 - <b>`tracking_files`</b>:  List of DLC tracking .csvs 
 - <b>`**kwargs`</b>:  described as above 



**Returns:**
 Dictionary whose keys are DLC tracking file names, and contains a dictionary with key,values containing the metadata provided 


---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/video.py#L506"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_videodataset`

```python
load_videodataset(fn_in: str) → VideosetDataFrame
```

Load VideosetDataFrame from file. 



**Args:**
 
 - <b>`fn_in`</b>:  path to file to load 



**Returns:**
 VideosetDataFrame object from pickle file 


---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/video.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MLDataFrame`
DataFrame useful for interfacing between pandas and sklearn. Stores a data table and metadata dictionary. When feature columns, label columns and fold columns are specified then creates properties features, labels, folds and splitter that sklearn accepts for ML. 

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/video.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    data: DataFrame,
    metadata: dict = {},
    fold_cols=None,
    feature_cols=None,
    label_cols=None
)
```






---

#### <kbd>property</kbd> features





---

#### <kbd>property</kbd> folds





---

#### <kbd>property</kbd> labels





---

#### <kbd>property</kbd> splitter







---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/video.py#L55"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_data`

```python
add_data(new_data, col_names)
```





---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/video.py#L95"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(fn)
```






---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/video.py#L146"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `VideosetDataFrame`




<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/video.py#L147"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    metadata: dict,
    label_key: dict = None,
    part_renamer: dict = None,
    animal_renamer: dict = None
)
```

Houses DLC tracking data and behavior annotations in pandas DataFrame for ML, along with relevant metadata, features and behavior annotation labels. 



**Args:**
 
 - <b>`metadata`</b>:  Dictionary whose keys are DLC tracking csvs, and value is a dictionary of associated metadata  for that video. Most easiest to create with 'clone_metadata'.  
 - <b>`Required keys are`</b>:  ['fps'] 
 - <b>`label_key`</b>:  Default None. Dictionary whose keys are behavior labels and values are integers  
 - <b>`part_renamer`</b>:  Default None. Dictionary that can rename body parts from tracking files if needed (for feature creation, e.g.) 
 - <b>`animal_renamer`</b>:  Default None. Dictionary that can rename animals from tracking files if needed 


---

#### <kbd>property</kbd> features





---

#### <kbd>property</kbd> folds





---

#### <kbd>property</kbd> group





---

#### <kbd>property</kbd> labels





---

#### <kbd>property</kbd> n_videos





---

#### <kbd>property</kbd> splitter





---

#### <kbd>property</kbd> videos







---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/video.py#L271"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `activate_features_by_name`

```python
activate_features_by_name(name: str) → list
```

Add already present columns in data frame to the feature set.  



**Args:**
 
 - <b>`name`</b>:  string for pattern matching -- any feature that starts with this string will be added 



**Returns:**
 List of matched columns (may include columns that were already activated). 

---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/video.py#L55"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_data`

```python
add_data(new_data, col_names)
```





---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/video.py#L318"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_features`

```python
add_features(
    feature_maker: Features,
    featureset_name: str,
    add_to_features=False,
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

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/video.py#L288"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_columns_regex`

```python
get_columns_regex(pattern: str) → list
```

Return a list of column names that match the provided regex pattern. 



**Args:**
 
 - <b>`pattern`</b>:  a regex pattern to match column names to 



**Returns:**
 list of column names 

---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/video.py#L444"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load`

```python
load(fn_in: str) → None
```

Load VideosetDataFrame object from pickle file. 



**Args:**
 
 - <b>`fn_in`</b>:  path to load pickle file from.  



**Returns:**
 None. Data in this object is populated with contents of file. 

---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/video.py#L456"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `make_movie`

```python
make_movie(label_columns, path_out: str, video_filenames=None) → None
```

Given columns indicating behavior predictions or whatever else, make a video with these predictions overlaid.  

VideosetDataFrame must have the keys 'video_file', so that the video associated with each set of DLC tracks is known. 



**Args:**
 
 - <b>`label_columns`</b>:  list or dict of columns whose values to overlay on top of video. If dict, keys are the columns and values are the print-friendly version. 
 - <b>`path_out`</b>:  the directory to output the videos too 
 - <b>`video_filenames`</b>:  list or string. The  



**Returns:**
 None. Videos are saved to 'path_out' 

---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/video.py#L351"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `remove_feature_cols`

```python
remove_feature_cols(col_names: list) → list
```

Remove provided columns from set of feature columns. 



**Args:**
 
 - <b>`col_names`</b>:  list of column names 



**Returns:**
 The columns that were removed from those designated as features. 

---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/video.py#L304"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `remove_features_by_name`

```python
remove_features_by_name(name: str) → list
```

Remove columns from the feature set.  



**Args:**
 
 - <b>`name`</b>:  string for pattern matching -- any feature that starts with this string will be removed 



**Returns:**
 List of removed columns. 

---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/video.py#L432"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(fn_out: str) → None
```

Save VideosetDataFrame object with pickle. 



**Args:**
 
 - <b>`fn_out`</b>:  location to write pickle file to 



**Returns:**
 None. File is saved to path. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
