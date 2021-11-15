<!-- markdownlint-disable -->

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/video.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `video`
Basic video tracking and behavior class that houses data  

**Global Variables**
---------------
- **XY_IDS**

---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/video.py#L85"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `clone_metadata`

```python
clone_metadata(tracking_files:list, **kwargs) → dict
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

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/video.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MLDataFrame`
DataFrame useful for interfacing between pandas and sklearn. Stores a data table and metadata dictionary. When feature columns, label columns and fold columns are specified then creates properties features, labels, folds and splitter that sklearn accepts for ML. 

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/video.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    data:DataFrame,
    metadata:dict={},
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

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/video.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_data`

```python
add_data(new_data, col_names)
```





---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/video.py#L70"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(fn)
```






---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/video.py#L121"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `VideosetDataFrame`




<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/video.py#L122"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    metadata:dict,
    label_key:dict=None,
    part_renamer:dict=None,
    animal_renamer:dict=None
)
```

Houses DLC tracking data and behavior annotations in pandas DataFrame for ML, along with relevant metadata, features and behavior annotation labels. 



**Args:**
 
 - <b>`metadata`</b>:  Dictionary whose keys are DLC tracking csvs, and value is a dictionary of associated metadata  for that video. Most easiest to create with 'clone_metadata'.  
 - <b>`Required keys are`</b>:  ['scale', 'fps', 'units', 'resolution', 'label_files'] 
 - <b>`label_key`</b>:  Default None. Dictionary whose keys are behavior labels and values are integers  
 - <b>`part_renamer`</b>:  Default None. Dictionary that can rename body parts from tracking files if needed (for feature creation, e.g.) 
 - <b>`animal_renamer`</b>:  Default None. Dictionary that re rename animals from tracking files if needed 


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

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/video.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_data`

```python
add_data(new_data, col_names)
```





---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/video.py#L186"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_features`

```python
add_features(
    feature_maker:Features,
    featureset_name:str,
    add_to_features=False,
    **kwargs
)
```

Houses DLC tracking data and behavior annotations in pandas DataFrame for ML, along with relevant metadata 



**Args:**
 
 - <b>`featuremaker`</b>:  (dict) Dictionary whose keys are DLC tracking csvs, and value is a dictionary of associated metadata  for that video. Most easiest to create with 'clone_metadata'.  
 - <b>`Required keys are`</b>:  ['scale', 'fps', 'units', 'resolution', 'label_files'] 
 - <b>`label_key`</b>:  (dict) Default None. Dictionary whose keys are behavior labels and values are integers  
 - <b>`part_renamer`</b>:  (dict) Default None. Dictionary that can rename body parts from tracking files if needed (for feature creation, e.g.) 

**Returns:**
 None 

---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/video.py#L271"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `make_movie`

```python
make_movie(prediction_column, fn_out, movie_in)
```

Given a column indicating behavior predictions, make a video outputting those predictiions alongside true labels. 

---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/video.py#L216"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `remove_feature_cols`

```python
remove_feature_cols(col_names)
```





---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/video.py#L70"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(fn)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
