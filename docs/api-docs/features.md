<!-- markdownlint-disable -->

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `features`
Functions to take pose tracks and compute a set of features from them. 

To make your own feature creator: 

Create a function, e.g. `create_custom_features`, and provide the Features class a list of columns that are needed by this function to compute the features. 

The function `create_custom_features` has the form: 

`create_custom_features(<df>, <raw_col_names>, <animal_setup>, **kwargs)` 

Where: 

`df` is the dataframe to compute the features on `raw_col_names` is a list of the names of the columns in the dataframe that contain the raw data used for the feature creation. These are required for the model. `animal_setup` is a dictionary with keys `bodypart_ids`, `mouse_ids`, `colnames`.  `bodypart_ids` is a list of the bodypart ids that are used in the dataframe  `mouse_ids` is a list of the mouse ids that are used in the dataframe  `colnames` is the list product(animals, XY_IDS, body_parts)  `**kwargs` are extra arguments passed onto the feature creation function. 

The function returns: 

A dataframe, that only contains the new features. These will be added to the ExperimentDataFrame as columns. 

Once you have such a function defined, you can create a "feature making object" with 

`custom_feature_maker = Features(create_custom_features, req_columns)` 

This could be used on datasets as: 

```
dataset.add_features(custom_feature_maker, featureset_name = 'CUSTOM', add_to_features = True)
``` 

**Global Variables**
---------------
- **default_tracking_columns**
- **mars_feature_maker**
- **marsreduced_feature_maker**
- **cnn_probability_feature_maker**
- **social_feature_maker**
- **com_interanimal_feature_maker**
- **com_interanimal_speed_feature_maker**
- **com_feature_maker**
- **com_velocity_feature_maker**
- **speed_feature_maker**
- **distance_feature_maker**


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features.py#L58"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Features`




<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features.py#L59"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(feature_maker: Callable, required_columns: list, **kwargs)
```

Feature creation object. This houses the feature creation function and the columns that are required to compute the features. Performs some checks on data to make sure has these columns. 

See docstring for the `features` model for more information. 



**Args:**
 
 - <b>`feature_maker`</b>:  The function that will be used to compute the features. 
 - <b>`required_columns`</b>:  The columns that are required to compute the features. 




---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `make`

```python
make(edf, **kwargs)
```

Make the features. This is called internally by the dataset object when running `add_features`. 



**Args:**
 
 - <b>`edf`</b>:  The ExperimentDataFrame to compute the features on. 
 - <b>`**kwargs`</b>:  Extra arguments passed onto the feature creation function. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
