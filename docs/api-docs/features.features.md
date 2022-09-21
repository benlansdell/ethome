<!-- markdownlint-disable -->

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `features.features`
Functions to take pose tracks and compute a set of features from them. 

**Global Variables**
---------------
- **default_tracking_columns**
- **FEATURE_MAKERS**

---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `feature_class_maker`

```python
feature_class_maker(name, compute_function, required_columns=[])
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CNN1DProb`




<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(required_columns=None, **kwargs)
```

Feature creation object. This houses the feature creation function and the columns that are required to compute the features. Performs some checks on data to make sure has these columns. 

See docstring for the `features` model for more information. 



**Args:**
 
 - <b>`required_columns`</b>:  The columns that are required to compute the features. 




---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `fit`

```python
fit(edf, **kwargs)
```





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `fit_transform`

```python
fit_transform(edf, **kwargs)
```





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `transform`

```python
transform(edf, **kwargs)
```

Make the features. This is called internally by the dataset object when running `add_features`. 



**Args:**
 
 - <b>`edf`</b>:  The ExperimentDataFrame to compute the features on. 
 - <b>`**kwargs`</b>:  Extra arguments passed onto the feature creation function. 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Centroid`




<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(required_columns=None, **kwargs)
```

Feature creation object. This houses the feature creation function and the columns that are required to compute the features. Performs some checks on data to make sure has these columns. 

See docstring for the `features` model for more information. 



**Args:**
 
 - <b>`required_columns`</b>:  The columns that are required to compute the features. 




---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `fit`

```python
fit(edf, **kwargs)
```





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `fit_transform`

```python
fit_transform(edf, **kwargs)
```





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `transform`

```python
transform(edf, **kwargs)
```

Make the features. This is called internally by the dataset object when running `add_features`. 



**Args:**
 
 - <b>`edf`</b>:  The ExperimentDataFrame to compute the features on. 
 - <b>`**kwargs`</b>:  Extra arguments passed onto the feature creation function. 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CentroidInteranimal`




<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(required_columns=None, **kwargs)
```

Feature creation object. This houses the feature creation function and the columns that are required to compute the features. Performs some checks on data to make sure has these columns. 

See docstring for the `features` model for more information. 



**Args:**
 
 - <b>`required_columns`</b>:  The columns that are required to compute the features. 




---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `fit`

```python
fit(edf, **kwargs)
```





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `fit_transform`

```python
fit_transform(edf, **kwargs)
```





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `transform`

```python
transform(edf, **kwargs)
```

Make the features. This is called internally by the dataset object when running `add_features`. 



**Args:**
 
 - <b>`edf`</b>:  The ExperimentDataFrame to compute the features on. 
 - <b>`**kwargs`</b>:  Extra arguments passed onto the feature creation function. 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CentroidInteranimalSpeed`




<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(required_columns=None, **kwargs)
```

Feature creation object. This houses the feature creation function and the columns that are required to compute the features. Performs some checks on data to make sure has these columns. 

See docstring for the `features` model for more information. 



**Args:**
 
 - <b>`required_columns`</b>:  The columns that are required to compute the features. 




---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `fit`

```python
fit(edf, **kwargs)
```





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `fit_transform`

```python
fit_transform(edf, **kwargs)
```





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `transform`

```python
transform(edf, **kwargs)
```

Make the features. This is called internally by the dataset object when running `add_features`. 



**Args:**
 
 - <b>`edf`</b>:  The ExperimentDataFrame to compute the features on. 
 - <b>`**kwargs`</b>:  Extra arguments passed onto the feature creation function. 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CentroidVelocity`




<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(required_columns=None, **kwargs)
```

Feature creation object. This houses the feature creation function and the columns that are required to compute the features. Performs some checks on data to make sure has these columns. 

See docstring for the `features` model for more information. 



**Args:**
 
 - <b>`required_columns`</b>:  The columns that are required to compute the features. 




---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `fit`

```python
fit(edf, **kwargs)
```





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `fit_transform`

```python
fit_transform(edf, **kwargs)
```





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `transform`

```python
transform(edf, **kwargs)
```

Make the features. This is called internally by the dataset object when running `add_features`. 



**Args:**
 
 - <b>`edf`</b>:  The ExperimentDataFrame to compute the features on. 
 - <b>`**kwargs`</b>:  Extra arguments passed onto the feature creation function. 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Distances`




<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(required_columns=None, **kwargs)
```

Feature creation object. This houses the feature creation function and the columns that are required to compute the features. Performs some checks on data to make sure has these columns. 

See docstring for the `features` model for more information. 



**Args:**
 
 - <b>`required_columns`</b>:  The columns that are required to compute the features. 




---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `fit`

```python
fit(edf, **kwargs)
```





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `fit_transform`

```python
fit_transform(edf, **kwargs)
```





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `transform`

```python
transform(edf, **kwargs)
```

Make the features. This is called internally by the dataset object when running `add_features`. 



**Args:**
 
 - <b>`edf`</b>:  The ExperimentDataFrame to compute the features on. 
 - <b>`**kwargs`</b>:  Extra arguments passed onto the feature creation function. 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MARSFeatures`




<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(required_columns=None, **kwargs)
```

Feature creation object. This houses the feature creation function and the columns that are required to compute the features. Performs some checks on data to make sure has these columns. 

See docstring for the `features` model for more information. 



**Args:**
 
 - <b>`required_columns`</b>:  The columns that are required to compute the features. 




---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `fit`

```python
fit(edf, **kwargs)
```





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `fit_transform`

```python
fit_transform(edf, **kwargs)
```





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `transform`

```python
transform(edf, **kwargs)
```

Make the features. This is called internally by the dataset object when running `add_features`. 



**Args:**
 
 - <b>`edf`</b>:  The ExperimentDataFrame to compute the features on. 
 - <b>`**kwargs`</b>:  Extra arguments passed onto the feature creation function. 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MARSReduced`




<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(required_columns=None, **kwargs)
```

Feature creation object. This houses the feature creation function and the columns that are required to compute the features. Performs some checks on data to make sure has these columns. 

See docstring for the `features` model for more information. 



**Args:**
 
 - <b>`required_columns`</b>:  The columns that are required to compute the features. 




---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `fit`

```python
fit(edf, **kwargs)
```





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `fit_transform`

```python
fit_transform(edf, **kwargs)
```





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `transform`

```python
transform(edf, **kwargs)
```

Make the features. This is called internally by the dataset object when running `add_features`. 



**Args:**
 
 - <b>`edf`</b>:  The ExperimentDataFrame to compute the features on. 
 - <b>`**kwargs`</b>:  Extra arguments passed onto the feature creation function. 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Social`




<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(required_columns=None, **kwargs)
```

Feature creation object. This houses the feature creation function and the columns that are required to compute the features. Performs some checks on data to make sure has these columns. 

See docstring for the `features` model for more information. 



**Args:**
 
 - <b>`required_columns`</b>:  The columns that are required to compute the features. 




---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `fit`

```python
fit(edf, **kwargs)
```





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `fit_transform`

```python
fit_transform(edf, **kwargs)
```





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `transform`

```python
transform(edf, **kwargs)
```

Make the features. This is called internally by the dataset object when running `add_features`. 



**Args:**
 
 - <b>`edf`</b>:  The ExperimentDataFrame to compute the features on. 
 - <b>`**kwargs`</b>:  Extra arguments passed onto the feature creation function. 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Speeds`




<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `__init__`

```python
__init__(required_columns=None, **kwargs)
```

Feature creation object. This houses the feature creation function and the columns that are required to compute the features. Performs some checks on data to make sure has these columns. 

See docstring for the `features` model for more information. 



**Args:**
 
 - <b>`required_columns`</b>:  The columns that are required to compute the features. 




---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `fit`

```python
fit(edf, **kwargs)
```





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `fit_transform`

```python
fit_transform(edf, **kwargs)
```





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `transform`

```python
transform(edf, **kwargs)
```

Make the features. This is called internally by the dataset object when running `add_features`. 



**Args:**
 
 - <b>`edf`</b>:  The ExperimentDataFrame to compute the features on. 
 - <b>`**kwargs`</b>:  Extra arguments passed onto the feature creation function. 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Features`




<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L26"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```








---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/features.py#L29"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `transform`

```python
transform(df)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
