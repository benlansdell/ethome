<!-- markdownlint-disable -->

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/dl_features.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `features.dl_features`




**Global Variables**
---------------
- **has_keras**
- **THIS_FILE_DIR**
- **default_config**
- **feature_spaces**
- **sweeps_baseline**

---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/dl_features.py#L61"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `seed_everything`

```python
seed_everything(seed=2012)
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/dl_features.py#L183"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `normalize_data`

```python
normalize_data(orig_pose_dictionary)
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/dl_features.py#L193"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `run_task`

```python
run_task(
    vocabulary,
    test_data,
    config_name,
    build_model,
    skip_test_prediction=False,
    seed=2021,
    Generator=<class 'features.cnn1d.MABe_Generator'>,
    use_callbacks=False,
    params=None,
    use_conv=True
)
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/dl_features.py#L259"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `lrs`

```python
lrs(epoch, lr, freq=10)
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/dl_features.py#L264"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `convert_to_mars_format`

```python
convert_to_mars_format(df, colnames, animal_setup)
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/dl_features.py#L277"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `convert_to_pandas_df`

```python
convert_to_pandas_df(data, colnames=None)
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/dl_features.py#L285"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_dl_probability_features`

```python
compute_dl_probability_features(df: DataFrame, raw_col_names: list, **kwargs)
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/dl_features.py#L69"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Trainer`




<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/dl_features.py#L70"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    feature_dim,
    num_classes,
    test_data=None,
    class_to_number=None,
    past_frames=0,
    future_frames=0,
    frame_gap=1,
    use_conv=False,
    build_model=<function build_baseline_model at 0x7fbe61fb2050>,
    Generator=<class 'features.cnn1d.MABe_Generator'>,
    use_callbacks=False,
    learning_decay_freq=10,
    featurizer=<function features_identity at 0x7fbe20a39200>
)
```








---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/dl_features.py#L116"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `delete_model`

```python
delete_model()
```





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/dl_features.py#L166"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_test_prediction_probabilities`

```python
get_test_prediction_probabilities()
```





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/dl_features.py#L119"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `initialize_model`

```python
initialize_model(**kwargs)
```





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/dl_features.py#L126"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `train`

```python
train(model_params, class_weight=None, n_folds=5)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
