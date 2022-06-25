<!-- markdownlint-disable -->

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/dl/dl_features.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `dl.dl_features`




**Global Variables**
---------------
- **sweeps_baseline**
- **feature_spaces**
- **has_keras**
- **THIS_FILE_DIR**

---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/dl/dl_features.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `seed_everything`

```python
seed_everything(seed=2012)
```






---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/dl/dl_features.py#L144"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `normalize_data`

```python
normalize_data(orig_pose_dictionary)
```






---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/dl/dl_features.py#L154"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `run_task`

```python
run_task(
    vocabulary,
    test_data,
    config_name,
    build_model,
    skip_test_prediction=False,
    seed=2021,
    Generator=<class 'behaveml.dl.dl_generators.MABe_Generator'>,
    use_callbacks=False,
    params=None,
    use_conv=True
)
```






---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/dl/dl_features.py#L220"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `lrs`

```python
lrs(epoch, lr, freq=10)
```






---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/dl/dl_features.py#L225"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `convert_to_mars_format`

```python
convert_to_mars_format(df, colnames, animal_setup)
```






---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/dl/dl_features.py#L238"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `convert_to_pandas_df`

```python
convert_to_pandas_df(data, colnames=None)
```






---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/dl/dl_features.py#L246"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_dl_probability_features`

```python
compute_dl_probability_features(
    df: DataFrame,
    raw_col_names: list,
    animal_setup: dict,
    **kwargs
)
```






---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/dl/dl_features.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Trainer`




<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/dl/dl_features.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
    build_model=<function build_baseline_model at 0x7f77b1efbcb0>,
    Generator=<class 'behaveml.dl.dl_generators.MABe_Generator'>,
    use_callbacks=False,
    learning_decay_freq=10,
    featurizer=<function features_identity at 0x7f78e6c654d0>
)
```








---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/dl/dl_features.py#L77"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `delete_model`

```python
delete_model()
```





---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/dl/dl_features.py#L127"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_test_prediction_probabilities`

```python
get_test_prediction_probabilities()
```





---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/dl/dl_features.py#L80"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `initialize_model`

```python
initialize_model(**kwargs)
```





---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/dl/dl_features.py#L87"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `train`

```python
train(model_params, class_weight=None, n_folds=5)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
