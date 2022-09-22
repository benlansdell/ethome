<!-- markdownlint-disable -->

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/cnn1d.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `features.cnn1d`




**Global Variables**
---------------
- **has_keras**

---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/cnn1d.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `build_baseline_model`

```python
build_baseline_model(
    input_dim,
    layer_channels=(512, 256),
    dropout_rate=0.0,
    learning_rate=0.001,
    conv_size=5,
    num_classes=4,
    class_weight=None
)
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/cnn1d.py#L62"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_df`

```python
make_df(pts, colnames=None)
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/cnn1d.py#L72"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `features_identity`

```python
features_identity(inputs)
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/cnn1d.py#L75"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `features_via_sklearn`

```python
features_via_sklearn(inputs, featurizer)
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/cnn1d.py#L82"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `features_mars`

```python
features_mars(x)
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/cnn1d.py#L87"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `features_mars_distr`

```python
features_mars_distr(x)
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/cnn1d.py#L90"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `features_distances`

```python
features_distances(inputs)
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/cnn1d.py#L117"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `features_distances_normalized`

```python
features_distances_normalized(inputs)
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/cnn1d.py#L148"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MABe_Generator`




<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/cnn1d.py#L149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    pose_dict,
    batch_size,
    dim,
    use_conv,
    num_classes,
    augment=False,
    class_to_number=None,
    past_frames=0,
    future_frames=0,
    frame_gap=1,
    shuffle=False,
    mode='fit',
    featurize=<function features_identity at 0x7fbe20a39200>
)
```








---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/cnn1d.py#L204"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `augment_fn`

```python
augment_fn(x)
```





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/cnn1d.py#L241"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `on_epoch_end`

```python
on_epoch_end()
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
