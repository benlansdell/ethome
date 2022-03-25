<!-- markdownlint-disable -->

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/dl/dl_generators.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `dl.dl_generators`




**Global Variables**
---------------
- **has_keras**

---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/dl/dl_generators.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_df`

```python
make_df(pts, colnames=None)
```






---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/dl/dl_generators.py#L29"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `features_identity`

```python
features_identity(inputs)
```






---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/dl/dl_generators.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `features_via_sklearn`

```python
features_via_sklearn(inputs, featurizer)
```






---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/dl/dl_generators.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `features_mars`

```python
features_mars(x)
```






---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/dl/dl_generators.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `features_mars_distr`

```python
features_mars_distr(x)
```






---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/dl/dl_generators.py#L47"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `features_distances`

```python
features_distances(inputs)
```






---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/dl/dl_generators.py#L74"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `features_distances_normalized`

```python
features_distances_normalized(inputs)
```






---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/dl/dl_generators.py#L105"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MABe_Generator`




<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/dl/dl_generators.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
    featurize=<function features_identity at 0x7f00422d4560>
)
```








---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/dl/dl_generators.py#L173"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `augment_fn`

```python
augment_fn(x)
```





---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/dl/dl_generators.py#L217"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `on_epoch_end`

```python
on_epoch_end()
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
