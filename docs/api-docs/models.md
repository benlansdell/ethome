<!-- markdownlint-disable -->

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/models.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `models`
Basic video tracking and behavior class that houses data  



---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/models.py#L66"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `F1Optimizer`




<a href="https://github.com/benlansdell/ethome/blob/master/ethome/models.py#L67"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(N=1000, labels=[1])
```








---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/models.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X, y)
```





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/models.py#L99"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_transform`

```python
fit_transform(X, y=None)
```





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/models.py#L90"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(X)
```





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/models.py#L93"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict_proba`

```python
predict_proba(X)
```





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/models.py#L96"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `transform`

```python
transform(X)
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/models.py#L103"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ModelTransformer`




<a href="https://github.com/benlansdell/ethome/blob/master/ethome/models.py#L104"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(Model, *args, **kwargs)
```

Turns an sklearn model into a model that can be used in a pipeline. Useful for stacking models. Basically, implements `transform` and `fit_transform` as model.predict_prob, without or with `fit` 



**Args:**
 
 - <b>`Model`</b>:  sklearn model to be used for prediction 
 - <b>`args`</b>:  args to be passed to Model.fit() 
 - <b>`kwargs`</b>:  kwargs to be passed to Model.fit() 




---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/models.py#L114"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X, y)
```





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/models.py#L120"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_transform`

```python
fit_transform(X, y=None)
```





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/models.py#L117"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `transform`

```python
transform(X)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
