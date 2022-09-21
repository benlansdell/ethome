<!-- markdownlint-disable -->

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/generic_features.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `features.generic_features`
Functions to take pose tracks and compute a set of features from them  


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/generic_features.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_centerofmass_interanimal_distances`

```python
compute_centerofmass_interanimal_distances(
    df: DataFrame,
    raw_col_names: list,
    **kwargs
) → DataFrame
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/generic_features.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_centerofmass_interanimal_speed`

```python
compute_centerofmass_interanimal_speed(
    df: DataFrame,
    raw_col_names: list,
    n_shifts=5,
    **kwargs
) → DataFrame
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/generic_features.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_centerofmass`

```python
compute_centerofmass(
    df: DataFrame,
    raw_col_names: list,
    bodyparts: list = [],
    **kwargs
) → DataFrame
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/generic_features.py#L112"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_centerofmass_velocity`

```python
compute_centerofmass_velocity(
    df: DataFrame,
    raw_col_names: list,
    n_shifts=5,
    bodyparts: list = [],
    **kwargs
) → DataFrame
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/generic_features.py#L144"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_speed_features`

```python
compute_speed_features(
    df: DataFrame,
    raw_col_names: list,
    n_shifts=5,
    **kwargs
) → DataFrame
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/generic_features.py#L193"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_distance_features`

```python
compute_distance_features(
    df: DataFrame,
    raw_col_names: list,
    **kwargs
) → DataFrame
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
