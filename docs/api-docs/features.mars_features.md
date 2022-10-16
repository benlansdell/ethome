<!-- markdownlint-disable -->

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/mars_features.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `features.mars_features`




**Global Variables**
---------------
- **XY_IDS**

---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/mars_features.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `augment_features`

```python
augment_features(window_size=5, n_shifts=3, mode='shift')
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/mars_features.py#L66"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `boiler_plate`

```python
boiler_plate(features_df)
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/mars_features.py#L299"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_features_distances`

```python
make_features_distances(df, animal_setup)
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/mars_features.py#L340"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_features_mars`

```python
make_features_mars(df, animal_setup, n_shifts=3, mode='shift')
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/mars_features.py#L390"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_features_mars_distr`

```python
make_features_mars_distr(df, animal_setup)
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/mars_features.py#L393"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_features_mars_reduced`

```python
make_features_mars_reduced(df, animal_setup, n_shifts=2, mode='diff')
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/mars_features.py#L432"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_features_velocities`

```python
make_features_velocities(df, animal_setup, n_shifts=5)
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/mars_features.py#L472"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_features_social`

```python
make_features_social(df, animal_setup, n_shifts=3, mode='shift')
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/mars_features.py#L520"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_mars_features`

```python
compute_mars_features(df: DataFrame, raw_col_names: list, **kwargs) → DataFrame
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/mars_features.py#L525"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_distance_features`

```python
compute_distance_features(
    df: DataFrame,
    raw_col_names: list,
    **kwargs
) → DataFrame
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/mars_features.py#L530"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_mars_reduced_features`

```python
compute_mars_reduced_features(
    df: DataFrame,
    raw_col_names: list,
    **kwargs
) → DataFrame
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/mars_features.py#L535"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_social_features`

```python
compute_social_features(
    df: DataFrame,
    raw_col_names: list,
    **kwargs
) → DataFrame
```






---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/mars_features.py#L540"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_velocity_features`

```python
compute_velocity_features(
    df: DataFrame,
    raw_col_names: list,
    **kwargs
) → DataFrame
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
