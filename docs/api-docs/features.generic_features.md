<!-- markdownlint-disable -->

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/generic_features.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `features.generic_features`
Functions to take pose tracks and compute a set of features from them  


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/generic_features.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_centerofmass_interanimal_distances`

```python
compute_centerofmass_interanimal_distances(
    df: DataFrame,
    raw_col_names: list,
    **kwargs
) → DataFrame
```

Distances between all animals' centroids 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/generic_features.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_centerofmass_interanimal_speed`

```python
compute_centerofmass_interanimal_speed(
    df: DataFrame,
    raw_col_names: list,
    n_shifts=5,
    **kwargs
) → DataFrame
```

Speeds between all animals' centroids 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/generic_features.py#L78"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_centerofmass`

```python
compute_centerofmass(
    df: DataFrame,
    raw_col_names: list,
    bodyparts: list = [],
    **kwargs
) → DataFrame
```

Centroid of all animals 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/generic_features.py#L104"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

Velocity of all animals' centroids 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/generic_features.py#L135"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_part_velocity`

```python
compute_part_velocity(
    df: DataFrame,
    raw_col_names: list,
    n_shifts=5,
    bodyparts: list = [],
    **kwargs
) → DataFrame
```

Velocity of all animals' bodyparts 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/generic_features.py#L165"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_part_speed`

```python
compute_part_speed(
    df: DataFrame,
    raw_col_names: list,
    n_shifts=5,
    bodyparts: list = [],
    **kwargs
) → DataFrame
```

Speed of all animals' bodyparts 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/generic_features.py#L195"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_speed_features`

```python
compute_speed_features(
    df: DataFrame,
    raw_col_names: list,
    n_shifts=5,
    **kwargs
) → DataFrame
```

Speeds between all body parts pairs (within and between animals) 


---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/features/generic_features.py#L244"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compute_distance_features`

```python
compute_distance_features(
    df: DataFrame,
    raw_col_names: list,
    **kwargs
) → DataFrame
```

Distances between all body parts pairs (within and between animals) 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
