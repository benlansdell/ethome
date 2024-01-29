<!-- markdownlint-disable -->

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/interpolation.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `interpolation`





---

<a href="https://github.com/benlansdell/ethome/blob/master/ethome/interpolation.py#L4"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `interpolate_lowconf_points`

```python
interpolate_lowconf_points(
    edf: DataFrame,
    conf_threshold: float = 0.9,
    in_place: bool = True,
    rolling_window: bool = True,
    window_size: int = 3
) → DataFrame
```

Interpolate raw tracking points if their probabilities are available.



**Args:**

 - <b>`edf`</b>:  pandas DataFrame containing the tracks to interpolate
 - <b>`conf_threshold`</b>:  default 0.9. Confidence below which to count as uncertain, and to interpolate its value instead
 - <b>`in_place`</b>:  default True. Whether to replace data in place
 - <b>`rolling_window`</b>:  default True. Whether to use a rolling window to interpolate
 - <b>`window_size`</b>:  default 3. The size of the rolling window to use



**Returns:**
 Pandas dataframe with the filtered raw columns. Returns None if opted for in_place modification




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
