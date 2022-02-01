<!-- markdownlint-disable -->

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/interpolation.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `interpolation`





---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/interpolation.py#L5"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `interpolate_lowconf_points`

```python
interpolate_lowconf_points(
    vdf: VideosetDataFrame,
    filter_out_lowconf: bool = True,
    filter_out_toofast: bool = True,
    conf_threshold: float = 0.9,
    jump_dur: int = 5,
    speed_threshold: float = 5,
    in_place=True
) → DataFrame
```

Interpolate raw tracking points if their probabilities are available. 



**Args:**
 
 - <b>`vdf`</b>:  VideosetDataFrame containing the tracks to interpolate 
 - <b>`filter_out_lowconf`</b>:  default True. Whether to filter out low confidence points 
 - <b>`filter_out_toofast`</b>:  default True. Whether to filter out tracks that jump too far in a number of frames 
 - <b>`conf_threshold`</b>:  default 0.9. Confidence below which to count as uncertain, and to interpolate its value instead 
 - <b>`jump_dur`</b>:  default 5. Number of frames to compute velocity which is used as basis for filtering out jumps 
 - <b>`speed_threshold`</b>:  default 5. Number of pixels to  
 - <b>`in_place`</b>:  default True. Whether to replace data in place 



**Returns:**
 Pandas dataframe with the filtered raw columns. Returns None if opted for in_place modification 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
