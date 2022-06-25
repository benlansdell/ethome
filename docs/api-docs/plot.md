<!-- markdownlint-disable -->

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/plot.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `plot`




**Global Variables**
---------------
- **global_config**

---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/plot.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_embedding`

```python
plot_embedding(
    dataset: VideosetDataFrame,
    col_names: list = ['embedding_0', 'embedding_1'],
    color_col: str = None,
    figsize: tuple = (10, 10),
    **kwargs
) → tuple
```

Scatterplot of a 2D TSNE or UMAP embedding from the dataset. 



**Args:**
 
 - <b>`dataset`</b>:  data 
 - <b>`col_names`</b>:  list of column names to use for the x and y axes 
 - <b>`color_col`</b>:  if provided, a column that will be used to color the points in the scatter plot 
 - <b>`figsize`</b>:  tuple with the dimensions of the plot (in inches) 
 - <b>`kwargs`</b>:  All other keyword pairs are sent to Matplotlib's scatter function 



**Returns:**
 tuple (fig, axes). The Figure and Axes objects.  


---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/plot.py#L54"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_unsupervised_results`

```python
plot_unsupervised_results(
    dataset: VideosetDataFrame,
    cluster_results: tuple,
    col_names: list = ['embedding_0', 'embedding_1'],
    figsize: tuple = (15, 4),
    **kwargs
)
```

Set of plots for unsupervised behavior clustering results 



**Args:**
 
 - <b>`dataset`</b>:  data 
 - <b>`cluster_results`</b>:  tuple output by 'cluster_behaviors' 
 - <b>`col_names`</b>:  list of column names to use for the x and y axes 
 - <b>`figsize`</b>:  tuple with the plot dimensions, in inches 
 - <b>`kwargs`</b>:  all other keyword pairs are sent to Matplotlib's scatter function 



**Returns:**
 tuple (fig, axes). The Figure and Axes objects.  


---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/plot.py#L104"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_ethogram`

```python
plot_ethogram(
    dataset: VideosetDataFrame,
    vid_key: str,
    query_label: str = 'unsup_behavior_label',
    frame_limit: int = 4000,
    figsize: tuple = (16, 2)
) → tuple
```

Simple ethogram of one video, up to a certain frame number. 



**Args:**
  dataset:  
 - <b>`vid_key`</b>:  key (in dataset.metadata) pointing to the video to make ethogram for 
 - <b>`query_label`</b>:  the column containing the behavior labels to plot 
 - <b>`frame_limit`</b>:  only make the ethogram for frames between [0, frame_limit] 
 - <b>`figsize`</b>:  tuple with figure size (in inches) 



**Returns:**
 tuple (fig, axes). The Figure and Axes objects 


---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/plot.py#L133"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `create_ethogram_video`

```python
create_ethogram_video(
    dataset: VideosetDataFrame,
    vid_key: str,
    query_label: str,
    out_file: str,
    frame_limit: int = 4000,
    im_dim: float = 16,
    min_frames: int = 3
) → None
```

Overlay ethogram on top of source video with ffmpeg 



**Args:**
 
 - <b>`dataset`</b>:  source dataset 
 - <b>`vid_key`</b>:  the key (in dataset.metadata) pointing to the video to make ethogram for. metadata must have field 'video_files' that points to the source video location 
 - <b>`query_label`</b>:  the column containing the behavior labels to plot 
 - <b>`out_file`</b>:  output path for created video 
 - <b>`frame_limit`</b>:  only make the ethogram/video for frames [0, frame_limit] 
 - <b>`in_dim`</b>:  x dimension (in inches) of ethogram 
 - <b>`min_frames`</b>:  any behaviors occurring for less than this number of frames are not labeled 



**Returns:**
 None 


---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/plot.py#L203"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `create_sample_videos`

```python
create_sample_videos(
    dataset: VideosetDataFrame,
    video_dir: str,
    out_dir: str,
    query_col: str = 'unsup_behavior_label',
    N_sample_rows: int = 16,
    window_size: int = 2,
    fps: float = 30,
    N_supersample_rows: int = 1000
) → None
```

Create a sample of videos displaying the labeled behaviors using ffmpeg.  

For each behavior label, randomly choose frames from the entire dataset and extract short clips from source videos based around those points. Tries to select frames where the labeled behavior is exhibited in many frames of the clip. 



**Args:**
 
 - <b>`dataset`</b>:  source dataset 
 - <b>`video_dir`</b>:  location of source video files 
 - <b>`out_dir`</b>:  base output directory to save videos. Videos are saved in the form: [out_dir]/[behavior_label]/[video_name]_[time in seconds].avi 
 - <b>`query_label`</b>:  the column containing the behavior labels to extract clips for. Each unique value in this column is treated as a separate behavior 
 - <b>`N_sample_rows`</b>:  number of clips to extract per behavior 
 - <b>`window_size`</b>:  amount of video to extract on either side of the sampled frame, in seconds 
 - <b>`fps`</b>:  frames per second of videos 
 - <b>`N_supersample_rows`</b>:  this many rows are randomly sampled for each behavior label, and the top N_sample_rows are returned (in terms of number of adjacent frames also exhibiting that behavior). Shouldn't need to play with this. 



**Returns:**
 None 


---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/plot.py#L304"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `create_mosaic_video`

```python
create_mosaic_video(
    vid_dir: str,
    output_file: str,
    ndim: tuple = (1600, 1200)
) → None
```

Take a set of video clips and turn them into a mosaic using ffmpeg  

16 videos are tiled. 



**Args:**
 
 - <b>`vid_dir`</b>:  source directory with videos in it 
 - <b>`output_file`</b>:  output video path 
 - <b>`ndim`</b>:  tuple with the output video dimensions, in pixels 



**Returns:**
 None     


---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/plot.py#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MplColorHelper`




<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/plot.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(cmap_name, start_val, stop_val)
```








---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/plot.py#L50"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_rgb`

```python
get_rgb(val)
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
