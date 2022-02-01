<!-- markdownlint-disable -->

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/plot.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `plot`




**Global Variables**
---------------
- **global_config**

---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/plot.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_embedding`

```python
plot_embedding(dataset, figsize=(10, 10))
```

Plot a 2D TSNE or UMAP embedding from the dataset 


---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/plot.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `plot_ethogram`

```python
plot_ethogram(
    dataset,
    vid_key,
    query_label='unsup_behavior_label',
    frame_limit=4000,
    figsize=(16, 2)
)
```






---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/plot.py#L29"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `create_ethogram_video`

```python
create_ethogram_video(
    dataset,
    vid_key,
    query_label,
    out_file,
    frame_limit=4000,
    im_dim=16,
    min_frames=3
)
```






---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/plot.py#L77"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `create_sample_videos`

```python
create_sample_videos(
    dataset,
    video_dir,
    out_dir,
    query_col='unsup_behavior_label',
    N_sample_rows=16,
    window_size=2,
    fps=30
)
```






---

<a href="https://github.com/benlansdell/behaveml/blob/master/behaveml/plot.py#L111"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `create_mosaic_video`

```python
create_mosaic_video(vid_dir, output_file, ndim=('1600', '1200'))
```








---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
