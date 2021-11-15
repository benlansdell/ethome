<!-- markdownlint-disable -->

# API Overview

## Modules

- [`features`](./features.md#module-features): Functions to take pose tracks and compute a set of features from them 
- [`io`](./io.md#module-io): Loading and saving tracking and behavior annotation files 
- [`mars_features`](./mars_features.md#module-mars_features)
- [`ml`](./ml.md#module-ml): Machine learning functions 
- [`video`](./video.md#module-video): Basic video tracking and behavior class that houses data 

## Classes

- [`features.Features`](./features.md#class-features)
- [`io.BufferedIOBase`](./io.md#class-bufferediobase): Base class for buffered IO objects.
- [`io.IOBase`](./io.md#class-iobase): The abstract base class for all I/O classes, acting on streams of
- [`io.RawIOBase`](./io.md#class-rawiobase): Base class for raw binary I/O.
- [`io.TextIOBase`](./io.md#class-textiobase): Base class for text I/O.
- [`io.UnsupportedOperation`](./io.md#class-unsupportedoperation)
- [`video.MLDataFrame`](./video.md#class-mldataframe): DataFrame useful for interfacing between pandas and sklearn. Stores a data
- [`video.VideosetDataFrame`](./video.md#class-videosetdataframe)

## Functions

- [`io.load_data`](./io.md#function-load_data): Load an object from a pickle file
- [`io.read_DLC_tracks`](./io.md#function-read_dlc_tracks): Read in tracks from DLC.
- [`io.read_boris_annotation`](./io.md#function-read_boris_annotation): Read behavior annotation from BORIS exported csv file
- [`io.rename_df_cols`](./io.md#function-rename_df_cols): Rename dataframe columns 
- [`io.save_DLC_tracks_h5`](./io.md#function-save_dlc_tracks_h5): Save DLC tracks in h5 format.
- [`mars_features.compute_mars_features`](./mars_features.md#function-compute_mars_features)
- [`mars_features.compute_mars_features_`](./mars_features.md#function-compute_mars_features_)
- [`mars_features.make_stacked_features`](./mars_features.md#function-make_stacked_features)
- [`video.clone_metadata`](./video.md#function-clone_metadata): Prepare a metadata dictionary for defining a VideosetDataFrame. 


---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
