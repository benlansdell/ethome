<!-- markdownlint-disable -->

# API Overview

## Modules

- [`interpolation`](./interpolation.md#module-interpolation)
- [`io`](./io.md#module-io): Loading and saving tracking and behavior annotation files 
- [`utils`](./utils.md#module-utils): Small helper utilities
- [`video`](./video.md#module-video): Basic video tracking and behavior class that houses data

## Classes

- [`io.BufferedIOBase`](./io.md#class-bufferediobase): Base class for buffered IO objects.
- [`io.IOBase`](./io.md#class-iobase): The abstract base class for all I/O classes, acting on streams of
- [`io.RawIOBase`](./io.md#class-rawiobase): Base class for raw binary I/O.
- [`io.TextIOBase`](./io.md#class-textiobase): Base class for text I/O.
- [`io.UnsupportedOperation`](./io.md#class-unsupportedoperation)
- [`video.EthologyFeaturesAccessor`](./video.md#class-ethologyfeaturesaccessor)
- [`video.EthologyIOAccessor`](./video.md#class-ethologyioaccessor)
- [`video.EthologyMLAccessor`](./video.md#class-ethologymlaccessor)
- [`video.EthologyMetadataAccessor`](./video.md#class-ethologymetadataaccessor)
- [`video.EthologyPoseAccessor`](./video.md#class-ethologyposeaccessor)

## Functions

- [`interpolation.interpolate_lowconf_points`](./interpolation.md#function-interpolate_lowconf_points): Interpolate raw tracking points if their probabilities are available.
- [`io.create_behavior_labels`](./io.md#function-create_behavior_labels): Create behavior labels from BORIS exported csv files.
- [`io.get_sample_data`](./io.md#function-get_sample_data): Load a sample dataset of 5 mice social interaction videos. Each video is approx. 5 minutes in duration
- [`io.get_sample_data_paths`](./io.md#function-get_sample_data_paths): Get path to sample data files provided with package. 
- [`io.get_sample_nwb_paths`](./io.md#function-get_sample_nwb_paths): Get path to a sample NWB file with tracking data for testing and dev purposes.
- [`io.load_data`](./io.md#function-load_data): Load an object from a pickle file
- [`io.load_sklearn_model`](./io.md#function-load_sklearn_model): Load sklearn model from file
- [`io.read_DLC_tracks`](./io.md#function-read_dlc_tracks): Read in tracks from DLC.
- [`io.read_NWB_tracks`](./io.md#function-read_nwb_tracks): Read in tracks from NWB PoseEstimiationSeries format (something saved using the DLC2NWB package).
- [`io.read_boris_annotation`](./io.md#function-read_boris_annotation): Read behavior annotation from BORIS exported csv file. 
- [`io.read_sleap_tracks`](./io.md#function-read_sleap_tracks): Read in tracks from SLEAP.
- [`io.save_DLC_tracks_h5`](./io.md#function-save_dlc_tracks_h5): Save DLC tracks in h5 format.
- [`io.save_sklearn_model`](./io.md#function-save_sklearn_model): Save sklearn model to file
- [`io.uniquifier`](./io.md#function-uniquifier): Return a sequence (e.g. list) with unique elements only, but maintaining original list order
- [`utils.checkFFMPEG`](./utils.md#function-checkffmpeg): Check for ffmpeg dependencies
- [`utils.check_keras`](./utils.md#function-check_keras)
- [`video.add_randomforest_predictions`](./video.md#function-add_randomforest_predictions): Perform cross validation of a RandomForestClassifier to predict behavior based on 
- [`video.create_dataset`](./video.md#function-create_dataset): Creates DataFrame that houses pose-tracking data and behavior annotations, along with relevant metadata, features and behavior annotation labels.
- [`video.create_metadata`](./video.md#function-create_metadata): Prepare a metadata dictionary for defining a ExperimentDataFrame. 
- [`video.get_sample_openfield_data`](./video.md#function-get_sample_openfield_data): Load a sample dataset of 1 mouse in openfield setup. The video is the sample that comes with DLC.
- [`video.load_experiment`](./video.md#function-load_experiment): Load DataFrame from file.


---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
