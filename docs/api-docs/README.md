<!-- markdownlint-disable -->

# API Overview

## Modules

- [`features`](./features.md#module-features)
- [`interpolation`](./interpolation.md#module-interpolation)
- [`io`](./io.md#module-io): Loading and saving tracking and behavior annotation files 
- [`plot`](./plot.md#module-plot)
- [`unsupervised`](./unsupervised.md#module-unsupervised)
- [`utils`](./utils.md#module-utils): Small helper utilities
- [`video`](./video.md#module-video): Basic video tracking and behavior class that houses data. 

## Classes

- [`io.BufferedIOBase`](./io.md#class-bufferediobase): Base class for buffered IO objects.
- [`io.IOBase`](./io.md#class-iobase): The abstract base class for all I/O classes, acting on streams of
- [`io.RawIOBase`](./io.md#class-rawiobase): Base class for raw binary I/O.
- [`io.TextIOBase`](./io.md#class-textiobase): Base class for text I/O.
- [`io.UnsupportedOperation`](./io.md#class-unsupportedoperation)
- [`plot.MplColorHelper`](./plot.md#class-mplcolorhelper)
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
- [`io.save_DLC_tracks_h5`](./io.md#function-save_dlc_tracks_h5): Save DLC tracks in h5 format.
- [`io.save_sklearn_model`](./io.md#function-save_sklearn_model): Save sklearn model to file
- [`io.uniquifier`](./io.md#function-uniquifier): Return a sequence (e.g. list) with unique elements only, but maintaining original list order
- [`plot.create_ethogram_video`](./plot.md#function-create_ethogram_video): Overlay ethogram on top of source video with ffmpeg
- [`plot.create_mosaic_video`](./plot.md#function-create_mosaic_video): Take a set of video clips and turn them into a mosaic using ffmpeg 
- [`plot.create_sample_videos`](./plot.md#function-create_sample_videos): Create a sample of videos displaying the labeled behaviors using ffmpeg. 
- [`plot.plot_embedding`](./plot.md#function-plot_embedding): Scatterplot of a 2D TSNE or UMAP embedding from the dataset.
- [`plot.plot_ethogram`](./plot.md#function-plot_ethogram): Simple ethogram of one video, up to a certain frame number.
- [`plot.plot_unsupervised_results`](./plot.md#function-plot_unsupervised_results): Set of plots for unsupervised behavior clustering results
- [`unsupervised.cluster_behaviors`](./unsupervised.md#function-cluster_behaviors): Cluster behaviors based on dimensionality reduction, kernel density estimation, and watershed clustering.
- [`unsupervised.compute_density`](./unsupervised.md#function-compute_density): Compute kernel density estimate of embedding.
- [`unsupervised.compute_morlet`](./unsupervised.md#function-compute_morlet): Compute morlet wavelet transform of a time series.
- [`unsupervised.compute_tsne_embedding`](./unsupervised.md#function-compute_tsne_embedding): Compute TSNE embedding. Only for a random subset of rows.
- [`unsupervised.compute_watershed`](./unsupervised.md#function-compute_watershed): Compute watershed clustering of a density matrix. 
- [`utils.checkFFMPEG`](./utils.md#function-checkffmpeg): Check for ffmpeg dependencies
- [`video.create_dataset`](./video.md#function-create_dataset): Houses DLC tracking data and behavior annotations in pandas DataFrame for ML, along with relevant metadata, features and behavior annotation labels.
- [`video.create_metadata`](./video.md#function-create_metadata): Prepare a metadata dictionary for defining a ExperimentDataFrame. 
- [`video.get_sample_openfield_data`](./video.md#function-get_sample_openfield_data): Load a sample dataset of 1 mouse in openfield setup. The video is the sample that comes with DLC.
- [`video.load_experiment`](./video.md#function-load_experiment): Load DataFrame from file.


---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
