<!-- markdownlint-disable -->

# API Overview

## Modules

- [`config`](./config.md#module-config): Configuration options for behaveml functions.
- [`dl`](./dl.md#module-dl)
- [`dl.dl_features`](./dl.dl_features.md#module-dldl_features)
- [`dl.dl_generators`](./dl.dl_generators.md#module-dldl_generators)
- [`dl.dl_models`](./dl.dl_models.md#module-dldl_models)
- [`dl.feature_engineering`](./dl.feature_engineering.md#module-dlfeature_engineering)
- [`dl.grid_searches`](./dl.grid_searches.md#module-dlgrid_searches)
- [`features`](./features.md#module-features): Functions to take pose tracks and compute a set of features from them.
- [`generic_features`](./generic_features.md#module-generic_features): Functions to take pose tracks and compute a set of features from them 
- [`interpolation`](./interpolation.md#module-interpolation)
- [`io`](./io.md#module-io): Loading and saving tracking and behavior annotation files 
- [`mars_features`](./mars_features.md#module-mars_features)
- [`ml`](./ml.md#module-ml): Machine learning functions 
- [`plot`](./plot.md#module-plot)
- [`unsupervised`](./unsupervised.md#module-unsupervised)
- [`utils`](./utils.md#module-utils): Small helper utilities
- [`video`](./video.md#module-video): Basic video tracking and behavior class that houses data. 

## Classes

- [`dl_features.Trainer`](./dl.dl_features.md#class-trainer)
- [`dl_generators.MABe_Generator`](./dl.dl_generators.md#class-mabe_generator)
- [`features.Features`](./features.md#class-features)
- [`io.BufferedIOBase`](./io.md#class-bufferediobase): Base class for buffered IO objects.
- [`io.IOBase`](./io.md#class-iobase): The abstract base class for all I/O classes, acting on streams of
- [`io.RawIOBase`](./io.md#class-rawiobase): Base class for raw binary I/O.
- [`io.TextIOBase`](./io.md#class-textiobase): Base class for text I/O.
- [`io.UnsupportedOperation`](./io.md#class-unsupportedoperation)
- [`plot.MplColorHelper`](./plot.md#class-mplcolorhelper)
- [`video.MLDataFrame`](./video.md#class-mldataframe): DataFrame useful for interfacing between pandas and sklearn. Stores a data
- [`video.VideosetDataFrame`](./video.md#class-videosetdataframe)

## Functions

- [`dl_features.compute_dl_probability_features`](./dl.dl_features.md#function-compute_dl_probability_features)
- [`dl_features.convert_to_mars_format`](./dl.dl_features.md#function-convert_to_mars_format)
- [`dl_features.convert_to_pandas_df`](./dl.dl_features.md#function-convert_to_pandas_df)
- [`dl_features.lrs`](./dl.dl_features.md#function-lrs)
- [`dl_features.normalize_data`](./dl.dl_features.md#function-normalize_data)
- [`dl_features.run_task`](./dl.dl_features.md#function-run_task)
- [`dl_features.seed_everything`](./dl.dl_features.md#function-seed_everything)
- [`dl_generators.features_distances`](./dl.dl_generators.md#function-features_distances)
- [`dl_generators.features_distances_normalized`](./dl.dl_generators.md#function-features_distances_normalized)
- [`dl_generators.features_identity`](./dl.dl_generators.md#function-features_identity)
- [`dl_generators.features_mars`](./dl.dl_generators.md#function-features_mars)
- [`dl_generators.features_mars_distr`](./dl.dl_generators.md#function-features_mars_distr)
- [`dl_generators.features_via_sklearn`](./dl.dl_generators.md#function-features_via_sklearn)
- [`dl_generators.make_df`](./dl.dl_generators.md#function-make_df)
- [`dl_models.build_baseline_model`](./dl.dl_models.md#function-build_baseline_model)
- [`feature_engineering.augment_features`](./dl.feature_engineering.md#function-augment_features)
- [`feature_engineering.boiler_plate`](./dl.feature_engineering.md#function-boiler_plate)
- [`feature_engineering.make_features_distances`](./dl.feature_engineering.md#function-make_features_distances)
- [`feature_engineering.make_features_mars`](./dl.feature_engineering.md#function-make_features_mars)
- [`feature_engineering.make_features_mars_distr`](./dl.feature_engineering.md#function-make_features_mars_distr)
- [`feature_engineering.make_features_mars_reduced`](./dl.feature_engineering.md#function-make_features_mars_reduced)
- [`feature_engineering.make_features_social`](./dl.feature_engineering.md#function-make_features_social)
- [`feature_engineering.make_features_velocities`](./dl.feature_engineering.md#function-make_features_velocities)
- [`generic_features.compute_centerofmass`](./generic_features.md#function-compute_centerofmass)
- [`generic_features.compute_centerofmass_interanimal_distances`](./generic_features.md#function-compute_centerofmass_interanimal_distances)
- [`generic_features.compute_centerofmass_interanimal_speed`](./generic_features.md#function-compute_centerofmass_interanimal_speed)
- [`generic_features.compute_centerofmass_velocity`](./generic_features.md#function-compute_centerofmass_velocity)
- [`generic_features.compute_distance_features`](./generic_features.md#function-compute_distance_features)
- [`generic_features.compute_speed_features`](./generic_features.md#function-compute_speed_features)
- [`interpolation.interpolate_lowconf_points`](./interpolation.md#function-interpolate_lowconf_points): Interpolate raw tracking points if their probabilities are available.
- [`io.create_behavior_labels`](./io.md#function-create_behavior_labels): Create behavior labels from BORIS exported csv files.
- [`io.get_sample_data`](./io.md#function-get_sample_data): Load a sample dataset of 5 mice social interaction videos. Each video is approx. 5 minutes in duration
- [`io.get_sample_data_paths`](./io.md#function-get_sample_data_paths): Get path to sample data files provided with package. 
- [`io.load_data`](./io.md#function-load_data): Load an object from a pickle file
- [`io.load_sklearn_model`](./io.md#function-load_sklearn_model): Load sklearn model from file
- [`io.read_DLC_tracks`](./io.md#function-read_dlc_tracks): Read in tracks from DLC.
- [`io.read_boris_annotation`](./io.md#function-read_boris_annotation): Read behavior annotation from BORIS exported csv file. 
- [`io.rename_df_cols`](./io.md#function-rename_df_cols): Rename dataframe columns 
- [`io.save_DLC_tracks_h5`](./io.md#function-save_dlc_tracks_h5): Save DLC tracks in h5 format.
- [`io.save_sklearn_model`](./io.md#function-save_sklearn_model): Save sklearn model to file
- [`io.uniquifier`](./io.md#function-uniquifier): Return a sequence (e.g. list) with unique elements only, but maintaining original list order
- [`mars_features.compute_distance_features`](./mars_features.md#function-compute_distance_features)
- [`mars_features.compute_mars_features`](./mars_features.md#function-compute_mars_features)
- [`mars_features.compute_mars_reduced_features`](./mars_features.md#function-compute_mars_reduced_features)
- [`mars_features.compute_social_features`](./mars_features.md#function-compute_social_features)
- [`mars_features.compute_velocity_features`](./mars_features.md#function-compute_velocity_features)
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
- [`video.clone_metadata`](./video.md#function-clone_metadata): Prepare a metadata dictionary for defining a VideosetDataFrame. 
- [`video.get_sample_openfield_data`](./video.md#function-get_sample_openfield_data): Load a sample dataset of 1 mouse in openfield setup. The video is the sample that comes with DLC.
- [`video.load_videodataset`](./video.md#function-load_videodataset): Load VideosetDataFrame from file.


---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
