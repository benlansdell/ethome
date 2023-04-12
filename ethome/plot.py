import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import os
import time
import numpy as np
from glob import glob
import pandas as pd

from ethome.config import global_config

def plot_embedding(dataset : pd.DataFrame, 
                   col_names : list  = ['embedding_0', 'embedding_1'],
                   col_labels : list = None,
                   color_col : str = None, 
                   figsize : tuple = (10,10),
                   **kwargs) -> tuple:
    """Scatterplot of a 2D TSNE or UMAP embedding from the dataset.
    
    Args:
        dataset: data
        col_names: list of column names to use for the x and y axes
        color_col: if provided, a column that will be used to color the points in the scatter plot
        figsize: tuple with the dimensions of the plot (in inches)
        **kwargs: All other keyword pairs are sent to Matplotlib's scatter function

    Returns:
        tuple (fig, axes). The Figure and Axes objects. 
    """

    if color_col is not None:
        c = dataset[color_col]
    else:
        c = None

    col1, col2 = col_names
    fig, axes = plt.subplots(1, 1, figsize = figsize)
    axes.scatter(x = dataset[col1], y = dataset[col2], s = 1, c = c, **kwargs)
    if col_labels is None: col_labels = col_names
    axes.set_xlabel(col_labels[0])
    axes.set_ylabel(col_labels[1])
    return fig, axes

class MplColorHelper:  # pragma: no cover

    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)    

def plot_unsupervised_results(dataset : pd.DataFrame, 
                              cluster_results : tuple, 
                              col_names : list = ['embedding_0', 'embedding_1'],
                              figsize : tuple = (15,4), 
                              **kwargs): # pragma: no cover
    """Set of plots for unsupervised behavior clustering results
    
    Args:
        dataset: data
        cluster_results: tuple output by 'cluster_behaviors'
        col_names: list of column names to use for the x and y axes
        figsize: tuple with the plot dimensions, in inches
        kwargs: all other keyword pairs are sent to Matplotlib's scatter function

    Returns:
        tuple (fig, axes). The Figure and Axes objects. 
    """    
    dens_matrix, labels, extent = cluster_results

    fig, axes = plt.subplots(1, 3, figsize = figsize)
    col1, col2 = col_names
    axes[0].scatter(x = dataset[col1], y = dataset[col2], s = 1, **kwargs)
    axes[0].set_xlabel('Embedding dim 1')
    axes[0].set_ylabel('Embedding dim 2')
    axes[0].set_title('Embedding')
    rectangle = plt.Rectangle((extent[0],extent[2]), extent[1]-extent[0], extent[3]-extent[2], fill=False,ec="red")
    axes[0].add_patch(rectangle)
    axes[1].imshow(dens_matrix, origin = 'lower')
    axes[1].set_title('Density estimate')
    axes[1].axis('off')
    axes[2].imshow(labels, origin = 'lower')
    axes[2].axis('off')
    axes[2].set_title('Watershed clustering')

    #For each label, find the mean location
    all_labels = np.unique(labels)
    all_labels = all_labels[all_labels != 0]
    colhelper = MplColorHelper('viridis', 0, max(all_labels))

    for lb in all_labels:
        locs = np.where(labels == lb)
        mean_y = np.mean(locs[0])
        mean_x = np.mean(locs[1])
        lab_col = colhelper.get_rgb(lb)
        lab_bright = lab_col[0]*0.299 + lab_col[1]*0.587 + lab_col[2]*0.114 > (120/255)
        text_col = 'black' if lab_bright else 'white'
        axes[2].text(mean_x-8, mean_y-4, str(lb), c = text_col)

    return fig, axes

def plot_ethogram(dataset : pd.DataFrame, 
                  vid_key : str, 
                  query_label : str = 'unsup_behavior_label', 
                  frame_limit : int = 4000, 
                  figsize : tuple = (16,2)) -> tuple:  # pragma: no cover
    """Simple ethogram of one video, up to a certain frame number.

    Args:
        dataset: 
        vid_key: key (in dataset.metadata) pointing to the video to make ethogram for
        query_label: the column containing the behavior labels to plot
        frame_limit: only make the ethogram for frames between [0, frame_limit]
        figsize: tuple with figure size (in inches)

    Returns:
        tuple (fig, axes). The Figure and Axes objects
    """
    fig, ax = plt.subplots(1,1,figsize = figsize)
    plot_data = dataset.loc[dataset['filename'] == vid_key, query_label][:frame_limit].to_numpy()
    b = np.zeros((plot_data.size, plot_data.max()+1))
    b[np.arange(plot_data.size),plot_data] = plot_data
    plt.imshow(b.T, aspect = 'auto', origin = 'lower', interpolation = 'none', alpha = (b.T > 0).astype(float))
    plt.axis('off')
    plt.tight_layout(pad = 0)
    plt.xlim([0, frame_limit])
    return fig, ax

#TODO
# Bug with trimming the jpg here. 
def create_ethogram_video(dataset : pd.DataFrame, 
                          vid_key : str, 
                          query_label : str, 
                          out_file : str, 
                          frame_limit : int = 4000, 
                          im_dim : float = 16, 
                          min_frames : int = 3) -> None:  # pragma: no cover
    """Overlay ethogram on top of source video with ffmpeg

    Args:
        dataset: source dataset
        vid_key: the key (in dataset.metadata) pointing to the video to make ethogram for. metadata must have field 'video_files' that points to the source video location
        query_label: the column containing the behavior labels to plot
        out_file: output path for created video
        frame_limit: only make the ethogram/video for frames [0, frame_limit]
        in_dim: x dimension (in inches) of ethogram
        min_frames: any behaviors occurring for less than this number of frames are not labeled

    Returns:
        None
    """
    vid_file = dataset.metadata.details[vid_key]['video_files']
    fps = dataset.metadata.details[vid_key]['fps']
    time_limit = frame_limit/fps
    fig, _ = plt.subplots(1,1,figsize = (im_dim,2))
    plot_data = dataset.loc[dataset['filename'] == vid_key, query_label][:frame_limit].to_numpy()

    #Create behavior_times
    behavior_times = []
    cur_behav = -2
    start_idx = 0
    for idx, lab in enumerate(plot_data):
        if cur_behav != lab:
            if cur_behav != -2:
                if idx - start_idx > min_frames:
                    behavior_times.append((start_idx/fps, (idx-1)/fps, cur_behav))
            start_idx = idx
        cur_behav = lab
    behavior_times.append((start_idx/fps, (idx-1)/fps, cur_behav))

    b = np.zeros((plot_data.size, plot_data.max()+1))
    b[np.arange(plot_data.size),plot_data] = plot_data
    plt.ioff()
    plt.imshow(b.T, aspect = 'auto', origin = 'lower', interpolation = 'none', alpha = (b.T > 0).astype(float))
    plt.axis('off')
    plt.tight_layout(pad = 0)
    plt.xlim([0, frame_limit])
    os.makedirs('./tmp/', exist_ok = True)
    fn_out = f'./tmp/{query_label}.jpg'
    trimmed_fn = fn_out.replace('.jpg', '_trimmed.jpg')
    dpi = 2000/im_dim
    fig.savefig(fn_out, dpi = dpi)
    plt.close()
    plt.ioff()
    trim_cmd = f'convert {fn_out} -fuzz 7% -trim -resize 1600x1600 {trimmed_fn}'
    os.system(trim_cmd)
    vid_path = vid_file
    #Combine query label with movie    
    start_time_str = time.strftime('%H:%M:%S', time.gmtime(time_limit))
    text_filter = f"drawtext=text='|':fontcolor=green:fontsize=60:y=1:x='-10+(mod(round((w+5)*t/{time_limit}),w+5))',drawtext=text='|':fontcolor=green:fontsize=60:y=20:x='-10+(mod(round((w+5)*t/{time_limit}),(w+5)))'"
    for str_time, end_time, behav_label in behavior_times:
        behav_text = f",drawtext=text='{behav_label}':fontcolor=black:fontsize=30:y=10:x=10:enable='between(t,{str_time},{end_time})'"
        text_filter += behav_text

    ffmpeg_cmd = f'''ffmpeg -y -hide_banner -loglevel error -i {vid_path} -i {trimmed_fn} \
    -filter_complex "[0:v][1:v]overlay=0:0,{text_filter}" \
    -t {start_time_str} \
    -threads 8 -q:v 3 {out_file}'''
    os.system(ffmpeg_cmd)

def create_sample_videos(dataset : pd.DataFrame, 
                         video_dir : str, 
                         out_dir : str, 
                         query_col : str = 'unsup_behavior_label', 
                         N_sample_rows : int = 16, 
                         window_size : int = 2, 
                         fps : float = 30,
                         N_supersample_rows : int = 1000) -> None:  # pragma: no cover
    """Create a sample of videos displaying the labeled behaviors using ffmpeg. 

    For each behavior label, randomly choose frames from the entire dataset and extract short clips from source videos based around those points. Tries to select frames where the labeled behavior is exhibited in many frames of the clip.

    Args:
        dataset: source dataset
        video_dir: location of source video files
        out_dir: base output directory to save videos. Videos are saved in the form: [out_dir]/[behavior_label]/[video_name]_[time in seconds].avi
        query_label: the column containing the behavior labels to extract clips for. Each unique value in this column is treated as a separate behavior
        N_sample_rows: number of clips to extract per behavior
        window_size: amount of video to extract on either side of the sampled frame, in seconds
        fps: frames per second of videos
        N_supersample_rows: this many rows are randomly sampled for each behavior label, and the top N_sample_rows are returned (in terms of number of adjacent frames also exhibiting that behavior). Shouldn't need to play with this.

    Returns:
        None
    """
    labels = dataset[query_col].unique()
    labels = labels[labels >= 0]
    #all_labels = np.unique(labels)

    def get_window_size(label_idx, sample_row, max_size = 500):
        s_m = 0
        for idx in range(max_size):
            try:
                if dataset.loc[sample_row-idx, query_col] == label_idx:
                    s_m += 1
                else:
                    break
            except:
                break
                    
        s_p = 0
        for idx in range(max_size):
            try:
                if dataset.loc[sample_row+idx, query_col] == label_idx:
                    s_p += 1
                else:
                    break
            except:
                break
        return sample_row-s_m, s_p+sample_row, sample_row, s_m+s_p


    for label_idx in labels:

        print(f"Making sample videos for behavior label {label_idx}")
        label_indices = dataset[query_col] == label_idx
        if sum(label_indices) == 0: continue

        ## Pull out some sample frames from each video for this behavior
        behavior_rows = dataset[dataset[query_col] == label_idx]
        random_sample_indices = np.random.choice(behavior_rows.index, min(len(behavior_rows.index), N_supersample_rows), replace = False)
        window_sizes = [get_window_size(label_idx, i) \
                        for i in random_sample_indices]
        window_sizes = sorted(window_sizes, key = lambda x: x[3], reverse = True)
        window_sizes = pd.DataFrame(window_sizes, columns = ['start', 'end', 'pt', 'len']).drop_duplicates(subset = ['start', 'end'])

        random_sample_indices = window_sizes['pt'].to_numpy()[:N_sample_rows]

        behavior_rows_sample = behavior_rows.loc[random_sample_indices].reset_index(drop = True)
        #For each filename in this list of samples, extract a part of that video with ffmpeg
        filenames = behavior_rows_sample.filename.unique()
        video_files = [os.path.basename(p).split('DLC')[0]+'.avi' for p in filenames]

        out_dir_vid = os.path.join(out_dir, f'behavior_label_{label_idx}')
        os.makedirs(out_dir_vid, exist_ok = True)


        for r_idx, (vid_file, fn) in enumerate(zip(video_files, filenames)):
            behave_rows_sample_vid = behavior_rows_sample[behavior_rows_sample['filename'] == fn]
            vid_name = os.path.basename(vid_file).split('.')[0]
            for idx in range(len(behave_rows_sample_vid)):

                sampl_window = window_sizes.iloc[r_idx,:].to_numpy()
                str_time = window_size + (sampl_window[0] - sampl_window[2])/fps
                end_time = window_size + (sampl_window[1] - sampl_window[2])/fps
                #str_time = 0
                #end_time = 2

                frame_number = behave_rows_sample_vid.reset_index().loc[idx, 'frame']
                behavior_time = int(frame_number/fps)
                out_file = os.path.join(out_dir_vid, f'{vid_name}_second_{behavior_time}.avi')
                start_time = max(0, behavior_time - window_size)
                start_time_str = time.strftime('%H:%M:%S', time.gmtime(start_time))
                #TODO
                # Get the times right on this line...
                text_filter = f"drawtext=text='{label_idx} active':fontcolor=red:fontsize=100:y=10:x=10:box=1:boxcolor=white:enable='between(t,{str_time},{end_time})'"
                ffmpeg_cmd = f'''ffmpeg -y -hide_banner -loglevel error -ss {start_time_str} -i {os.path.join(video_dir, vid_file)} -t 00:00:{2*window_size} \
                    -filter_complex "{text_filter}" \
                    -threads 4 {out_file}'''
                os.system(ffmpeg_cmd)

def create_mosaic_video(vid_dir : str, 
                        output_file : str, 
                        ndim : tuple = (1600, 1200)) -> None:  # pragma: no cover
    """Take a set of video clips and turn them into a mosaic using ffmpeg 
    
    16 videos are tiled.

    Args:
        vid_dir: source directory with videos in it
        output_file: output video path
        ndim: tuple with the output video dimensions, in pixels
    
    Returns:
        None    
    """
    max_mosaic_vids = global_config['create_mosaic_video__max_mosaic_vids']
    mosaic_vid_files = glob(vid_dir)[:max_mosaic_vids]

    block_size_x = ndim[0]/np.sqrt(max_mosaic_vids)
    block_size_y = ndim[1]/np.sqrt(max_mosaic_vids)

    mosaic_cmd = f'''ffmpeg -y -hide_banner -loglevel error \
    {' '.join([f'-i {f}' for f in mosaic_vid_files])} \
    -filter_complex " \
        nullsrc=size={'x'.join([str(i) for i in ndim])} [base]; \
        [0:v] setpts=PTS-STARTPTS, scale={block_size_x}x{block_size_y} [upper1]; \
        [1:v] setpts=PTS-STARTPTS, scale={block_size_x}x{block_size_y} [upper2]; \
        [2:v] setpts=PTS-STARTPTS, scale={block_size_x}x{block_size_y} [upper3]; \
        [3:v] setpts=PTS-STARTPTS, scale={block_size_x}x{block_size_y} [upper4]; \
        [4:v] setpts=PTS-STARTPTS, scale={block_size_x}x{block_size_y} [uppermid1]; \
        [5:v] setpts=PTS-STARTPTS, scale={block_size_x}x{block_size_y} [uppermid2]; \
        [6:v] setpts=PTS-STARTPTS, scale={block_size_x}x{block_size_y} [uppermid3]; \
        [7:v] setpts=PTS-STARTPTS, scale={block_size_x}x{block_size_y} [uppermid4]; \
        [8:v] setpts=PTS-STARTPTS, scale={block_size_x}x{block_size_y} [lowermid1]; \
        [9:v] setpts=PTS-STARTPTS, scale={block_size_x}x{block_size_y} [lowermid2]; \
        [10:v] setpts=PTS-STARTPTS, scale={block_size_x}x{block_size_y} [lowermid3]; \
        [11:v] setpts=PTS-STARTPTS, scale={block_size_x}x{block_size_y} [lowermid4]; \
        [12:v] setpts=PTS-STARTPTS, scale={block_size_x}x{block_size_y} [lower1]; \
        [13:v] setpts=PTS-STARTPTS, scale={block_size_x}x{block_size_y} [lower2]; \
        [14:v] setpts=PTS-STARTPTS, scale={block_size_x}x{block_size_y} [lower3]; \
        [15:v] setpts=PTS-STARTPTS, scale={block_size_x}x{block_size_y} [lower4]; \
        [base][upper1] overlay=shortest=1 [tmp1]; \
        [tmp1][upper2] overlay=shortest=1:x={block_size_x} [tmp2]; \
        [tmp2][upper3] overlay=shortest=1:x={2*block_size_x} [tmp3]; \
        [tmp3][upper4] overlay=shortest=1:x={3*block_size_x} [tmp4];\
        [tmp4][uppermid1] overlay=shortest=1:y={block_size_y} [tmp5]; \
        [tmp5][uppermid2] overlay=shortest=1:x={block_size_x}:y={block_size_y} [tmp6]; \
        [tmp6][uppermid3] overlay=shortest=1:x={2*block_size_x}:y={block_size_y} [tmp7]; \
        [tmp7][uppermid4] overlay=shortest=1:x={3*block_size_x}:y={block_size_y} [tmp8];\
        [tmp8][lowermid1] overlay=shortest=1:y={2*block_size_y} [tmp9]; \
        [tmp9][lowermid2] overlay=shortest=1:x={block_size_x}:y={2*block_size_y} [tmp10]; \
        [tmp10][lowermid3] overlay=shortest=1:x={2*block_size_x}:y={2*block_size_y} [tmp11]; \
        [tmp11][lowermid4] overlay=shortest=1:x={3*block_size_x}:y={2*block_size_y} [tmp12];\
        [tmp12][lower1] overlay=shortest=1:y={3*block_size_y} [tmp13]; \
        [tmp13][lower2] overlay=shortest=1:x={block_size_x}:y={3*block_size_y} [tmp14]; \
        [tmp14][lower3] overlay=shortest=1:x={2*block_size_x}:y={3*block_size_y} [tmp15]; \
        [tmp15][lower4] overlay=shortest=1:x={3*block_size_x}:y={3*block_size_y} \
    " -c:v libx264 {output_file}'''
    os.system(mosaic_cmd)