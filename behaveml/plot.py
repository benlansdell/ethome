import matplotlib.pyplot as plt
import os
import time
import numpy as np
from glob import glob
from behaveml.config import global_config

def plot_embedding(dataset, figsize = (10,10)):
    """Plot a 2D TSNE or UMAP embedding from the dataset"""


    fig, axes = plt.subplots(1,1, figsize = figsize)
    axes.scatter(x = dataset['embedding_0'], y = dataset['embedding_1'], s = 1)
    axes.set_xlabel('Embedding dim 1')
    axes.set_ylabel('Embedding dim 2')
    return fig, axes

def plot_ethogram(dataset, vid_key, query_label = 'unsup_behavior_label', frame_limit = 4000, figsize = (16,2)):
    fig, ax = plt.subplots(1,1,figsize = figsize)
    plot_data = dataset.data.loc[dataset.data['filename'] == vid_key, query_label][:frame_limit].to_numpy()
    b = np.zeros((plot_data.size, plot_data.max()+1))
    b[np.arange(plot_data.size),plot_data] = plot_data
    plt.imshow(b.T, aspect = 'auto', origin = 'lower', interpolation = 'none', alpha = (b.T > 0).astype(float))
    plt.axis('off')
    plt.tight_layout(pad = 0)
    plt.xlim([0, frame_limit])
    return fig, ax

def create_ethogram_video(dataset, vid_key, query_label, out_file, frame_limit = 4000, im_dim = 16, min_frames = 3):
    vid_file = dataset.metadata[vid_key]['video_files']
    fps = dataset.metadata[vid_key]['fps']
    time_limit = frame_limit/fps
    fig, _ = plt.subplots(1,1,figsize = (im_dim,2))
    plot_data = dataset.data.loc[dataset.data['filename'] == vid_key, query_label][:frame_limit].to_numpy()

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
    plt.imshow(b.T, aspect = 'auto', origin = 'lower', interpolation = 'none', alpha = (b.T > 0).astype(float))
    plt.axis('off')
    plt.tight_layout(pad = 0)
    plt.xlim([0, frame_limit])
    os.makedirs('./tmp/', exist_ok = True)
    fn_out = f'./tmp/{query_label}.jpg'
    trimmed_fn = fn_out.replace('.jpg', '_trimmed.jpg')
    dpi = 2000/im_dim
    fig.savefig(fn_out, dpi = dpi)
    trim_cmd = f'convert {fn_out} -fuzz 7% -trim -resize 1600x1600 {trimmed_fn}'
    os.system(trim_cmd)
    vid_path = vid_file
    #Combine query label with movie    
    start_time_str = time.strftime('%H:%M:%S', time.gmtime(time_limit))
    text_filter = f"drawtext=text='|':fontcolor=green:fontsize=60:y=1:x='-10+(mod(round((w+5)*t/{time_limit}),w+5))',drawtext=text='|':fontcolor=green:fontsize=60:y=20:x='-10+(mod(round((w+5)*t/{time_limit}),(w+5)))'"
    for str_time, end_time, behav_label in behavior_times:
        behav_text = f",drawtext=text='{behav_label}':fontcolor=black:fontsize=30:y=10:x=10:enable='between(t,{str_time},{end_time})'"
        text_filter += behav_text

    ffmpeg_cmd = f'''ffmpeg -y -i {vid_path} -i {trimmed_fn} \
    -filter_complex "[0:v][1:v]overlay=0:0,{text_filter}" \
    -t {start_time_str} \
    -threads 8 -q:v 3 {out_file}'''
    os.system(ffmpeg_cmd)


def create_sample_videos(dataset, video_dir, out_dir, query_col = 'unsup_behavior_label', N_sample_rows = 16, window_size = 2, fps = 30):

    n_labels = 0
    for label_idx in range(n_labels):
        print(f"Making sample videos for behavior label {label_idx}")
        label_indices = dataset[query_col] == label_idx
        if sum(label_indices) == 0: continue

        ## Pull out some sample frames from each video for this behavior
        behavior_rows = dataset[dataset[query_col] == label_idx]
        random_sample_indices = np.random.choice(behavior_rows.index, N_sample_rows, replace = False)
        behavior_rows_sample = behavior_rows.loc[random_sample_indices].reset_index(drop = True)

        #For each filename in this list of samples, extract a part of that video with ffmpeg
        filenames = behavior_rows_sample.filename.unique()
        video_files = [os.path.basename(p).split('DLC')[0]+'.avi' for p in filenames]

        out_dir_vid = os.path.join(out_dir, f'behavior_label_{label_idx}')
        os.makedirs(out_dir_vid, exist_ok = True)
            
        for vid_file, fn in zip(video_files, filenames):
            behave_rows_sample_vid = behavior_rows_sample[behavior_rows_sample['filename'] == fn]
            vid_name = os.path.basename(vid_file).split('.')[0]
            for idx in range(len(behave_rows_sample_vid)):
                frame_number = behave_rows_sample_vid.reset_index().loc[idx, 'frame']
                behavior_time = int(frame_number/fps)
                out_file = os.path.join(out_dir_vid, f'{vid_name}_second_{behavior_time}.avi')
                start_time = max(0, behavior_time - window_size)
                start_time_str = time.strftime('%H:%M:%S', time.gmtime(start_time))
                ffmpeg_cmd = f'ffmpeg -ss {start_time_str} -i {os.path.join(video_dir, vid_file)} -t 00:00:{2*window_size} -threads 4 {out_file}'
                os.system(ffmpeg_cmd)
                
#TODO
#Make the dimension not hard coded here
def create_mosaic_video(vid_dir, output_file, ndim = ('1600','1200')):
    max_mosaic_vids = global_config['create_mosaic_video__max_mosaic_vids']
    mosaic_vid_files = glob(vid_dir)[:max_mosaic_vids]

    mosaic_cmd = f'''ffmpeg -y \
    {' '.join([f'-i {f}' for f in mosaic_vid_files])} \
    -filter_complex " \
        nullsrc=size={'x'.join(ndim)} [base]; \
        [0:v] setpts=PTS-STARTPTS, scale=400x300 [upper1]; \
        [1:v] setpts=PTS-STARTPTS, scale=400x300 [upper2]; \
        [2:v] setpts=PTS-STARTPTS, scale=400x300 [upper3]; \
        [3:v] setpts=PTS-STARTPTS, scale=400x300 [upper4]; \
        [4:v] setpts=PTS-STARTPTS, scale=400x300 [uppermid1]; \
        [5:v] setpts=PTS-STARTPTS, scale=400x300 [uppermid2]; \
        [6:v] setpts=PTS-STARTPTS, scale=400x300 [uppermid3]; \
        [7:v] setpts=PTS-STARTPTS, scale=400x300 [uppermid4]; \
        [8:v] setpts=PTS-STARTPTS, scale=400x300 [lowermid1]; \
        [9:v] setpts=PTS-STARTPTS, scale=400x300 [lowermid2]; \
        [10:v] setpts=PTS-STARTPTS, scale=400x300 [lowermid3]; \
        [11:v] setpts=PTS-STARTPTS, scale=400x300 [lowermid4]; \
        [12:v] setpts=PTS-STARTPTS, scale=400x300 [lower1]; \
        [13:v] setpts=PTS-STARTPTS, scale=400x300 [lower2]; \
        [14:v] setpts=PTS-STARTPTS, scale=400x300 [lower3]; \
        [15:v] setpts=PTS-STARTPTS, scale=400x300 [lower4]; \
        [base][upper1] overlay=shortest=1 [tmp1]; \
        [tmp1][upper2] overlay=shortest=1:x=400 [tmp2]; \
        [tmp2][upper3] overlay=shortest=1:x=800 [tmp3]; \
        [tmp3][upper4] overlay=shortest=1:x=1200 [tmp4];\
        [tmp4][uppermid1] overlay=shortest=1:y=300 [tmp5]; \
        [tmp5][uppermid2] overlay=shortest=1:x=400:y=300 [tmp6]; \
        [tmp6][uppermid3] overlay=shortest=1:x=800:y=300 [tmp7]; \
        [tmp7][uppermid4] overlay=shortest=1:x=1200:y=300 [tmp8];\
        [tmp8][lowermid1] overlay=shortest=1:y=600 [tmp9]; \
        [tmp9][lowermid2] overlay=shortest=1:x=400:y=600 [tmp10]; \
        [tmp10][lowermid3] overlay=shortest=1:x=800:y=600 [tmp11]; \
        [tmp11][lowermid4] overlay=shortest=1:x=1200:y=600 [tmp12];\
        [tmp12][lower1] overlay=shortest=1:y=900 [tmp13]; \
        [tmp13][lower2] overlay=shortest=1:x=400:y=900 [tmp14]; \
        [tmp14][lower3] overlay=shortest=1:x=800:y=900 [tmp15]; \
        [tmp15][lower4] overlay=shortest=1:x=1200:y=900 \
    " -c:v libx264 {output_file}'''
    os.system(mosaic_cmd)