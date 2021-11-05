import deeplabcut
import glob 
import os

#Better to use full paths here
path_config_file = '/home/blansdel/projects/brett_dlc/dlc_projects/pilot_study-Brett-2021-07-19/config.yaml'

#video_path = '/home/blansdel/projects/brett_dlc/dlc_projects/pilot_study-Brett-2021-07-19/videos/to_analyze/'
video_path = '/home/blansdel/projects/brett_dlc/dlc_projects/pilot_study-Brett-2021-07-19/videos/to_analyze_iteration_3/'

overwrite = True

videos = ['e3v813a-20210610T120637-121213.avi',
       'e3v813a-20210714T094126-094745.avi.re-encoded.1600px.16265k.avi',
       'e3v813a-20210610T121558-122141.avi',
       'e3v813a-20210714T091732-092358.avi.re-encoded.1600px.16265k.avi',
       'e3v813a-20210610T122332-122642.avi',
       'e3v813a-20210714T095234-095803.avi.re-encoded.1600px.16265k.avi',
       'e3v813a-20210610T122758-123309.avi',
       'e3v813a-20210714T092722-093422.avi.re-encoded.1600px.16265k.avi',
       'e3v813a-20210610T123521-124106.avi',
       'e3v813a-20210714T095950-100613.avi.re-encoded.1600px.16265k.avi']

videos_short = ['e3v813a-20210610T120637-121213.avi']

#Overwrite with short set of videos
#videos = videos_short

videos = [video_path + i for i in videos]

shuffle = 1

#track_methods = ['ellipse', 'skeleton', 'box']
#extensions = ['el', 'sk', 'bx']

##Ellipse seems best, just make videos for that one
track_methods = ['ellipse']
extensions = ['el']

deeplabcut.analyze_videos(path_config_file, videos, videotype='.avi')

deeplabcut.create_video_with_all_detections(path_config_file, videos)

##Clear beforehand with this command:
# rm *el*.mp4; rm *filtered.h5; rm *filtered.csv

for track_method, ext in zip(track_methods, extensions):
    deeplabcut.convert_detections2tracklets(path_config_file, videos, videotype='avi', \
                                            shuffle=shuffle, trainingsetindex=0, track_method=track_method,
                                            overwrite = overwrite)

    tracklet_pickles = glob.glob(f'{video_path}/*{ext}.pickle')
    mapping = {v: None for v in videos}
    for pickle in tracklet_pickles:
        for v in videos:
            if v[:-4] in pickle:
                mapping[v] = pickle

    for vid in videos:
        pickle_file = mapping[vid]
        deeplabcut.stitch_tracklets(path_config_file, pickle_file)

        deeplabcut.refine_tracklets(path_config_file, pickle_file.replace('pickle', 'h5'), vid, \
                    max_gap=0, min_swap_len=2, min_tracklet_len=2, trail_len=50)

    # Filter tracks
    #deeplabcut.filterpredictions(path_config_file, videos, shuffle = shuffle, videotype='avi', \
    #                                    trainingsetindex = 0, track_method = track_method)    

    #Filter with ARIMA
    #deeplabcut.filterpredictions(path_config_file, videos, shuffle = shuffle, videotype='avi', \
    #                                    trainingsetindex = 0, track_method = track_method,
    #                                    filtertype = 'arima', ARdegree=5,MAdegree=2, p_bound = 0)

    #Filter with median
    deeplabcut.filterpredictions(path_config_file, videos, shuffle = shuffle, videotype='avi', \
                                        trainingsetindex = 0, track_method = track_method,
                                        filtertype = 'median', windowlength = 15)

    #Filter with spline
    #deeplabcut.filterpredictions(path_config_file, videos, shuffle = shuffle, videotype='avi', \
    #                                    trainingsetindex = 0, track_method = track_method,
    #                                    filtertype = 'spline', windowlength = 15)

    # Create videos and plots
    #deeplabcut.plot_trajectories(path_config_file, videos, shuffle = shuffle, \
    #                            trainingsetindex = 0, track_method = track_method)

    #Set fastmode=False to better work on cluster
    deeplabcut.create_labeled_video(path_config_file, videos, videotype='.avi', 
                        filtered=True, shuffle = shuffle, trainingsetindex = 0, 
                        track_method = track_method, fastmode = True,
                        color_by = 'individual', draw_skeleton = True)

    #deeplabcut.create_labeled_video(path_config_file, videos, videotype='.avi', 
    #                    filtered=False, shuffle = shuffle, trainingsetindex = 0, 
    #                    track_method = track_method, fastmode = True,
    #                    color_by = 'individual', draw_skeleton = True)

#Active learning code
#deeplabcut.extract_outlier_frames(path_config_file, videos, outlieralgorithm = 'jump')
#deeplabcut.refine_labels(path_config_file)