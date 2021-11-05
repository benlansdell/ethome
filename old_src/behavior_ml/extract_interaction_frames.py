import numpy as np
import os
import glob
import pandas as pd
import pickle 
import argparse
from collections import defaultdict
from joblib import load
import hashlib
import json
import cv2
from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader

from lib.utils import seed_everything

rate = 1/30
out_dir = './data/frame_selection/'
vid_path = './data/dlc/videos/'
scorer = 'DLC_resnet50_pilot_studyAug17shuffle1_100000'
boris_project_file = './data/boris/pilot_study.boris'
n_frames = 100

pilot_2_durations = np.array([360+26.00,
                        360+59.97,
                        360+19.03,
                        300+29.03,
                        360+23.00])

boris_files = ['./data/boris/DLC1.csv', #e3v813a-20210610T120637-121213
                './data/boris/DLC2.csv', #e3v813a-20210610T121558-122141
                './data/boris/DLC3.csv', #e3v813a-20210610T122332-122642
                './data/boris/DLC4.csv', #e3v813a-20210610T122758-123309
                './data/boris/DLC5.csv'] #e3v813a-20210610T123521-124106

def extract_frames(videopath, output_path, prefix, frame_list):

    cap = cv2.VideoCapture(videopath)
    
    for frame_number in frame_list:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
        res, frame = cap.read()
        #frame = cap.get_frame(frame_number*rate)
        if frame is not None:
            cv2.imwrite(f"{output_path}/img_{prefix}_{frame_number}.png", frame)
        else:
            print(f"Reading frame {frame_number} from video failed")
            break
    cap.release()

def format_labels(boris_in, length):
    length = int(length)
    boris_labels = pd.read_csv(boris_in, skiprows = 15)
    boris_labels['index'] = (boris_labels.index//2)
    boris_labels = boris_labels.pivot_table(index = 'index', columns = 'Status', values = 'Time').reset_index()
    boris_labels = list(np.array(boris_labels[['START', 'STOP']]))
    boris_labels = [list(i) for i in boris_labels]
    ground_truth = np.zeros(length)
    for start, end in boris_labels:
        ground_truth[int(start/rate):int(end/rate)] = 1
    return ground_truth

seed_everything()

#Load in first pilot study data lengths
f = open(boris_project_file,)  
project_data = json.load(f)

lengths = {}
for idx_b, fn in enumerate(boris_files):
    name = os.path.basename(fn)[:-4]
    try:
        key = list(project_data['observations'][name]['media_info']['length'].keys())[0]
        lengths[fn] = int(project_data['observations'][name]['media_info']['length'][key]/rate)
    except KeyError:
        lengths[fn] = round(pilot_2_durations[idx_b-5]/rate)

files_in = sorted(glob.glob('./data/dlc_improved/*_improved.csv'))

boris_to_file = {i:j for i,j in zip(boris_files, files_in)}

for idx in range(len(boris_files)):
    labels = format_labels(boris_files[idx], pilot_2_durations[idx]/rate)
    vid_file = os.path.basename(boris_to_file[boris_files[idx]]).split(scorer)[0]
    labeled_vid_file = glob.glob(vid_path + '/tracked/' + vid_file + '*.mp4')[0]

    #Randomly select 100 frames that are interaction times
    interaction_frames = np.where(labels == 1)[0]
    interaction_frames = interaction_frames[interaction_frames < 10000]
    frames = np.random.choice(interaction_frames, n_frames)

    #Save interaction frames
    if not os.path.isdir(out_dir + vid_file):
        os.mkdir(out_dir + vid_file)
    extract_frames(vid_path + vid_file + '.avi', out_dir + vid_file, 'raw', frames)
    extract_frames(labeled_vid_file, out_dir + vid_file, 'labeled', frames)


#### Once extraction is done:

# Remove raw files that aren't matching

#List all labeled files
dirs = ['e3v813a-20210610T120637-121213',
        'e3v813a-20210610T121558-122141',
        'e3v813a-20210610T122332-122642',
        'e3v813a-20210610T122758-123309',
        'e3v813a-20210610T123521-124106']
base = './data/frame_selection/'
for directory in dirs:
    print(f"Working in directory {directory}")
    labeled_files = glob.glob(base + directory + '/img_labeled*.png')
    raw_files = glob.glob(base + directory + '/img_raw*.png')
    for raw_file in raw_files:
        if raw_file.replace('raw', 'labeled') not in labeled_files:
            print(f'rm {raw_file}')
            os.system(f'rm {raw_file}')
    
## Rename files to get rid of raw, and make longer format
for directory in dirs:
    print(f"Working in directory {directory}")
    raw_files = glob.glob(directory + '/img_raw*.png')
    for raw_file in raw_files:
        frame_number = int(os.path.basename(raw_file).split('raw')[1].split('.')[0][1:])
        new_file = raw_file.split('raw')[0][:-1] + f'{frame_number:05d}' + '.png'
        print(f'mv {raw_file} {new_file}')
        os.system(f'mv {raw_file} {new_file}')

#img00347.png