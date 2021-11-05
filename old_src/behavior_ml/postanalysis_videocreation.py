import numpy as np
import pandas as pd
import hashlib
import os 

from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

#vid_in = './data/dlc_w_crop/videos/e3v813a-20210610T122758-123309DLC_resnet50_pilot_studyJun15shuffle1_50000_el_filtered_id_labeled.mp4'
#vid_out = './data/dlc_w_crop/videos/behavior_labeled/e3v813a-20210610T122758-123309DLC_resnet50_pilot_studyJun15shuffle1_50000_el_filtered_id_labeled.mp4'

vid_in = './data/dlc_improved/videos/e3v813a-20210610T120637-121213DLC_dlcrnetms5_pilot_studyAug31shuffle1_100000_el_filtered_id_labeled.mp4'
vid_out = './data/dlc_improved/videos/behavior_labeled/e3v813a-20210610T120637-121213DLC_dlcrnetms5_pilot_studyAug31shuffle1_100000_el_filtered_id_labeled.mp4'
boris_in = './data/boris/DLC1.csv'
hashed = 'c6b29b1c'

# vid_in = './data/dlc_improved/videos/e3v813a-20210610T122758-123309DLC_resnet50_pilot_studyAug17shuffle1_50000_el_filtered_id_labeled.mp4'
# vid_out = './data/dlc_improved/videos/behavior_labeled/e3v813a-20210610T122758-123309DLC_resnet50_pilot_studyAug17shuffle1_50000_el_filtered_id_labeled.mp4'
#boris_in = './data/boris/DLC4.csv'
#hashed = '5b63f715'

rate = 1/30

#Load in MABE results
behavior_results_in = './results/submission_mars_distr_stacked_w_1dcnn_ml_xgb_paramset_default_hmm.npy'

#Load in BORIS results
#boris_in = './data/boris/DLC4.csv'

mabe_labels = np.load(behavior_results_in,allow_pickle=True).item()

boris_labels = pd.read_csv(boris_in, skiprows = 15)
boris_labels['index'] = (boris_labels.index//2)
boris_labels = boris_labels.pivot_table(index = 'index', columns = 'Status', values = 'Time').reset_index()
boris_labels = list(np.array(boris_labels[['START', 'STOP']]))
boris_labels = [list(i) for i in boris_labels]

#Pull out annotation for the video we want
predictions = np.array(mabe_labels[hashed]) 
#Simplify into behavior/no-behavior
predictions = list(1 - (predictions == 3))

#Put into start/stop pairs
pairs = []
in_pair = False
start = 0
for idx, behavior in enumerate(predictions):
    if behavior == 0:
        if in_pair:
            pairs.append([start, idx*rate])
        in_pair = False
    if behavior == 1:
        if not in_pair:
            start = idx*rate
        in_pair = True

if in_pair:
    pairs.append([start, idx*rate])

print(boris_labels)
print(pairs)

boris_label_string = ','.join([f"drawtext=text=\'interaction\':x=90:y=90:fontsize=36:fontcolor=red:enable=\'between(t,{pair[0]},{pair[1]})\'" for pair in boris_labels])
mabe_label_string = ','.join([f"drawtext=text=\'predicted\':x=90:y=60:fontsize=36:fontcolor=green:enable=\'between(t,{pair[0]},{pair[1]})\'" for pair in pairs])
label_string = boris_label_string + ',' + mabe_label_string

#print(label_string)
#Prepare the ffmpeg command
cmd = f'ffmpeg -y -i {vid_in} -vf "{label_string}" {vid_out}'

os.system(cmd)

################################################################################

#Also, pull out some stats from the model

#We have the predictions, in predictions, need to convert the pairs to a list of values for every frame, too

ground_truth = np.zeros(len(predictions))
for start, end in boris_labels:
    ground_truth[int(start/rate):int(end/rate)] = 1

print('Precision:', precision_score(ground_truth, predictions))
print('Recall:', recall_score(ground_truth, predictions))
print('F1 score:', f1_score(ground_truth, predictions))
print('Accuracy score:', accuracy_score(ground_truth, predictions))

#Extract just a short part of the video
cmd2 = f'ffmpeg -y -i {vid_out} -ss 210 -t 300 -c copy dlc2_example.mp4'
os.system(cmd2)