import numpy as np
import pandas as pd
import hashlib
import glob 

#files_in = ['./data/dlc/e3v813a-20210610T120637-121213DLC_resnet_50_pilot_studyJun15shuffle1_40000_bx_filtered.csv',
#            './data/dlc/e3v813a-20210610T121558-122141DLC_resnet_50_pilot_studyJun15shuffle1_40000_bx_filtered.csv',
#            './data/dlc/e3v813a-20210610T122332-122642DLC_resnet_50_pilot_studyJun15shuffle1_40000_bx_filtered.csv',
#            './data/dlc/e3v813a-20210610T122758-123309DLC_resnet_50_pilot_studyJun15shuffle1_40000_bx_filtered.csv',
#            './data/dlc/e3v813a-20210610T123521-124106DLC_resnet_50_pilot_studyJun15shuffle1_40000_bx_filtered.csv']

#files_in = ['./data/dlc_w_crop/e3v813a-20210610T120637-121213DLC_resnet50_pilot_studyJun15shuffle1_50000_el_filtered.csv',
#            './data/dlc_w_crop/e3v813a-20210610T122758-123309DLC_resnet50_pilot_studyJun15shuffle1_50000_el_filtered.csv',
#            './data/dlc_w_crop/e3v813a-20210610T121558-122141DLC_resnet50_pilot_studyJun15shuffle1_50000_el_filtered.csv',
#            './data/dlc_w_crop/e3v813a-20210610T123521-124106DLC_resnet50_pilot_studyJun15shuffle1_50000_el_filtered.csv',
#            './data/dlc_w_crop/e3v813a-20210610T122332-122642DLC_resnet50_pilot_studyJun15shuffle1_50000_el_filtered.csv']

# files_out = ['./data/dlc_w_crop/e3v813a-20210610T120637-121213DLC_resnet50_pilot_studyJun15shuffle1_50000_el_filtered_improved.csv',
#             './data/dlc_w_crop/e3v813a-20210610T122758-123309DLC_resnet50_pilot_studyJun15shuffle1_50000_el_filtered_improved.csv',
#             './data/dlc_w_crop/e3v813a-20210610T121558-122141DLC_resnet50_pilot_studyJun15shuffle1_50000_el_filtered_improved.csv',
#             './data/dlc_w_crop/e3v813a-20210610T123521-124106DLC_resnet50_pilot_studyJun15shuffle1_50000_el_filtered_improved.csv',
#             './data/dlc_w_crop/e3v813a-20210610T122332-122642DLC_resnet50_pilot_studyJun15shuffle1_50000_el_filtered_improved.csv']

files_in = glob.glob('./data/dlc_improved/*_improved.csv')
files_out = [fn.replace('.csv', '_reformatted.csv') for fn in files_in]

fn_out = '/mnt/storage2/blansdel/projects/mabe_final/mabetask1_ml/data/test_inference.npy'

scorer = 'DLC_resnet50_pilot_studyJul19shuffle1_50000'
mice = ['juvenile', 'adult']
bodyparts = ['nose', 'leftear', 'rightear', 'neck', 'lefthip', 'righthip', 'tail']
threshold = 0.2
filter_lowconf = False

#Load tracks from DLC
test_out = {'sequences':{}}
for f_idx, fn_in in enumerate(files_in):
    print('Loading', fn_in)
    df = pd.read_csv(fn_in, header = [0,1,2,3])

    #Impute missing values

    #Do some other magic here...
    #Including

    # Removing any large jerks
    # Taking out low confidence predictions and patching those over, too

    df = df.sort_index()

    if filter_lowconf:
        for m in mice:
            for bp in bodyparts:
                low_conf = df[scorer][m][bp]['likelihood'] < threshold
                df.loc[low_conf, (scorer, m, bp, 'x')] = np.nan
                df.loc[low_conf, (scorer, m, bp, 'y')] = np.nan
                df.loc[low_conf, (scorer, m, bp, 'likelihood')] = np.nan

        #And now filter
        df = df.interpolate(axis = 0, method = 'polynomial', order = 5)

    #Save for visualization
    df.to_csv(files_out[f_idx])

    dlc_tracks = np.array(df)[:,1:]
    vid_name = hashlib.md5(fn_in.encode()).hexdigest()[:8]

    #Adult, then juvenile. Same as resident then intruder... or should be flipped?
    n_rows = dlc_tracks.shape[0]
    selected_cols = [[3*i, 3*i+1] for i in range(14)]
    selected_cols = [j for i in selected_cols for j in i]
    dlc_tracks = dlc_tracks[:,selected_cols]

    #Put in shape:
    # (frame, mouse_id, x/y coord, body part)
    dlc_tracks = dlc_tracks.reshape((n_rows, 2, 7, 2))
    keypoints = dlc_tracks.transpose([0, 1, 3, 2])

    test_out['sequences'][vid_name] = {'annotator_id': 0, 'keypoints': keypoints}

#Save as a numpy file
np.save(fn_out, test_out)
