import os
import cv2
import pandas as pd 
import numpy as np 

dirs = ['e3v813a-20210610T120637-121213',
        'e3v813a-20210610T121558-122141',
        'e3v813a-20210610T122332-122642',
        'e3v813a-20210610T122758-123309',
        'e3v813a-20210610T123521-124106']

for dir_name in dirs:

    print("Processing", dir_name)

    labels_path = './labeled-data/' + dir_name + '/CollectedData_Brett.csv'
    labels = pd.read_csv(labels_path, header = [0,1,2,3])
    frames_ = np.squeeze(labels['scorer'].values)
    frames = [int(i.split('/')[2].split('.')[0][3:]) for i in frames_]

    videopath = './videos/' + dir_name + '.avi'
    cap = cv2.VideoCapture(videopath)

    for frame_number in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
        res, frame = cap.read()
        #Save as png: img00639.png 
        out_png = f'./labeled-data/{dir_name}/img{frame_number:05d}.png'
        cv2.imwrite(out_png, frame)
    cap.release()