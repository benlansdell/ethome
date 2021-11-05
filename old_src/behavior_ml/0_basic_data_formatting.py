import numpy as np
import pandas as pd
from lib.helper import colnames

test = np.load('data/test_inference.npy',allow_pickle=True).item()

def make_df(raw_data, add_annotations = True):
    """
    Basic preprocessing of data:
    * Turn dictionry into a pandas dataframe
    * Add annotator column to DF -- stored as string
    * Name columns according to body part, etc
    """
    df = []
    labels = []
    seqs = []
    for seq_id in raw_data['sequences']:
        pts = raw_data['sequences'][seq_id]['keypoints']
        for idx in range(len(pts)):
            data = pts[idx].flatten()
            df.append(list(data))
        if 'annotations' in raw_data['sequences'][seq_id] and add_annotations:
            labels += list(train['sequences'][seq_id]['annotations'])
        seqs += [seq_id]*len(pts)

    #Make this a dataframe
    df_ = pd.DataFrame(df, columns = colnames)
    if len(labels) > 0 and add_annotations:
        df_['annotation'] = labels
    df_['seq_id'] = seqs
    return df_

test_df = make_df(test)
test_df.to_csv('./data/intermediate/test_df.csv')