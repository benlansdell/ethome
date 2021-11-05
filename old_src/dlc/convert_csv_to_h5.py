#Convert filtered poses csvs to h5 format
fn_in = 'e3v813a-20210610T120637-121213DLC_resnet50_pilot_studyJun15shuffle1_50000_el_filtered.csv'
fn_out = 'e3v813a-20210610T120637-121213DLC_resnet50_pilot_studyJun15shuffle1_50000_el_filtered.h5'

import pandas as pd

df = pd.read_csv(fn_in, header = [0,1,2,3], index_col = 0)
df = df.set_index(('scorer', 'individuals', 'bodyparts', 'coords'))
df.columns = df.columns.set_names(['scorer', 'individuals', 'bodyparts', 'coords'])

#df.columns.set_levels([['adult', 'juvenile'], ['nose', 'leftear', 'rightear', 'neck', 'lefthip', 'righthip','tail'], \
#                    ['x', 'y', 'likelihood']])

df.to_hdf(fn_out, "df_with_missing", format = 'table', mode="w")