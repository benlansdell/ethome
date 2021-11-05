#Convert csv to h5 for labeled images
import pandas as pd

scorer = "Brett"

# to_convert = ['e3v813a-20210714T091732-092358',
#                 'e3v813a-20210714T092722-093422',
#                 'e3v813a-20210714T094126-094745',
#                 'e3v813a-20210714T095234-095803',
#                 'e3v813a-20210714T095950-100613']

to_convert = ['e3v813a-20210610T120637-121213',
                'e3v813a-20210610T122332-122642',
                'e3v813a-20210610T123521-124106',
                'e3v813a-20210610T121558-122141',
                'e3v813a-20210610T122758-123309']
#video = to_convert[0]

for video in to_convert:

    fn = f'./labeled-data/{video}/CollectedData_{scorer}.csv'

    with open(fn) as datafile:
        next(datafile)
        if "individuals" in next(datafile):
            header = list(range(4))
        else:
            header = list(range(3))

    data = pd.read_csv(fn, index_col=0, header=header)
    data.columns = data.columns.set_levels([scorer], level="scorer")
    data = data.sort_index()
    data.to_hdf(fn.replace(".csv", ".h5"), key="df_with_missing", mode="w")


