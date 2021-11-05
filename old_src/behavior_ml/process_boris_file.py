#Process BORIS file

import glob 
import numpy as np

files = sorted(glob.glob('./data/boris/DLC_Set2_*.csv'))

#Reset time stamps

# e3v813a-20210714T091732-092358.avi.re-encoded.1600px.16265k.avi
# e3v813a-20210714T092722-093422.avi.re-encoded.1600px.16265k.avi
# e3v813a-20210714T094126-094745.avi.re-encoded.1600px.16265k.avi
# e3v813a-20210714T095234-095803.avi.re-encoded.1600px.16265k.avi
# e3v813a-20210714T095950-100613.avi.re-encoded.1600px.16265k.avi

durations = np.array([360+26.00,
                        360+59.97,
                        360+19.03,
                        300+29.03,
                        360+23.00])

cum_durations = [0] + list(np.cumsum(durations))[:-1]

for offset, fn in zip(cum_durations, files):
    fn_out = fn.replace('.csv', '_reoffset.csv')
    f_out = open(fn_out, 'w')
    f_in = open(fn, 'r')
    for idx, line in enumerate(f_in):
        print(idx, line)
        if idx < 16:
            f_out.write(line)
        else:
            words = line.split(',')
            new_line = ','.join([str(float(words[0])-offset)] + words[1:])
            f_out.write(new_line)
    f_in.close()
    f_out.close()