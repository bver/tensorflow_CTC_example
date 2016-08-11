
import numpy as np
import json
import os

# Bad ideas:
# import linecache 
# line = linecache.getline('my_data/normalized.asc', line_no) 
# (File is in memory.)

# memory consumed:
# x = np.loadtxt('normalized.asc')

with open('my_data/tran.json') as json_file:    
    tran = json.load(json_file)
print('buckets:', len(tran['buckets']))

data_idx = [0]
with open('my_data/normalized.asc') as data_file:
    while data_file.readline():
        data_idx.append(data_file.tell())
print('read', len(data_idx), 'lines')

classes = " abcdefghijklmnopqrstuvwxyz'"
i = 0
classmap = {}
for c in classes:
    classmap[c] = i
    i += 1

sub = 0
fno = 0
with open('my_data/normalized.asc') as data_file:
    for bucket in tran['buckets']:
        for clue in bucket:
            from_time, to_time, text = clue
            mfcc = []
            data_file.seek(data_idx[from_time])
            for line_no in range(to_time - from_time): # not including to_time
                line = data_file.readline() 
                mfcc.append([ float(item) for item in line.split() ])
            char_y = []    
            for c in text:
                if c in classmap:
                    char_y.append(classmap[c])
            if not os.path.exists("mfcc/%d" % (sub)):
              os.makedirs("mfcc/%d" % (sub))
              print ("making mfcc/%d" % (sub))
            if not os.path.exists("char_y/%d" % (sub)):
              os.makedirs("char_y/%d" % (sub))
              print ("making char_y/%d" % (sub))
            np.save("mfcc/%d/%d.npy" % (sub, fno), np.array(mfcc))
            np.save("char_y/%d/%d.npy" % (sub, fno), np.array(char_y))
            if fno == 1000:
              fno = 0
              sub += 1
            else:
              fno += 1

