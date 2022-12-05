import itertools
import numpy as np
import collections
import copy

def dict2ndarray(d, kmer):
    kmer_list = []
    for i in itertools.product(['A', 'T', 'G', 'C', 'N'], repeat=kmer):
        kmer_list.append(''.join(i))

    real_data = copy.deepcopy(d)
    for k1, v1 in d.items():
        for mer in kmer_list:
            if mer not in v1.keys(): real_data[k1][mer] = 0
    
    for k1, v1 in real_data.items():
        real_data[k1] = sorted(v1.items())

    del d

    return np.fromiter(sorted(real_data.items()), dtype=object, count=len(real_data))

data = {'A1':{'AAT':1, 'AAA':2, 'AAC':4}, 'B2':{'AAT':2, 'AAC':5}}

print(dict2ndarray(data, 3))
