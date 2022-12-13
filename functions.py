import sys
sys.setrecursionlimit(10000)
import shelve
from random import randint
import os
import subprocess
import Bio.SeqIO as SeqIO
import re
import json
import itertools
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import numpy as np
import copy
import pandas as pd
from pprint import pprint
from tqdm import tqdm

def ltr_finding(threads):
    try:
        subprocess.run(['perl', 'ltrfinder/LTR_FINDER_parallel.pl', '-seq', 'temp/genome.fa', '-threads', f'{threads}'], check = True)
    except subprocess.CalledProcessError:
        print('Something went wrong......')
   

def pre_processing(genome_path):
    genome_data = {}

    with open(f'{genome_path}', 'r') as f:
        for record in SeqIO.parse(f, 'fasta'):
            if bool(re.compile('.*(chr).*').search(record.description)) and not bool(re.compile('.*(ando).*').search(record.description)):
                genome_data[f'{record.id}'] = str(record.seq)

    if not os.path.exists('./temp'):
        os.mkdir('./temp')

    with open(f'temp/genome.fa', 'w') as g:
        for k, v in genome_data.items():
            g.write(f'>{k.replace(".", "_")}\n{v}\n')

    del genome_data


def ltr_extract():
    genome_data = {}
    with open('temp/genome.fa', 'r') as f:
        for record in SeqIO.parse(f, 'fasta'):
            genome_data[f'{record.id}'] = record.seq

        extract = {}
        file = [f for f in os.listdir() if f.endswith('.scn')][0]
        with open(file, 'r') as g:
            for line in g:
                try: 
                    chro = str(line.split("\t")[1].strip())
                    start = int(line.split("\t")[2].strip().split("-")[0]) - 1
                    end = int(line.split("\t")[2].strip().split("-")[1]) - 1
                    if chro not in extract.keys(): extract[chro] = []
                    extract[chro].append(str(genome_data[record.id][start:end+1]))
                except ValueError:
                    continue

        for k, v in extract.items():
            extract[k] = ''.join(v)

        with open('temp/extract.fa', 'w') as h:
            for k, v in extract.items():
                if len(v):
                    h.write(f'>{k}\n{v.upper()}\n')
                else:
                    continue

    del genome_data
    del extract


def kmer_count(kmer):
    split_command = '''awk -F "|" '/^>/ {close(F) ; F = "temp/"substr($1,2,length($1)-1)".split.fa"} {print >> F}' temp/extract.fa'''
    os.system(split_command)
    try:
        file_list1 = [f for f in os.listdir('./temp') if f.endswith('.split.fa')]
        for split_file in file_list1:
            with open(f'temp/{split_file.split(".")[0]}_kmer_count.tsv', 'w') as f:
                subprocess.run(['kalnal-kmer/target/release/kalnal-kmer', f'{kmer}', f'temp/{split_file}'], stdout=f, check = True)
    except subprocess.CalledProcessError:
        print("Something went wrong with kalnal-kmer")

    kmer_data = pDict()
    file_list2 = [f for f in os.listdir('./temp') if f.endswith('.tsv')]
    for split_file in file_list2:
        with open(f'temp/{split_file}', 'r') as f:
            if split_file.split("_")[0] not in kmer_data.keys(): kmer_data[split_file.split("_")[0]] = pDict()
            for line in f:
                kmer_data[split_file.split("_")[0]][line.split('\t')[0].strip()] = line.split('\t')[1].strip()

    with open(f'temp/K{kmer}.combine.json', 'w') as f:
        json.dump(kmer_data, f)

    for r in [f for f in os.listdir('./temp') if f.endswith('*.tsv')]:
        os.remove(r)
    for r in [f for f in os.listdir('./temp') if f.endswith('*.split.fa')]:
        os.remove(r)

    del kmer_data

class pDict(object):
    #=====================================================================================
    FOLER='./temp'
    INT_PREFIX='__int__'
    #=====================================================================================
    def _open(self):
        if not os.path.isdir(self.FOLER):
            self.FOLER = './temp'
        if not os.path.isdir(self.FOLER):
            raise IOError('Cannot wirte at folder "%s"' % self.FOLER)
        while (True):
            self.filename = '%s/.%08d.__dict__' % (self.FOLER, randint(1,99999999))
            if not os.path.exists(self.filename):
                break
        self.d = shelve.open(self.filename)
    #=====================================================================================
    def _close(self, is_delete=True):
        if self.d is not None:
            self.d.close()
            self.d = None
        if is_delete and os.path.exists(self.filename):
            os.remove(self.filename)
    #=====================================================================================
    def __init__(self, d={}):
        self.filename = None
        self.d = None
        self._open()
        if d and not isinstance(d, (dict, pDict)):
            raise ReferenceError('pDict construct need only dict or pDict type but <%s>'
                % str(type(d)))
        for k, v in d.items():
            self.__setitem__(k, v)
    #=====================================================================================
    def __del__(self):
        self._close()
    #=====================================================================================
    def __repr__(self):
        rl = list()
        rl.append('{')
        for i,k in enumerate(sorted(self.d.keys())):
            if i > 0: rl.append(',')
            rk = self.__r_keytransform__(k)
            if isinstance(rk, str):
                rk = '"%s"' % rk
            rv = self.d[k]
            if isinstance(rv, str):
                rv = '"%s"' % rv
            rl.append('%s:%s'%(rk,rv))
        rl.append('}')
        return ''.join(rl)
    #=====================================================================================
    def __getitem__(self, key):
        if isinstance(key, slice):
            return [ self.d[self.__keytransform__(k)]
                     for k in range(key.start, key.stop, key.step) ]
        return self.d[self.__keytransform__(key)]
    #=====================================================================================
    def __setitem__(self, key, value):
        self.d[self.__keytransform__(key)] = value
    #=====================================================================================
    def __delitem__(self, key):
        r = self.d[self.__keytransform__(key)]
        del self.d[self.__keytransform__(key)]
        return r
    #=====================================================================================
    def __iter__(self):
        return iter(self.d)
    #=====================================================================================
    def __len__(self):
        return len(self.d)
    #=====================================================================================
    def __keytransform__(self, key):
        if isinstance(key, str):
            return key
        if isinstance(key, (int, int)):
            return '%s%10d' % (self.INT_PREFIX, key)
        return str(key)
    #=====================================================================================
    def __r_keytransform__(self, key):
        if key.startswith(self.INT_PREFIX):
            return int(key[len(self.INT_PREFIX):].strip())
        return key
    #=====================================================================================
    def __contains__(self, key):
        # return self.__keytransform__(key) in self.d
        return self.has_key(key)
    #=====================================================================================
    def has_key(self, key):
        return self.d.has_key(self.__keytransform__(key))
    #=====================================================================================
    def keys(self):
        # for k in self.d.keys():
        #     yield self.__r_keytransform__(k)
        return [ self.__r_keytransform__(k) for k in sorted(self.d.keys()) ]
    #=====================================================================================
    def values(self):
        vl = list()
        for k in sorted(self.d.keys()):
            # yield self.d[k]
            vl.append(self.d[k])
        return vl
    #=====================================================================================
    def items(self):
        for k in sorted(self.d.keys()):
            yield self.__r_keytransform__(k), self.d[k]


class pList(pDict):
    #=====================================================================================
    def __init__(self, l=[]):
        pDict.__init__(self)
        self.len = 0
        if l and not isinstance(l, (list, pList)):
            raise ReferenceError('pList construct need only list or pList type but <%s>'
                                 % str(type(l)))
        for v in l:
            self.append(v)
    #=====================================================================================
    def __repr__(self):
        rl = list()
        rl.append('[')
        for i in range(self.len):
            if i > 0: rl.append(',')
            rv = self.d[self.__keytransform__(i)]
            if isinstance(rv, str):
                rv = '"%s"' % rv
            rl.append('%s'%rv)
        rl.append(']')
        return ''.join(rl)
    #=====================================================================================
    def __setitem__(self, ndx, value):
        if ndx < 0 or ndx > self.len:
            raise IndexError('Invalid index <%s>' % ndx)
        self.d[self.__keytransform__(ndx)] = value
    #=====================================================================================
    def __delitem__(self, ndx):
        if ndx < 0 or ndx >= self.len:
            raise IndexError('Invalid index <%s>' % ndx)
        r = self.d[self.__keytransform__(ndx)]
        del self.d[self.__keytransform__(ndx)]
        for i in range(ndx, self.len-1):
            self.d[self.__keytransform__(i)] = self.d[self.__keytransform__(i+1)]
        self.len -= 1
        if self.len > 0:
            del self.d[self.__keytransform__(self.len)]
        return r
    #=====================================================================================
    def __contains__(self, v):
        return v in self.values()
    #=====================================================================================
    def __iter__(self):
        return iter(self.values())
    #=====================================================================================
    def append(self, v):
        self.__setitem__(self.len, v)
        self.len += 1
    #=====================================================================================
    def extend(self, l):
        for item in l:
            self.append(item)
        self.len += len(l)
    #====================================================================================



def dict2array(d, kmer):
    kmer_list = pList()
    for i in itertools.product(['A', 'T', 'G', 'C', 'N'], repeat=kmer):
        kmer_list.append(''.join(i))
    
    print(len(kmer_list))
    print("step1")
    foo = copy.copy(d)

    for k1, v1 in foo.items():
        for mer in kmer_list:
            print(mer)
            if not d[k1].get(mer): d[k1][mer] = 0

    for i, (k1, v1) in enumerate(d.items()):
        if i == 0:
            tmp = len(v1)
        else:
            try: 
                assert tmp == len(v1)
            except AssertionError:
                print(f"{tmp} does not match with {len(v1)}")

    print("step2")

    for k1, v1 in d.items():
        d[k1] = sorted(v1.items())
    
    print("step3")

    #final_data = pDict()
    #for k, v in d.items():
    #    for mer, n in v:
    #        if k not in final_data.keys(): final_data[k] = pList()
    #        final_data[k].append(n)
   

    final_data = pList()
    names = []
    for k, v in d.items():
        temp = []
        for mer, n in v:
            temp.append(int(n))
        final_data.append(temp)
        names.append(str(k))

    del d
    
    print(np.shape(final_data))
    return final_data, names


def ploting(kmer):
    with open(f'temp/K{kmer}.combine.json') as f:
        data = pDict(json.load(f))

    plt.figure(figsize=(30, 20))
    data, names = dict2array(data, kmer)
    linked = linkage(data, method='ward')
    dend = dendrogram(linked, orientation='top', distance_sort='descending', labels=names, show_leaf_counts=True)

    plt.savefig(f'{kmer}_analyzed.png')


def finalize():
    if  len([f for f in os.listdir() if f.endswith('.scn')]):
        os.remove([f for f in os.listdir() if f.endswith('.scn')][0])
        os.remove([f for f in os.listdir() if f.endswith('.gff3')][0])
        os.remove([f for f in os.listdir() if f.endswith('.list')][0])
        for r in [f for f in os.listdir() if f.startswith('genome')]:
            os.remove(r)

    if os.path.exists('./temp'):
        for r in os.listdir('./temp'):
            os.remove(f'temp/{r}')
        os.rmdir('./temp')


def analyze(args):
    #pre_processing(args.genome)
    #ltr_finding(args.threads)
    #ltr_extract()
    kmer = int(args.kmer)
    #for k in [kmer-4, kmer-2, kmer]:
        #kmer_count(k)
        #ploting(k)
    kmer_count(kmer)
    ploting(kmer) # TEST for pList, pDict
    finalize()
