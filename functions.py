import sys
sys.setrecursionlimit(10000)

import os
import subprocess
import Bio.SeqIO as SeqIO
import re
import json
import itertools
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import copy
import pandas as pd


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

    kmer_data = {}
    file_list2 = [f for f in os.listdir('./temp') if f.endswith('.tsv')]
    for split_file in file_list2:
        with open(f'temp/{split_file}', 'r') as f:
            if split_file.split("_")[0] not in kmer_data.keys(): kmer_data[split_file.split("_")[0]] = {}
            for line in f:
                kmer_data[split_file.split("_")[0]][line.split('\t')[0].strip()] = line.split('\t')[1].strip()

    with open(f'temp/K{kmer}.combine.json', 'w') as f:
        json.dump(kmer_data, f)

    for r in [f for f in os.listdir('./temp') if f.endswith('*.tsv')]:
        os.remove(r)
    for r in [f for f in os.listdir('./temp') if f.endswith('*.split.fa')]:
        os.remove(r)

    del kmer_data
    

def dict2dict(d, kmer):
    kmer_list = []
    for i in itertools.product(['A', 'T', 'G', 'C', 'N'], repeat=kmer):
        kmer_list.append(''.join(i))

    for k1, v1 in d.copy().items():
        for mer in kmer_list:
            if mer not in v1.keys(): d[k1][mer] = 0

    for k1, v1 in d.items():
        d[k1] = sorted(v1.items())
    
    final_data = {}
    for k, v in d.items():
        for mer, n in v:
            if k not in final_data.keys(): final_data[k] = []
            final_data[k].append(n)
    
    del d

    return final_data


def ploting(kmer):
    with open(f'temp/K{kmer}.combine.json') as f:
        data = dict(json.load(f))

    plt.figure(figsize=(30, 20))
    data = dict2dict(data, kmer)
    data = pd.DataFrame(data)
    linked = linkage(data.T, method='ward')
    dend = dendrogram(linked, orientation='top', distance_sort='descending', labels=list(data.T.index), show_leaf_counts=True)

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
    pre_processing(args.genome)
    ltr_finding(args.threads)
    ltr_extract()
    kmer = int(args.kmer)
    for k in [kmer-4, kmer-2, kmer]:
        kmer_count(k)
        ploting(k)
    finalize()
