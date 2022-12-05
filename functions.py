import os
import subprocess
import Bio.SeqIO as SeqIO
import re
import json
import itertools
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import copy
import numpy as np

def ltr_finding(genome_path, threads):
    try:
        subprocess.run(['perl', 'ltrfinder/LTR_FINDER_parallel.pl', '-seq', f'{genome_path}', '-threads', f'{threads}'], check = True)
    except subprocess.CalledProcessError:
        print('Something went wrong......')
   
   return 


def pre_processing(genome_path):
    genome_data = []

    with open(f'{genome_path}', 'r') as f:
        for record in SeqIO.parse(f, 'fasta'):
            if not (bool(re.compile('.*(hlo|ito|onti).*').search(record.description)))
                genome_data[record.id] = str(record.seq)

    if os.path.exists('./temp'):
        os.makedir('./temp')

    with open(f'temp/genome.fa', 'w') as g:
        for k, v in genome_data.items():
            g.write(f'>{k.replace(".", "_")}\n{v}\n')

    del genome_data

    return


def ltr_extract():
    genome_data = []
    with open('temp/genome.fa', 'r') as f:
        for record in SeqIO.parse(f, 'fasta'):
            genome_data[record.id] = record.seq

        extract = {}
        with open('*.scn', 'r') as g:
            for line in g:
                chro = str(line.split("\t")[1].strip())
                start = int(line.split("\t")[2].strip().split("-")[0]) - 1
                end = int(line.split("\t")[2].strip().split("-")[1]) - 1
                if chro not in extract.keys(): extract[chro] = ''
                extract[chro] += genome_data[record.id][start:end+1]

        with open('temp/extract.fa', 'w') as h:
            for k, v in extract.items():
                h.write(f'>{k}\n{v.upper()}\n')

    del genome_data
    del extract


def kmer_count(kmer):
    split_command = '''awk -F "|" '/^>/ {close(F) ; F = "temp/"substr($1,2,length($1)-1)".split.fa"} {print >> F}' temp/extract.fa'''
    os.system(split_command)
    try:
        for split_file in os.listdir('temp/*.split.fa'):
            subprocess(['kalnal-kmer/target/release/kalnal', kmer, f'{split_file}', '>', 'temp/{split_file.split(".")[0].split("/")[1]}_kmer_count.tsv'], check = True)
    except subprocess.CalledProcessError:
        print("Something went wrong with kalnal-kmer")

    kmer_data = {}
    for split_file in os,listdir('temp/*.tsv')
        with open(split_file, 'r') as f:
            if split_file.split("_")[0] not in kmer_data.keys(): kmer_data[split_file.split("_")[0]] = {}
            for line in f:
                kmer_data[split_file.split("_")[0]][line.split('\t')[0].strip()] = line.split('\t')[1].strip()

    with open(f'temp/K{kmer}.combine.json', 'w') as f:
        json.dump(kmer_data, f)

    os.remove('temp/*.tsv')
    os.remove('temp/*.split.fa')
    del kmer_data
    
    return


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


def ploting(kmer):
    data = dict(json.load(f'temp/K{kmer}.combine.json'))

    plt.figure(figsize=(30, 20))
    data = dict2ndarray(data, kmer)
    linked = linkage(data, method='ward')
    dend = dendrogram(linked, orientation='top', distance_sort='descending', labels=list(df.T.index), show_leaf_counts=True)

    plt.savefig(f'{kmer}_analyzed.png')


def finalize():
    if os.path.exists('*.scn'):
        os.remove('*.scn')
        os.remove('*.gff3')
        os.remove('*.list')

    if os.path.exists('./temp'):
        os.remove('temp/*')
        os.rmdir('./temp')


def analyze(args):
    pre_processing(args.genome)
    ltr_finding(args.genome, args.threads)
    ltr_extract()
    kmer = int(args.kmer)
    for k in [kmer-4, kmer-2, kmer, kmer+2, kmer+4]:
        kmer_count(k)
        ploting(k)
    finalize()




