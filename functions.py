import os
import subprocess
import Bio.SeqIO as SeqIO
import re

def ltr_finding(genome_path, threads):
    try:
        subprocess.run(['perl', 'ltrfinder/LTR_FINDER_parallel.pl', '-seq', f'{genome_path}', '-threads', f'{threads}'], check = True)
    except subprocess.CalledProcessError:
        print('Something went wrong......')
   
   return 


def pre_processing(genome_path):
    genome_data = []

    with open(f'{genome_path}', 'r') as f:
        for i, record in enumerate(SeqIO.parse(f, 'fasta')):
            if not (bool(re.compile('.*(hlo|ito|onti).*').search(record.description)))
                genome_data[record.id] = str(record.seq)

    if os.path.exists('./temp'):
        os.makedir('./temp')

    with open(f'temp/genome.fa', 'w') as f:
        for k, v in genome_data.items():
            print(f'>{k.replace(".", "/")}\n{v}\n')


def ltr_extract():
    with open('temp/genome.fa', 'r') as f:
        with open('*.scn', 'r') as g:




def kmer_count():


def finalize():
    if os.path.exists('*.scn'):
        os.remove('*.scn')
        os.remove('*.gff3')
        os.remove('*.list')

    if os.path.exists('./temp'):
        os.remove('temp/*')
        os.rmdir('./temp')

