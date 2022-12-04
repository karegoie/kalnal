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
                h.write(f'>{k}\n{v}\n')

    del genome_data
    del extract


def kmer_count(kmer):
    try:
        subprocess(['kalnal-kmer/target/release/kalnal', kmer, 'temp/extract.fa', '>', 'temp/kmer_count.fa'], check = True)
    except subprocess.CalledProcessError:
        print("Something went wrong with kalnal-kmer")

# TODO: adding


def ploting():




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
    kmer_count(args.kmer)
    ploting()
    finalize()




