#! /usr/bin/env python3

import argparse
from functions import analyze
import os
os.chdir('/data/HS_graduation/kalnal/')

def parse():
    parser = argparse.ArgumentParser(description="Kalnal - The polyploid subgenome divider", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--genome', type=str, default=None, help='Scaffolded target genome file')
    parser.add_argument('-k', '--kmer', type=int, default=21, help='parameter for k-mer analysis, (k-4), (k-2), k will be analyzed')
    parser.add_argument('-t', '--threads', type=int, default=12, help='number of threads for calculation')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse()

    analyze(args)
