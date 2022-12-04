#! /usr/bin/env python3

import argparse
from functions import analyze

def parse():
    parser = argparse.ArgumentParser(description="GUM; the polyploid subgenome divider", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--genome', type=int, default=None, help='Scaffolded target genome file')
    parser.add_argument('-k', '--kmer', type=int, default=21, help='parameter for k-mer analysis, (k-4), (k-2), k, (k+2), (k+4) will be analyzed')


    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse()

    analyze(args)
