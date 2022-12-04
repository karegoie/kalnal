import os
import subprocess

def ltr_finding(genome_path, threads):
    try:
        subprocess.run(['perl', 'LTR_FINDER_parallel', '-seq', f'{genome_path}', '-threads', f'{threads}'], check = True)
    except subprocess.CalledProcessError:
        print('Something went wrong......')
   
   return 

def ltr_processing(
