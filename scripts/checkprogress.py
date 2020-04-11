#!/usr/bin/env python

import glob
import os
import argparse


def get_sorted_chunk_ids(dirs):
    ids = []
    for d in dirs:
        for f in glob.glob(os.path.join(d, "training.*.gz")):
            ids.append(int(os.path.basename(f).split('.')[-2]))
    ids.sort(reverse=True)
    return ids


def main(argv):
    a = get_sorted_chunk_ids([argv.input])
    with open(argv.progress, "r") as file:
        threshold = int(file.read())
    if a[0] < threshold:
        exit(1)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=\
            'Link input to test/train subdirectories of output in 10:90 ratio.')
    argparser.add_argument('-i', '--input', type=str, help='input directory')
    argparser.add_argument('-p', '--progress', type=str, help='progress file')

    main(argparser.parse_args())
