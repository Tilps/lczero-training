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
    n = min(argv.wsize, len(a))
    if argv.trim:
        if n < len(a):
            print("Trimming {}".format(len(a)-n))
        for i in a[n:]:
            os.remove(os.path.join(argv.input, "training.{}.gz".format(i)))
    existing_test = get_sorted_chunk_ids([os.path.join(argv.output, "test")])
    existing_train = get_sorted_chunk_ids([os.path.join(argv.output, "train")])
    new_test = []
    new_train = []
    for i in a[:n]:
        if i % 100 >= 90:
            new_test.append(i)
        else:
            new_train.append(i)
    added_test = set(new_test) - set(existing_test)
    added_train = set(new_train) - set(existing_train)
    removed_test = set(existing_test) - set(new_test)
    removed_train = set(existing_train) - set(new_train)
    print("Will do: +{}, -{}, +{}, -{}".format(len(added_test), len(removed_test), len(added_train), len(removed_train)))
    for i in added_test:
        os.link(os.path.join(argv.input, "training.{}.gz".format(i)),
                os.path.join(argv.output, "test/training.{}.gz".format(i)))
    for i in removed_test:
        os.remove(os.path.join(argv.output, "test/training.{}.gz".format(i)))
    for i in added_train:
        os.link(os.path.join(argv.input, "training.{}.gz".format(i)),
                os.path.join(argv.output, "train/training.{}.gz".format(i)))
    for i in removed_train:
        os.remove(os.path.join(argv.output, "train/training.{}.gz".format(i)))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=\
            'Link input to test/train subdirectories of output in 10:90 ratio.')
    argparser.add_argument('-i', '--input', type=str, help='input directory')
    argparser.add_argument(
        '-w',
        '--wsize',
        type=int,
        help=
        'window size - should be padded a bit to ensure both sides of split exceed fraction of target'
    )
    argparser.add_argument('-t', '--trim', type=bool, help='trim the input directory to window size as well')
    argparser.add_argument('-o', '--output', type=str, help='output directory')

    main(argparser.parse_args())
