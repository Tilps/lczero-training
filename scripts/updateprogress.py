#!/usr/bin/env python

import argparse


def main(argv):
    with open(argv.progress, "r") as file:
        new_target = int(file.read()) + argv.increment
        print(new_target)
    with open(argv.progress, "w") as file:
        file.seek(0)
        file.truncate()
        file.write(str(new_target))
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=\
            'Link input to test/train subdirectories of output in 10:90 ratio.')
    argparser.add_argument('-p', '--progress', type=str, help='progress file')
    argparser.add_argument('-i', '--increment', type=int, help='Amount to increment progress')

    main(argparser.parse_args())
