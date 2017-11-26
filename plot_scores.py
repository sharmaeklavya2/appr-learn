#!/usr/bin/env python3

import argparse
from matplotlib import pyplot as plt

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fpath', help='Path to scores file')
    parser.add_argument('window_size', type=int)
    parser.add_argument('skip_size', type=int)
    args = parser.parse_args()

    with open(args.fpath) as fobj:
        scores = [float(x) for x in fobj.read().split()]

    n = len(scores)
    scores2 = []
    for i in range(0, n-args.window_size+1, args.skip_size):
        scores2.append(sum(scores[i: i+args.window_size]))
    for i in range(len(scores2)):
        scores2[i] /= args.window_size

    plt.plot(scores2)
    plt.show()

if __name__ == '__main__':
    main()
