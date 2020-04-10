"""Find mutual NNs."""

from argparse import ArgumentParser


def load_dict(filename):
    d = {}
    with open(filename, 'r') as f:
        for line in f:
            k, v = line.strip().split()
            d[k] = v
    return d


def main():
    parser = ArgumentParser()
    parser.add_argument('dict1')
    parser.add_argument('dict2')
    parser.add_argument('output')
    args = parser.parse_args()

    dict1 = load_dict(args.dict1)
    dict2 = load_dict(args.dict2)
    n = 0
    with open(args.output, 'w') as f:
        for w1 in dict1:
            w2 = dict1[w1]
            if dict2[w2] == w1:
                n += 1
                print(w1, w2, file=f)
    print('Found %d translation pairs.' % n)


if __name__ == '__main__':
    main()
