"""Lowercase words and cap embedding size."""

from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument('--vocab', type=int, default=200000)
    parser.add_argument('--dim', type=int, default=300)
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    vocab = set()
    in_f = open(args.input, 'r')
    out_f = open(args.output, 'w')
    print(args.vocab, args.dim, file=out_f)
    for line in in_f:
        tokens = line.lower().split()
        if len(tokens) == 2:  # skip header
            continue
        if tokens[0] not in vocab:
            vocab.add(tokens[0])
            print(line.lower().strip(), file=out_f)
        if len(vocab) == args.vocab:
            break 


if __name__ == '__main__':
    main()
