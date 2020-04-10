"""Translate words with CSLS scores."""


from argparse import ArgumentParser
import collections
import io

import logging
import numpy as np


def load_vectors(fname, maxload=200000, norm=True, center=False, verbose=True):
    if verbose:
        print("Loading vectors from %s" % fname)
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    if maxload > 0:
        n = min(n, maxload)
    x = np.zeros([n, d])
    words = []
    for i, line in enumerate(fin):
        if i >= n:
            break
        tokens = line.rstrip().split(' ')
        words.append(tokens[0])
        v = np.array(tokens[1:], dtype=float)
        x[i, :] = v
    if norm:
        x /= np.linalg.norm(x, axis=1)[:, np.newaxis] + 1e-8
    if center:
        x -= x.mean(axis=0)[np.newaxis, :]
        x /= np.linalg.norm(x, axis=1)[:, np.newaxis] + 1e-8
    if verbose:
        print("%d word vectors loaded" % (len(words)))
    return words, x


def idx(words):
    w2i = {}
    for i, w in enumerate(words):
        if w not in w2i:
            w2i[w] = i
    return w2i


def load_lexicon(filename, words_src, words_tgt, verbose=True):
    f = io.open(filename, 'r', encoding='utf-8')
    lexicon = collections.defaultdict(set)
    idx_src , idx_tgt = idx(words_src), idx(words_tgt)
    vocab = set()
    for line in f:
        word_src, word_tgt = line.split()
        if word_src in idx_src and word_tgt in idx_tgt:
            lexicon[idx_src[word_src]].add(idx_tgt[word_tgt])
        vocab.add(word_src)
    f.close()
    if verbose:
        coverage = len(lexicon) / float(len(vocab))
        print("Coverage of source vocab: %.4f" % (coverage))
    return lexicon, float(len(vocab))


def get_translations(x_src, x_tgt, bsz=1000, k=10):
    nns = []
    for i0 in range(0, x_src.shape[0], bsz):
        j0 = min(i0 + bsz, x_src.shape[0])
        sr = x_src[i0:j0]
        sc = np.dot(sr, x_tgt.T)
        similarities = 2 * sc
        sc2 = np.zeros(x_tgt.shape[0])
        for i in range(0, x_tgt.shape[0], bsz):
            j = min(i + bsz, x_tgt.shape[0])
            sc_batch = np.dot(x_tgt[i:j, :], x_src.T)
            dotprod = np.partition(sc_batch, -k, axis=1)[:, -k:]
            sc2[i:j] = np.mean(dotprod, axis=1)
        similarities -= sc2[np.newaxis, :]
        nns.extend(np.argmax(similarities, axis=1).tolist())
    return nns


def main():
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    parser = ArgumentParser()
    parser.add_argument('--src', required=True, help='source embeddings')
    parser.add_argument('--tgt', required=True, help='target embeddings')
    parser.add_argument('--max_vocab', type=int, default=200000,
                        help='vocab size')
    parser.add_argument('--start', default=0, type=int, help='start index')
    parser.add_argument('--end', default=-1, type=int, help='end index')
    parser.add_argument('--output', required=True, help='path to output')
    args = parser.parse_args()
    logging.info(vars(args))

    words_src, x_src = load_vectors(args.src, maxload=args.max_vocab)
    words_tgt, x_tgt = load_vectors(args.tgt, maxload=args.max_vocab)

    logging.info('Generating nearest neighbors')
    nn_src = get_translations(x_src[args.start:args.end, :], x_tgt)

    logging.info('Save dictionaries')
    fout = io.open(args.output, 'w', encoding='utf-8')
    for i in range(len(nn_src)):
        fout.write(u'%s %s\n' % (words_src[args.start + i], words_tgt[nn_src[i]]))
    fout.close()


if __name__ == '__main__':
    main()
