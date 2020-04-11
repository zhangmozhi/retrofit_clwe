import argparse
import os

# Create the files required for the vocab directory.

parser = argparse.ArgumentParser()
parser.add_argument('--emb', default='',
                    help='concatenated CLWE')
parser.add_argument('--vocab_path', default='',
                    help='path to save vocab')
args = parser.parse_args()

f_vocab = open(os.path.join(args.vocab_path, "vocab.txt"), "w")
f_namespaces = open(os.path.join(args.vocab_path, "non_padded_namespaces.txt"), "w")

# add OOV
oov_token = "@@UNKNOWN@@"
f_vocab.write("%s\n" % oov_token)

# Read embedding & add vocabs
for line in open(args.emb):
    word = line.strip().split()[0]
    f_vocab.write("%s\n" % word)
