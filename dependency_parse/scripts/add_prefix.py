from argparse import ArgumentParser
import os

parser = ArgumentParser("")
parser.add_argument('--lang', type=str, default="en",
                    help='')
parser.add_argument('--f_in', type=str, default="PATH_TO_UD/en_ewt-ud-train.conllu",
                    help='')
args = parser.parse_args()

tgt_lang = args.lang

f_in = args.f_in
f_name = os.path.basename(f_in)
f_out_name = f_in + "_prefix.conllu"  # for allenNLP
print(f_out_name)
f_out = open(f_out_name, "w")

for line in open(f_in):
    if line.strip() and (not line.startswith("#")):
        cols = line.strip().split()
        cols[1] = tgt_lang + ":" + cols[1]
        f_out.write("\t".join(cols) + "\n")
        print("\t".join(cols))
    else:
        f_out.write(line)
