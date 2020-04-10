import argparse
import os


def count_words(f_name):
    f = open(f_name)
    words = 0
    for line in f:
        words += 1
    f.close()
    return words - 1


def copy_lines(f, f_out):
    for line in f:
        f_out.write(line)


def copy_lines_lang(f, f_out, lang):
    n = 0
    for line in f:
        if not lang:
            f_out.write(line)
        else:
            f_out.write(lang + ":" + line)
        n += 1


def merge_vectors(embed1, embed2, out_fname, lang1, lang2, dim):
    f_en = embed1
    f_target = embed2
    f_out = open(out_fname, "w")
    words = count_words(f_en) + count_words(f_target)
    f_out.write("%i %i\n" % (words, dim))
    f_en_ptr = open(f_en)
    f_en_ptr.readline()
    f_target_ptr = open(f_target)
    f_target_ptr.readline()
    copy_lines_lang(f_en_ptr, f_out, lang1)
    copy_lines_lang(f_target_ptr, f_out, lang2)
    print("merged vector file at: " + out_fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("w2v_1")
    parser.add_argument("w2v_2")
    parser.add_argument("output")
    parser.add_argument("lang1")
    parser.add_argument("lang2")
    parser.add_argument("dim", type=int)
    args = parser.parse_args()
    merge_vectors(args.w2v_1, args.w2v_2, args.output, args.lang1, args.lang2, args.dim)
