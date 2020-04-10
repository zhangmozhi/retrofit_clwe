# Retrofitting Cross-Lingual Word Embeddings (CLWE) to Dictionaries

This repository contains code to reproduce results from our ACL 2020 paper:

Mozhi Zhang, Yoshinari Fujinuma, Michael J. Paul, Jordan Boyd-Graber. *Why Overfitting Isn't Always Bad: Retrofitting Cross-Lingual Word Embeddings to Dictionaries*.

If you find this repository helpful, please cite:

    @inproceedings{zhang-2020-overfit,
    	Title = {Why Overfitting Isn't Always Bad: Retrofitting Cross-Lingual Word Embeddings to Dictionaries},
    	Author = {Zhang, Mozhi and Fujinuma, Yoshinari and Paul, Michael J. and  Boyd-Graber, Jordan},
    	Booktitle = {Proceedings of the Association for Computational Linguistics},
    	Year = {2020}
    }


### Train and Preprocess CLWE

The first step is to train CLWE using a projection-based method.
In the paper, we start with monolingual [Wikipedia FastText embeddings](https://fasttext.cc/docs/en/pretrained-vectors.html).
We normalize the monolingual embeddings with [Iterative Normalization](https://github.com/zhangmozhi/iternorm) and align them with [CCA](https://github.com/mfaruqui/crosslingual-cca), [MUSE](https://github.com/facebookresearch/MUSE), or [RCSLS](https://github.com/facebookresearch/fastText/tree/master/alignment).
We preprocess them to lowercase all words and keep only the top 200K words:

    python preprocess_embeds [INPUT_EMBEDDING_FILE] [OUTPUT_EMBEDDING_FILE]
    
### Retrofit CLWE to a Dictionary

Once we have pre-trained CLWE, we can retrofit to a dictionary.
First, we need to merge the trained CLWE into a single file with `merge_clwe.py`.
For example, if we have aligned 300 dimensional English embeddings at `embed/vectors-en.txt` and Chinese embeddings at `embed/vectors-zh.txt`, we merge them with:

    python merge_clwe.py \
        embed/vectors-en.txt \
        embed/vectors-zh.txt \
        embed/vectors-en-zh.txt \
        en zh 300
        
We then retrofit it to a dictionary.  Here we use the [English-Chinese training dictionary](https://dl.fbaipublicfiles.com/arrival/dictionaries/en-zh.0-5000.txt) from MUSE:
    
    python retrofit.py \
        -i embed/vectors-en-zh.txt \
        -l dictionaries/en-zh.0-5000.txt \
        -o embed_retrofit \
        --src_lang en \
        --tgt_lang zh
        
The retrofitted vectors are at `embed_retrofit/vectors-en.txt` and `embed_retrofit/vectors-zh.txt` (specified by `-o` flag).

### Generate Synthetic Dictionary from CLWE

We can use `word_translation.py` and `merge_translation.py` to build a synthetic dictionary from CLWE.
For example, if we want to build an English-Chinese synthetic dictionary, we first
translate every English word to Chinese:
    
    python word_translation.py \
        --src embed/vectors-en.txt \
        --tgt embed/vectors-zh.txt \
        --output en-zh.txt

We then translate every Chinese word to English:

    python word_translation.py \
        --src embed/vectors-zh.txt \
        --tgt embed/vectors-en.txt \
        --output zh-en.txt
        
Finally, we build a high-confidence synthetic dictionary using only mutual translations:

    python merge_translation.py en-zh.txt zh-en.txt en-zh-merged.txt

We can combine the final synthetic dictionary `en-zh-merged.txt` with the training dictionary.
Retrofitting to the combined dictionary often improves downstream task accuracy.

### Evaluate on Document Classification

Download [RCV2](https://trec.nist.gov/data/reuters/reuters.html) and split the data with [MLDoc](https://github.com/facebookresearch/MLDoc).
The script `cldc.py` trains a CNN with CLWE features and reports test accuracy.
For example, we can run a English-Chinese experiment:
    
    python cldc.py \
        --src embed/vectors-en.txt \
        --tgt embed/vectors-zh.txt \
        --train_lang en \
        --test_lang zh \
        --train mldoc/english.train.5000 \
        --dev mldoc/english.dev \
        --test mldoc/chinese.test

### Evaluate on Dependency Parsing

Coming soon.