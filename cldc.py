"""Evaluate CLWE on cross-lingual document classification (CLDC)"""

from argparse import ArgumentParser

import logging
import random

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

import jieba

RCV2_TAGS = ['CCAT', 'ECAT', 'MCAT', 'GCAT']
OOV = '<OOV>'


class CNN(nn.Module):
    """Convolutional neural networks text classifier (Kim, 2014).
    We assume that input are word embeddings.
    """

    def __init__(self, embedding_dim, n_classes,
                 n_filters=100, filter_sizes=(3, 4, 5), dropout=.5):
        super(CNN, self).__init__()
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
                                              out_channels=n_filters,
                                              kernel_size=(fs, embedding_dim))
                                    for fs in filter_sizes])
        self.classifier = nn.Linear(len(filter_sizes) * n_filters, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedded):
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.classifier(cat)


def load_embeddings(emb_file, max_vocab=-1):
    logging.info('Load embeddings from ' + emb_file)
    embeds = {}
    with open(emb_file, 'r') as f:
        for line in f:
            fields = line.split()
            if len(fields) == 2:  # skip header
                embed_dim = int(fields[1])
                continue
            word = fields[0].lower()
            if word in embeds:
                logging.warning('%s is found more than once' % word)
            else:
                embeds[word] = [float(x) for x in fields[1:]]
            if len(embeds) == max_vocab:
                break
    assert OOV not in embeds, 'OOV symbol is already in the vocabulary.'
    logging.info('Find %d word vectors' % len(embeds))
    embeds[OOV] = [0 for _ in range(embed_dim)]
    return embeds


def load_data(data_file, embeds, lang, labels):
    logging.info('Load data from ' + data_file)
    data = []
    with open(data_file, 'r') as f:
        for line in f:
            label, doc = line.split(maxsplit=1)
            doc = eval(doc).decode('utf-8').lower()
            if lang == 'zh':
                tokens = jieba.cut(doc)
            else:
                tokens = doc.split()
            x = []
            for w in tokens:
                if w in embeds:
                    x.append(embeds[w])
            while len(x) < 5:  # CNN minimum length requirement
                x.append(embeds[OOV])
            x = torch.Tensor([x])
            y = torch.LongTensor([labels.index(label)])
            data.append((x, y))
    logging.info('Find %d documents' % len(data))
    return data


def train(model, data, optimizer, loss_fn, device):
    model.train()
    tot_loss, correct = .0, .0
    random.shuffle(data)
    for x, y in data:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        tot_loss += loss.data.item()
        loss.backward()
        optimizer.step()
        pred = torch.argmax(out.data)
        if pred == y:
            correct += 1
    return tot_loss / len(data), correct / len(data)


def evaluate(model, data, loss_fn, device):
    model.eval()
    tot_loss, correct = .0, .0
    for x, y in data:
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        tot_loss += loss.data.item()
        pred = torch.argmax(out.data)
        if pred == y:
            correct += 1
    return tot_loss / len(data), correct / len(data)


def main():
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument('--src', required=True, help='source embeddings')
    parser.add_argument('--tgt', required=True, help='target embeddings')
    parser.add_argument('--train', required=True, help='train set')
    parser.add_argument('--dev', required=True, help='dev set')
    parser.add_argument('--test', required=True, help='test set')
    parser.add_argument('--dw', default=300, help='embed dimensions')
    parser.add_argument('--max_vocab', default=200000, help='vocab size')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--runs', type=int, default=10,
                        help='number of restarts')
    parser.add_argument('--labels', default=','.join(RCV2_TAGS),
                        help='label set (separated by comma)')
    parser.add_argument('--train_lang', default='en', help='train language')
    parser.add_argument('--test_lang', required=True, help='test language')
    parser.add_argument('--epoch', type=int, default=10)
    args = parser.parse_args()
    logging.info(vars(args))

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    src_emb = load_embeddings(args.src, max_vocab=args.max_vocab)
    tgt_emb = load_embeddings(args.tgt, max_vocab=args.max_vocab)

    labels = args.labels.split(',')
    train_set = load_data(args.train, src_emb, lang=args.train_lang, labels=labels)
    dev_set = load_data(args.dev, src_emb, lang=args.train_lang, labels=labels)
    test_set = load_data(args.test, tgt_emb, lang=args.test_lang, labels=labels)

    test_accs = []
    for n_run in range(args.runs):
        logging.info('Run %d' % n_run)
        model = CNN(args.dw, len(labels)).to(device)
        loss_fn = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters())
        best_dev_acc, final_test_acc = 0.0, None
        for n_epoch in range(args.epoch):
            train_loss, train_acc = train(model, train_set, optimizer, loss_fn,
                                          device)
            dev_loss, dev_acc = evaluate(model, dev_set, loss_fn, device)
            test_loss, test_acc = evaluate(model, test_set, loss_fn, device)
            logging.info('Epoch {} | Train loss: {:.3f} | Train acc: {:.4f} | '
                         'Dev loss: {:.3f} | Dev acc: {:.4f} | Test acc: {:.4f}'
                         .format(n_epoch, train_loss, train_acc, dev_loss,
                                 dev_acc, test_acc))
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                final_test_acc = test_acc
        logging.info('Test acc of best model: {:4f}'.format(final_test_acc))
        test_accs.append(final_test_acc)
    logging.info('Average test accuracy: {:4f}'.format(np.mean(test_accs)))
    logging.info('Standard deviation: {:4f}'.format(np.std(test_accs)))


if __name__ == '__main__':
    main()
