# -*- coding: utf-8 -*-
from utils import *
from vocabulary import *
import sys
from io import open
import pandas as pd
from tqdm import tqdm
import argparse

# sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)


# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    try:
        return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH
    except AttributeError:
        return False


# Filter pairs using filterPair condition
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


# Read query/response pairs and return a voc object
def readVocs(data, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines

    pairs = []

    for row in range(len(data)):
        pairs.append([data.iloc[row]['context'],
                      data.iloc[row]['answer']])
        if row % 10000 == 0:
            print(row)
    voc = Voc(corpus_name)

    return voc, pairs


def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)

    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-csv_path', default='./data/pikabu/pikabu.csv')
    parser.add_argument('-nrows', type=int, default=1000)
    parser.add_argument('-res_path', default='./data/pikabu/pikabu_res.p')
    parser.add_argument('-voc_path', default='./data/pikabu/voc.txt')
    args = parser.parse_args()

    data = pd.read_csv(args.csv_path, nrows=args.nrows)
    tqdm.pandas(desc='')
    data['context'] = data['context'].progress_apply(normalizeString)
    data['answer'] = data['answer'].progress_apply(normalizeString)

    voc, pairs = readVocs(data, 'pikabu')
    pairs = filterPairs(pairs)

    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])

    pairs = trimRareWords(voc, pairs, 2)

    # for i in range(min(len(pairs), 50)):
    #     print(i, pairs[i])
    voc.save(args.voc_path)

    # TODO remove pickle
    import pickle
    pickle.dump(pairs, open(args.res_path, 'wb'))


if __name__ == '__main__':
    sys.exit(main())

