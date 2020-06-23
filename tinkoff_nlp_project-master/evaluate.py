# -*- coding: utf-8 -*-
import sys
from utils import *
from models import *
from vocabulary import *
import argparse


sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ') if word if word] + [EOS_token]


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    # Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        # try:
        # Get input sentence
        input_sentence = input('> ')
        # Check if it is quit case
        if input_sentence == 'q' or input_sentence == 'quit': break
        # Normalize sentence

        input_sentence = normalizeString(input_sentence)
        # Evaluate sentence
        output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
        # Format and print response sentence
        output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
        #sys.stdout.write('Bot: {}'.format())
        print('Bot: ', ' '.join(output_words))

        try:
            pass
        except KeyError:
            print("Error: Encountered unknown word.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', default='4000_checkpoint.tar')
    args = parser.parse_args()

    checkpoint = torch.load(args.model)

    voc = Voc('pikabu')
    voc.__dict__ = checkpoint['voc_dict']


    # print(checkpoint['voc_dict'])
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)
    embedding.load_state_dict(checkpoint['embedding'])

    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])

    encoder.eval()
    decoder.eval()
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    searcher = GreedySearchDecoder(encoder, decoder)

    evaluateInput(encoder, decoder, searcher, voc)


if __name__ == '__main__':
    sys.exit(main())
