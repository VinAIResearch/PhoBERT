from fairseq.data import Dictionary
from fairseq.data.encoders.fastbpe import fastBPE
from vncorenlp import VnCoreNLP

import argparse
import numpy as np
import torch


VNCORENLP_ADDRESS = "http://127.0.0.1"
VNCORENLP_PORT = 9000
BPE_PATH = "PhoBERT_large_fairseq/bpe.codes"
DICT_PATH = "PhoBERT_large_fairseq/dict.txt"


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--bpe-codes', type=str, help='path to fastBPE BPE', default=BPE_PATH)
args = parser.parse_args("")


class PhoBertTokenizer:
    def __init__(self, vncore=True):
        """
        Hacky way to run VnCoreNLP tokenizer with PhoBERT
        :param vncore: Set it to `False` if your sentences are already tokenized by VnCoreNLP
        """
        self.dictionary = Dictionary.load(open(DICT_PATH))
        self.annotator = None
        self.vncore = vncore
        self.bpe = fastBPE(args)
        
    def convert_tokens_to_ids(self, text_spans_bpe):
        return self.dictionary.encode_line(
            '<s> ' + text_spans_bpe + ' </s>',
            append_eos=False,
            add_if_not_exist=False)
    
    def tokenize(self, raw_sentence: str):
        if self.vncore:
            if self.annotator is None:
                self.annotator = VnCoreNLP(VNCORENLP_ADDRESS, port=VNCORENLP_PORT)
            word_tokenizes = ' '.join(sum(self.annotator.tokenize(raw_sentence), []))
        else:
            word_tokenizes = raw_sentence
        return self.bpe.encode(word_tokenizes)
    
    def encode(self, raw_sentence: str):
        return self.convert_tokens_to_ids(self.tokenize(raw_sentence)).long()
    
    def decode(self, tokens: torch.LongTensor, remove_underscore=True):
        assert tokens.dim() == 1
        tokens = tokens.numpy()
        if tokens[0] == self.dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = (tokens == self.dictionary.eos())
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        if remove_underscore:
            sentences = [self.bpe.decode(self.dictionary.string(s)).replace("_", " ") for s in sentences]
        else:
            sentences = [self.bpe.decode(self.dictionary.string(s)) for s in sentences]
            
        if len(sentences) == 1:
            return sentences[0]
        return sentences
