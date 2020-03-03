# PhoBERT: Pre-trained language models for Vietnamese

Pre-trained PhoBERT models are the state-of-the-art language models for Vietnamese ([Pho](https://en.wikipedia.org/wiki/Pho), i.e. "Phở", is a popular food in Vietnam): 

 - Two versions of PhoBERT "base" and "large" are the first public large-scale monolingual language models pre-trained for Vietnamese. PhoBERT pre-training approach is based on [RoBERTa](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.md)  which optimizes the [BERT](https://github.com/google-research/bert) pre-training method for more robust performance.
 - PhoBERT outperforms previous monolingual and multilingual approaches, obtaining new state-of-the-art performances on three downstream Vietnamese NLP tasks of Part-of-speech tagging, Named-entity recognition and Natural language inference.

The general architecture and experimental results of PhoBERT can be found in our [following paper](https://arxiv.org/abs/2003.00744):

    @article{phobert,
    title     = {{PhoBERT: Pre-trained language models for Vietnamese}},
    author    = {Dat Quoc Nguyen and Anh Tuan Nguyen},
    journal   = {arXiv preprint},
    volume    = {arXiv:2003.00744},
    year      = {2020}
    }

**Please cite** our paper when PhoBERT is used to help produce published results or incorporated into other software.

## Experimental results

<img width="900" alt="PhoBERT results: POS Tagging, NER, NLI" src="https://user-images.githubusercontent.com/2412555/75759331-f0baa580-5d67-11ea-943a-8163cf716e7e.png">

Experiments show that using a straightforward finetuning manner as we use for PhoBERT can lead to state-of-the-art results. Note that we might boost our downstream task performances even further by doing a more careful hyper-parameter fine-tuning.

## Using PhoBERT in [`fairseq`](https://github.com/pytorch/fairseq)

### Installation

 -  Python version >= 3.6
 - [PyTorch](http://pytorch.org/) version >= 1.2.0
 - [`fairseq`](https://github.com/pytorch/fairseq)
 - `fastBPE`: `pip3 install fastBPE`

### Pre-trained models

Model | #params | size | Download
---|---|---|---
`PhoBERT-base` | 135M | 1.2GB | [PhoBERT_base_fairseq.tar.gz](https://public.vinai.io/PhoBERT_base_fairseq.tar.gz)
`PhoBERT-large` | 370M | 3.2GB | [PhoBERT_large_fairseq.tar.gz](https://public.vinai.io/PhoBERT_large_fairseq.tar.gz)

`PhoBERT-base`:
 - `wget https://public.vinai.io/PhoBERT_base_fairseq.tar.gz`
 - `tar -xzvf PhoBERT_base_fairseq.tar.gz`

`PhoBERT-large`:
 - `wget https://public.vinai.io/PhoBERT_large_fairseq.tar.gz`
 - `tar -xzvf PhoBERT_large_fairseq.tar.gz`

### Example usage

```Python

# Load PhoBERT-base in fairseq
from fairseq.models.roberta import RobertaModel
phobert = RobertaModel.from_pretrained('/path/to/PhoBERT_base_fairseq', checkpoint_file='model.pt')
phobert.eval()  # disable dropout (or leave in train mode to finetune)

# Incorporate the BPE encoder into PhoBERT-base 
from fairseq.data.encoders.fastbpe import fastBPE  
from fairseq import options  
parser = options.get_preprocessing_parser()  
parser.add_argument('--bpe-codes', type=str, help='path to fastBPE BPE', default="/path/to/PhoBERT_base_fairseq/bpe.codes")  
args = parser.parse_args()  
phobert.bpe = fastBPE(args) #Incorporate the BPE encoder into PhoBERT


# Extract the last layer's features  
line = "Tôi là sinh_viên trường đại_học Công_nghệ"  # INPUT TEXT IS WORD-SEGMENTED!
subwords = phobert.encode(line)  
last_layer_features = phobert.extract_features(subwords)  
assert last_layer_features.size() == torch.Size([1, 8, 768])  
  
# Extract all layer's features (layer 0 is the embedding layer)  
all_layers = phobert.extract_features(subwords, return_all_hiddens=True)  
assert len(all_layers) == 13  
assert torch.all(all_layers[-1] == last_layer_features)  
  
# Extract features aligned to words  
words = phobert.extract_features_aligned_to_words(line)  
for word in words:  
    print('{:10}{} (...)'.format(str(word), word.vector[:5]))  
  
# Filling marks  
masked_line = 'Tôi là  <mask> trường đại_học Công_nghệ '  
topk_filled_outputs = phobert.fill_mask(masked_line, topk=5)  
print(topk_filled_outputs)

```

## Using PhoBERT in Hugging Face [`transformers`](https://github.com/huggingface/transformers)

### Installation

 -  [`transformers`](https://github.com/huggingface/transformers): `pip3 install transformers`

### Pre-trained models

Model | #params | size | Download
---|---|---|---
`PhoBERT-base` | 135M | 307MB | [PhoBERT_base_transformers.tar.gz](https://public.vinai.io/PhoBERT_base_transformers.tar.gz)
`PhoBERT-large` | 370M | 834MB | [PhoBERT_large_transformers.tar.gz](https://public.vinai.io/PhoBERT_large_transformers.tar.gz)

`PhoBERT-base`:
 - `wget https://public.vinai.io/PhoBERT_base_transformers.tar.gz`
 - `tar -xzvf PhoBERT_base_transformers.tar.gz`

`PhoBERT-large`:
 - `wget https://public.vinai.io/PhoBERT_large_transformers.tar.gz`
 - `tar -xzvf PhoBERT_large_transformers.tar.gz`

### Example usage

Under construction, coming soon!
