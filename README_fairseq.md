# PhoBERT: Pre-trained language models for Vietnamese <a name="introduction"></a>

Pre-trained PhoBERT models are the state-of-the-art language models for Vietnamese ([Pho](https://en.wikipedia.org/wiki/Pho), i.e. "Phở", is a popular food in Vietnam): 

 - Two PhoBERT versions of "base" and "large" are the first public large-scale monolingual language models pre-trained for Vietnamese. PhoBERT pre-training approach is based on [RoBERTa](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.md)  which optimizes the [BERT](https://github.com/google-research/bert) pre-training procedure for more robust performance.
 - PhoBERT outperforms previous monolingual and multilingual approaches, obtaining new state-of-the-art performances on four downstream Vietnamese NLP tasks of Part-of-speech tagging, Dependency parsing, Named-entity recognition and Natural language inference.

The general architecture and experimental results of PhoBERT can be found in our [paper](https://www.aclweb.org/anthology/2020.findings-emnlp.92/):

    @inproceedings{phobert,
    title     = {{PhoBERT: Pre-trained language models for Vietnamese}},
    author    = {Dat Quoc Nguyen and Anh Tuan Nguyen},
    booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
    year      = {2020},
    pages     = {1037--1042}
    }

**Please CITE** our paper when PhoBERT is used to help produce published results or is incorporated into other software.

## Using PhoBERT in [`fairseq`](https://github.com/pytorch/fairseq) <a name="fairseq"></a>

### Installation <a name="install1"></a>

 -  Python version >= 3.6
 - [PyTorch](http://pytorch.org/) version >= 1.4.0
 - [`fairseq`](https://github.com/pytorch/fairseq)
 - `fastBPE`: `pip3 install fastBPE`

### Pre-trained models <a name="models1"></a>

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

### Example usage <a name="usage1"></a>

_Assume that the input texts are already word-segmented!_

```python
import torch

# Load PhoBERT-base in fairseq
from fairseq.models.roberta import RobertaModel
phobert = RobertaModel.from_pretrained('/Absolute-path-to/PhoBERT_base_fairseq', checkpoint_file='model.pt')
phobert.eval()  # disable dropout (or leave in train mode to finetune)

# Incorporate the BPE encoder into PhoBERT-base 
from fairseq.data.encoders.fastbpe import fastBPE  
from fairseq import options  
parser = options.get_preprocessing_parser()  
parser.add_argument('--bpe-codes', type=str, help='path to fastBPE BPE', default="/Absolute-path-to/PhoBERT_base_fairseq/bpe.codes")  
args = parser.parse_args()  
phobert.bpe = fastBPE(args) #Incorporate the BPE encoder into PhoBERT

# INPUT TEXT IS WORD-SEGMENTED!
line = "Tôi là sinh_viên trường đại_học Công_nghệ ."  

# Extract the last layer's features  
subwords = phobert.encode(line)  
last_layer_features = phobert.extract_features(subwords)  
assert last_layer_features.size() == torch.Size([1, 9, 768])  
  
# Extract all layer's features (layer 0 is the embedding layer)  
all_layers = phobert.extract_features(subwords, return_all_hiddens=True)  
assert len(all_layers) == 13  
assert torch.all(all_layers[-1] == last_layer_features)  

# Filling marks  
masked_line = 'Tôi là  <mask> trường đại_học Công_nghệ .'  
topk_filled_outputs = phobert.fill_mask(masked_line, topk=5)  
print(topk_filled_outputs)

```


## License
    
	MIT License

	Copyright (c) 2020 VinAI Research

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all
	copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
	SOFTWARE.
