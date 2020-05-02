
#### Table of contents
1. [Introduction](#introduction)
2. [Experimental results](#exp)
3. [Using VnCoreNLP's word segmenter to pre-process input raw texts](#vncorenlp)
4. [Using PhoBERT in `fairseq`](#fairseq)
	- [Installation](#install1)
	- [Pre-trained models](#models1)
	- [Example usage](#usage1)
5. [Using PhoBERT in `transformers`](#transformers)
	- [Installation](#install2)
	- [Pre-trained models](#models2)
	- [Example usage](#usage2)




# PhoBERT: Pre-trained language models for Vietnamese <a name="introduction"></a>

Pre-trained PhoBERT models are the state-of-the-art language models for Vietnamese ([Pho](https://en.wikipedia.org/wiki/Pho), i.e. "Phở", is a popular food in Vietnam): 

 - Two PhoBERT versions of "base" and "large" are the first public large-scale monolingual language models pre-trained for Vietnamese. PhoBERT pre-training approach is based on [RoBERTa](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.md)  which optimizes the [BERT](https://github.com/google-research/bert) pre-training procedure for more robust performance.
 - PhoBERT outperforms previous monolingual and multilingual approaches, obtaining new state-of-the-art performances on four downstream Vietnamese NLP tasks of Part-of-speech tagging, Dependency parsing, Named-entity recognition and Natural language inference.

The general architecture and experimental results of PhoBERT can be found in our [paper](https://arxiv.org/abs/2003.00744):

    @article{phobert,
    title     = {{PhoBERT: Pre-trained language models for Vietnamese}},
    author    = {Dat Quoc Nguyen and Anh Tuan Nguyen},
    journal   = {arXiv preprint},
    volume    = {arXiv:2003.00744},
    year      = {2020}
    }

**Please CITE** our paper when PhoBERT is used to help produce published results or incorporated into other software.

## Main results <a name="exp"></a>

<img width="888" alt="posdep" src="https://user-images.githubusercontent.com/2412555/80791768-dc0a5a80-8bbc-11ea-964e-486280c8e616.png">
<img width="888" alt="nernli" src="https://user-images.githubusercontent.com/2412555/80791763-d6147980-8bbc-11ea-8f11-eb4a545ad0ae.png">

## Using VnCoreNLP's word segmenter to pre-process input raw texts <a name="vncorenlp"></a>

In case the input texts are `raw`, i.e. without word segmentation, a word segmenter must be applied to produce word-segmented texts before feeding to PhoBERT. As PhoBERT employed the [RDRSegmenter](https://github.com/datquocnguyen/RDRsegmenter) from [VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP) to pre-process the pre-training data, it is recommended to also use the same word segmenter for PhoBERT-based downstream applications w.r.t. the input raw texts.

### Installation

	# Install the vncorenlp python wrapper
	pip3 install vncorenlp
	
	# Download VnCoreNLP-1.1.1.jar & its word segmentation component (i.e. RDRSegmenter) 
	mkdir -p vncorenlp/models/wordsegmenter
	wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar
	wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab
	wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr
	mv VnCoreNLP-1.1.1.jar vncorenlp/ 
	mv vi-vocab vncorenlp/models/wordsegmenter/
	mv wordsegmenter.rdr vncorenlp/models/wordsegmenter/

`VnCoreNLP-1.1.1.jar` (27MB) and folder `models` must be placed in the same working folder, here is `vncorenlp`!

### Example usage

```python
# See more details at: https://github.com/vncorenlp/VnCoreNLP

# Load rdrsegmenter from VnCoreNLP
from vncorenlp import VnCoreNLP
rdrsegmenter = VnCoreNLP("/Absolute-path-to/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 

# Input 
text = "Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."

# To perform word segmentation only
word_segmented_text = rdrsegmenter.tokenize(text) 
print(word_segmented_text)
```

```
[['Ông', 'Nguyễn_Khắc_Chúc', 'đang', 'làm_việc', 'tại', 'Đại_học', 'Quốc_gia', 'Hà_Nội', '.'], ['Bà', 'Lan', ',', 'vợ', 'ông', 'Chúc', ',', 'cũng', 'làm_việc', 'tại', 'đây', '.']]
```


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


#####  Using VnCoreNLP's RDRSegmenter with PhoBERT in `fairseq`

```python
text = "Tôi là sinh viên trường đại học Công nghệ."
sentences = rdrsegmenter.tokenize(text) 
# Extract the last layer's features  
for sentence in sentences:
	subwords = phobert.encode(" ".join(sentence))
	last_layer_features = phobert.extract_features(subwords)  

```


## Using PhoBERT in HuggingFace's [`transformers`](https://github.com/huggingface/transformers) <a name="transformers"></a>

### Installation <a name="install2"></a>
- Prerequisites: [Installation w.r.t. `fairseq`](#install1), and [`VnCoreNLP-RDRSegmenter` if processing raw texts](#vncorenlp)
-  [`transformers`](https://github.com/huggingface/transformers): `pip3 install transformers`

### Pre-trained models <a name="models2"></a>

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

### Example usage <a name="usage2"></a>

```python
import torch
import argparse

from transformers import RobertaConfig
from transformers import RobertaModel

from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary

# Load model
config = RobertaConfig.from_pretrained(
    "/Absolute-path-to/PhoBERT_base_transformers/config.json"
)
phobert = RobertaModel.from_pretrained(
    "/Absolute-path-to/PhoBERT_base_transformers/model.bin",
    config=config
)

# Load BPE encoder 
parser = argparse.ArgumentParser()
parser.add_argument('--bpe-codes', 
    default="/Absolute-path-to/PhoBERT_base_transformers/bpe.codes",
    required=False,
    type=str,  
    help='path to fastBPE BPE'
)
args = parser.parse_args()
bpe = fastBPE(args)

# Load the dictionary  
vocab = Dictionary()
vocab.add_from_file("/Absolute-path-to/PhoBERT_base_transformers/dict.txt")

# INPUT TEXT IS WORD-SEGMENTED!
line = "Tôi là sinh_viên trường đại_học Công_nghệ ."  

# Encode the line using fastBPE & Add prefix <s> and suffix </s> 
subwords = '<s> ' + bpe.encode(line) + ' </s>'

# Map subword tokens to corresponding indices in the dictionary
input_ids = vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()

# Convert into torch tensor
all_input_ids = torch.tensor([input_ids], dtype=torch.long)

# Extract features  
with torch.no_grad():  
    features = phobert(all_input_ids)  
  
# Represent each word by the contextualized embedding of its first subword token  
# i. Get indices of the first subword tokens of words in the input sentence 
listSWs = subwords.split()  
firstSWindices = []  
for ind in range(1, len(listSWs) - 1):  
    if not listSWs[ind - 1].endswith("@@"):  
        firstSWindices.append(ind)  

# ii. Extract the corresponding contextualized embeddings  
words = line.split()  
assert len(firstSWindices) == len(words)  
vectorSize = features[0][0, 0, :].size()[0]  
for word, index in zip(words, firstSWindices):  
    print(word + " --> " + " ".join([str(features[0][0, index, :][_ind].item()) for _ind in range(vectorSize)]))
    # print(word + " --> " + listSWs[index] + " --> " + " ".join([str(features[0][0, index, :][_ind].item()) for _ind in range(vectorSize)]))

```


## License
    PhoBERT  Copyright (C) 2020  VinAI Research

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
