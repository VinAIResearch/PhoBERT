
#### Table of contents
1. [Introduction](#introduction)
2. [Using PhoBERT with `transformers`](#transformers)
	- [Installation](#install2)
	- [Pre-trained models](#models2)
	- [Example usage](#usage2)
3. [Using PhoBERT with `fairseq`](#fairseq)
4. [Using VnCoreNLP's word segmenter to pre-process input raw texts](#vncorenlp)





# <a name="introduction"></a> PhoBERT: Pre-trained language models for Vietnamese 

Pre-trained PhoBERT models are the state-of-the-art language models for Vietnamese ([Pho](https://en.wikipedia.org/wiki/Pho), i.e. "Phở", is a popular food in Vietnam): 

 - Two PhoBERT versions of "base" and "large" are the first public large-scale monolingual language models pre-trained for Vietnamese. PhoBERT pre-training approach is based on [RoBERTa](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.md)  which optimizes the [BERT](https://github.com/google-research/bert) pre-training procedure for more robust performance.
 - PhoBERT outperforms previous monolingual and multilingual approaches, obtaining new state-of-the-art performances on four downstream Vietnamese NLP tasks of Part-of-speech tagging, Dependency parsing, Named-entity recognition and Natural language inference.

The general architecture and experimental results of PhoBERT can be found in our EMNLP-2020 Findings [paper](https://arxiv.org/abs/2003.00744):

    @article{phobert,
    title     = {{PhoBERT: Pre-trained language models for Vietnamese}},
    author    = {Dat Quoc Nguyen and Anh Tuan Nguyen},
    journal   = {Findings of EMNLP},
    year      = {2020}
    }

**Please CITE** our paper when PhoBERT is used to help produce published results or is incorporated into other software.

## <a name="transformers"></a> Using PhoBERT with `transformers` 

### Installation <a name="install2"></a>
 -  Python version >= 3.6
 - [PyTorch](http://pytorch.org/) version >= 1.4.0
-  Install `transformers` from our development branch:
	- `git clone https://github.com/datquocnguyen/transformers.git`
	- `cd transformers`
	- `pip install --upgrade .`

We also created a pull request to integrate PhoBERT into the master branch of the `transformers` library. Please see the latest updates at:  https://github.com/huggingface/transformers/pull/6129

### Pre-trained models <a name="models2"></a>


Model | #params | Arch.	 | Pre-training data
---|---|---|---
`vinai/phobert-base` | 135M | base | 20GB  of texts
`vinai/phobert-large` | 370M | large | 20GB  of texts

### Example usage <a name="usage2"></a>

```python
import torch
from transformers import AutoModel, AutoTokenizer #, PhobertTokenizer

phobert = AutoModel.from_pretrained("vinai/phobert-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
#tokenizer = PhobertTokenizer.from_pretrained("vinai/phobert-base")

# INPUT TEXT MUST BE ALREADY WORD-SEGMENTED!
line = "Tôi là sinh_viên trường đại_học Công_nghệ ."

input_ids = torch.tensor([tokenizer.encode(line)])

with torch.no_grad():
    features = phobert(input_ids)  # Models outputs are now tuples
```


## <a name="fairseq"></a> Using PhoBERT with `fairseq`

Please see details at [HERE](https://github.com/VinAIResearch/PhoBERT/blob/master/README_fairseq_and_old_transformers_version.md)!

## <a name="vncorenlp"></a> Using VnCoreNLP's word segmenter to pre-process input raw texts 

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

# To perform word (and sentence) segmentation
sentences = rdrsegmenter.tokenize(text) 
for sentence in sentences:
	print(" ".join(sentence))
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
