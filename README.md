
#### Table of contents
1. [Introduction](#introduction)
2. [Using PhoBERT with `transformers`](#transformers)
	- [Installation](#install2)
	- [Pre-trained models](#models2)
	- [Example usage](#usage2)
3. [Using PhoBERT with `fairseq`](#fairseq)
4. [Notes](#vncorenlp)

# <a name="introduction"></a> PhoBERT: Pre-trained language models for Vietnamese 

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

## <a name="transformers"></a> Using PhoBERT with `transformers` 

### Installation <a name="install2"></a>
- Install `transformers` with pip: `pip install transformers`, or [install `transformers` from source](https://huggingface.co/docs/transformers/installation#installing-from-source).  <br /> 
Note that we merged a slow tokenizer for PhoBERT into the main `transformers` branch. The process of merging a fast tokenizer for PhoBERT is in the discussion, as mentioned in [this pull request](https://github.com/huggingface/transformers/pull/17254#issuecomment-1133932067). If users would like to utilize the fast tokenizer, the users might install `transformers` as follows:


```
git clone --single-branch --branch fast_tokenizers_BARTpho_PhoBERT_BERTweet https://github.com/datquocnguyen/transformers.git
cd transformers
pip3 install -e .
```

- Install `tokenizers` with pip: `pip3 install tokenizers`

### Pre-trained models <a name="models2"></a>


Model | #params | Arch.	 | Max length | Pre-training data | License
---|---|---|---|---|---
[`vinai/phobert-base-v2`](https://huggingface.co/vinai/phobert-base-v2) | 135M | base | 256 | 20GB  of Wikipedia and News texts + 120GB of texts from OSCAR-2301 | [GNU Affero GPL v3](https://github.com/VinAIResearch/PhoBERT/blob/master/LICENSE_for_PhoBERT_v2)
[`vinai/phobert-base`](https://huggingface.co/vinai/phobert-base) | 135M | base | 256 | 20GB  of Wikipedia and News texts | [MIT License](https://github.com/VinAIResearch/PhoBERT/blob/master/LICENSE)
[`vinai/phobert-large`](https://huggingface.co/vinai/phobert-large) | 370M | large | 256 | 20GB  of Wikipedia and News texts | [MIT License](https://github.com/VinAIResearch/PhoBERT/blob/master/LICENSE)


### Example usage <a name="usage2"></a>

```python
import torch
from transformers import AutoModel, AutoTokenizer

phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

# INPUT TEXT MUST BE ALREADY WORD-SEGMENTED!
sentence = 'Chúng_tôi là những nghiên_cứu_viên .'  

input_ids = torch.tensor([tokenizer.encode(sentence)])

with torch.no_grad():
    features = phobert(input_ids)  # Models outputs are now tuples

## With TensorFlow 2.0+:
# from transformers import TFAutoModel
# phobert = TFAutoModel.from_pretrained("vinai/phobert-base")
```


## <a name="fairseq"></a> Using PhoBERT with `fairseq`

Please see details at [HERE](https://github.com/VinAIResearch/PhoBERT/blob/master/README_fairseq.md)!

## <a name="vncorenlp"></a> Notes 

In case the input texts are `raw`, i.e. without word segmentation, a word segmenter must be applied to produce word-segmented texts before feeding to PhoBERT. As PhoBERT employed the [RDRSegmenter](https://github.com/datquocnguyen/RDRsegmenter) from [VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP) to pre-process the pre-training data (including [Vietnamese tone normalization](https://github.com/VinAIResearch/BARTpho/blob/main/VietnameseToneNormalization.md) and word and sentence segmentation), it is recommended to also use the same word segmenter for PhoBERT-based downstream applications w.r.t. the input raw texts.

#### Installation

    pip install py_vncorenlp

#### Example usage <a name="example"></a>

```python
import py_vncorenlp

# Automatically download VnCoreNLP components from the original repository
# and save them in some local machine folder
py_vncorenlp.download_model(save_dir='/absolute/path/to/vncorenlp')

# Load the word and sentence segmentation component
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='/absolute/path/to/vncorenlp')

text = "Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."

output = rdrsegmenter.word_segment(text)

print(output)
# ['Ông Nguyễn_Khắc_Chúc đang làm_việc tại Đại_học Quốc_gia Hà_Nội .', 'Bà Lan , vợ ông Chúc , cũng làm_việc tại đây .']
```
