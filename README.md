Bilingual Sentiment Embeddings: Joint Projection of Sentiment Across Languages
==============

This is forked from the source code from the ACL paper:

Jeremy Barnes, Roman Klinger, and Sabine Schulde im Walde. 2018. [**Bilingual Sentiment Embeddings: Joint Projection of Sentiment Across Languages**](http://aclweb.org/anthology/P18-1231). In *Proceedings of ACL 2018*.


If you use the code for academic research, please cite the paper in question:
```
@inproceedings{Barnes2018blse,
  author = 	"Barnes, Jeremy
		and Klinger, Roman
		and Schulte im Walde, Sabine",
  title = 	"Bilingual Sentiment Embeddings: Joint Projection of Sentiment Across Languages",
  booktitle = 	"Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"2483--2493",
  location = 	"Melbourne, Australia",
  url = 	"http://aclweb.org/anthology/P18-1231"
}


```


Requirements to run the experiments
--------
- Python 3
- NumPy
- sklearn [http://scikit-learn.org/stable/]
- pytorch [http://pytorch.org/]



Usage
--------

First, clone the repo:

```
git clone https://github.com/UriSha/blse.git
```


Then, get monolingual embeddings, either by training your own,
or by downloading the [pretrained embeddings](https://drive.google.com/open?id=1GpyF2h0j8K5TKT7y7Aj0OyPgpFc8pMNS) mentioned in the paper,
unzipping them and putting them in the 'embeddings' directory

Finally, run main.y with the desired parameters. For example

```
python3 main.py --model rnn_attn_blse --binary False --target_lang ca --alpha 0.01
```


``` 

License
-------

Copyright (C) 2018, Jeremy Barnes

Licensed under the terms of the Creative Commons CC-BY public license
