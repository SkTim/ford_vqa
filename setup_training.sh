#!/bin/bash

# Download pretrained embeddings.
curl -O https://lil.cs.washington.edu/coref/turian.50d.txt
curl -O https://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip

python get_char_vocab.py
python filter_embeddings.py glove.840B.300d.txt train.jsonlines dev.jsonlines test.jsonlines
