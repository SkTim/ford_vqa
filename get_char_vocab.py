#!/usr/bin/env python

import sys
import json

def get_char_vocab(input_filenames, output_filename):
  vocab = set()
  for filename in input_filenames:
    with open(filename) as f:
      for line in f.readlines():
        for word in json.loads(line)["question"].split(' '):
          vocab.update(word)
        for word in json.loads(line)["script"].split(' '):
          vocab.update(word)
  vocab = sorted(list(vocab))
  with open(output_filename, "w") as f:
    for char in vocab:
      f.write(u"{}\n".format(char).encode("utf8"))
  print("Wrote {} characters to {}".format(len(vocab), output_filename))

def get_char_vocab_language(language):
  get_char_vocab(["{}.jsonlines".format(partition) for partition in ("train", "test")], "char_vocab.{}.txt".format(language))

get_char_vocab_language("english")
