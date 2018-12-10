#!/usr/bin/env python

import sys
import json

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("Usage: {} <embeddings> <json1> <json2> ...".format(sys.argv[0]))

    words_to_keep = set()
    for json_filename in sys.argv[2:]:
        with open(json_filename) as json_file:
            for line in json_file.readlines():
                example = json.loads(line)
                # for sentence in json.loads(line)["sentences"]:
                words_to_keep.update(example['question'].split(' '))
                words_to_keep.update(example['script'].split(' '))

    print "Found {} words in {} dataset(s).".format(len(words_to_keep), len(sys.argv) - 2)

    total_lines = 0
    kept_lines = 0
    n_words = len(words_to_keep)
    out_filename = "{}.filtered".format(sys.argv[1])
    with open(out_filename, "w") as out_file:
        for i, word in enumerate(list(words_to_keep)):
            vec = ['0.0'] * n_words
            vec[i] = '1.0'
            vec = [word.encode('utf-8')] + vec
            out_file.write('%s\n' % ' '.join(vec))
    # with open(sys.argv[1]) as in_file:
    #     with open(out_filename, "w") as out_file:
    #         for line in in_file.readlines():
    #             total_lines += 1
    #             word = line.split()[0]
    #             if word in words_to_keep:
    #                 kept_lines += 1
    #                 out_file.write(line)

    print "Kept {} out of {} lines.".format(kept_lines, total_lines)
    print "Wrote result to {}.".format(out_filename)
