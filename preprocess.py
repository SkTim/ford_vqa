import re
import sys
import csv
import json
import random
from nltk import word_tokenize

"""WEBVTT
Kind: captions
Language: en
00:00:00.000 --> 00:00:03.421
[MUSIC]
00:00:03.421 --> 00:00:07.043
Your vehicle may have an available feature
that can help keep your distance from
00:00:07.043 --> 00:00:10.230
the car in front of you when
you use cruise control.
00:00:10.230 --> 00:00:14.120
It functions just like normal cruise
control, with a few exceptions.
00:00:14.120 --> 00:00:15.650
Let me tell you about it."""

def select_str(line):
    stop_words = set(["WEBVTT", "Kind: captions", "Language: en"])
    # print line
    if len(line) == 0:
        return False
    if line in stop_words:
        # print 'stop owrd'
        return False
    if not line[0].isalpha():
        # print 'not english'
        return False
    # print 'valid'
    return True

def process_script(s):
    lines = s.split('\n')
    processed_lines = []
    for line in lines:
        if select_str(line):
            processed_lines.append(line)
    words = word_tokenize(' '.join(processed_lines))
    processed_words = []
    for word in words:
        if select_str(word):
            processed_words.append(word)
    return ' '.join(processed_words).lower()

def process_question(q):
    words = word_tokenize(q)
    processed_words = []
    for word in words:
        if select_str(word):
            processed_words.append(word)
    return ' '.join(processed_words).lower()

def read_csv(fn):
    with open(fn, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        v_ids = []
        q_ids = []
        scripts = []
        questions = []
        for row in reader:
            v_ids.append(row['video_id'])
            q_ids.append(row['question_id'])
            scripts.append(row['video_trans'])
            questions.append(row['question'])
    return v_ids, q_ids, scripts, questions

def negative_sampling(v_ids, q_ids, num_samples):
    new_v = []
    new_q = []
    labels = []
    for i, q_id in enumerate(q_ids):
        n = 0
        v_id = v_ids[i]
        new_v.append(v_id)
        new_q.append(q_id)
        labels.append(1)
        while n < num_samples:
            v_id_neg = random.choice(v_ids)
            if v_id_neg != q_id:
                new_v.append(v_id_neg)
                new_q.append(q_id)
                labels.append(0)
                n += 1
    return new_v, new_q, labels

def process_data(input_file, output_file, num_samples):
    v_ids, q_ids, scripts, questions = read_csv(input_file)
    print len(q_ids)
    scripts = [process_script(x) for x in scripts]
    questions = [process_question(x) for x in questions]
    q_dict = {x: y for x, y in zip(q_ids, questions)}
    v_dict = {x: y for x, y in zip(v_ids, scripts)}
    v_ids, q_ids, labels = negative_sampling(v_ids, q_ids, num_samples)
    print len(v_ids)
    scripts = [v_dict[x] for x in v_ids]
    questions = [q_dict[x] for x in q_ids]
    samples = [{'question': x, 'script': y, 'label': z} for x, y, z in zip(questions, scripts, labels)]
    print len(samples)
    out_handle = open('%s.jsonlines' % output_file, 'w')
    for sample in samples:
        out_handle.write(json.dumps(sample))
        out_handle.write('\n')
    out_handle.close()

if __name__ == '__main__':
    process_data('data/%s.csv' % sys.argv[1], sys.argv[1], 1)