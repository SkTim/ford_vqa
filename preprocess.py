import re
import sys
import csv
import json
import random
from nltk import word_tokenize

def select_str(line):
    stop_words = set(["WEBVTT", "Kind: captions", "Language: en"])
    if len(line) == 0:
        return False
    if line in stop_words:
        return False
    if not line[0].isalpha():
        return False
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

def get_length(s):
    return len(s.split(' '))

def process_data(input_file, output_file, num_samples):
    v_ids, q_ids, scripts, questions = read_csv(input_file)
    scripts = [process_script(x) for x in scripts]
    questions = [process_question(x) for x in questions]
    q_dict = {x: y for x, y in zip(q_ids, questions)}
    v_dict = {x: y for x, y in zip(v_ids, scripts)}
    v_ids, q_ids, labels = negative_sampling(v_ids, q_ids, num_samples)
    scripts = [v_dict[x] for x in v_ids]
    questions = [q_dict[x] for x in q_ids]
    samples = [{'question': x, 'script': y, 'label': z, 'q_len': get_length(x), 's_len': get_length(y), 'v_id': v, 'q_id': q} for x, y, z, v, q in zip(questions, scripts, labels, v_ids, q_ids)]
    out_handle = open('%s_%s.jsonlines' % (output_file, sys.argv[2]), 'w')
    for sample in samples:
        out_handle.write(json.dumps(sample))
        out_handle.write('\n')
    out_handle.close()

def all_pairs(vq_dict, v_ids, q_ids):
    qv_dict = {}
    v_set = set(v_ids)
    for v, qs in vq_dict.items():
        for q in qs:
            qv_dict[q] = v
    new_v = []
    new_q = []
    labels = []
    for q in q_ids:
        for v in v_set:
            new_q.append(q)
            new_v.append(v)
            if v == qv_dict[q]:
                labels.append(1)
            else:
                labels.append(0)
    return new_v, new_q, labels

def process_test(input_file, output_file, num_samples):
    v_ids, q_ids, scripts, questions = read_csv(input_file)

    scripts = [process_script(x) for x in scripts]
    questions = [process_question(x) for x in questions]
    q_dict = {x: y for x, y in zip(q_ids, questions)}
    v_dict = {x: y for x, y in zip(v_ids, scripts)}

    vq_dict = {}
    for v, q in zip(v_ids, q_ids):
        if v not in vq_dict:
            vq_dict[v] = [q]
        else:
            vq_dict[v].append(q)
    for v, qs in vq_dict.items():
        vq_dict[v] = qs[:min(len(qs), 5)]
    v_ids = []
    q_ids = []
    for v, qs in vq_dict.items():
        for q in qs:
            v_ids.append(v)
            q_ids.append(q)

    v_ids, q_ids, labels = all_pairs(vq_dict, v_ids, q_ids)

    scripts = [v_dict[x] for x in v_ids]
    questions = [q_dict[x] for x in q_ids]
    samples = [{'question': x, 'script': y, 'label': z, 'v_id': v_id, 'q_id': q_id} for x, y, z, v_id, q_id in zip(questions, scripts, labels, v_ids, q_ids)]
    out_handle = open('%s_%s.jsonlines' % (output_file, sys.argv[2]), 'w')
    for sample in samples:
        out_handle.write(json.dumps(sample))
        out_handle.write('\n')
    out_handle.close()

if __name__ == '__main__':
    if sys.argv[1] == 'train' or sys.argv[1] == 'dev':
        process_data('data/%s_%s.csv' % (sys.argv[1], sys.argv[2]), sys.argv[1], 0)
    else:
        process_data('data/%s_%s.csv' % (sys.argv[1], sys.argv[2]), sys.argv[1], 0)
