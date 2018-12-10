import re
import sys
import csv
import json
import random

from nltk import word_tokenize
from preprocess import select_str

def get_time_stamp(line, i):
    h, m, s = line.split('-->')[i].split(':')
    return float(h) * 3600 + float(m) * 60 + float(s)

def get_video_length(script):
    lines = script.split('\n')
    max_len = 0
    last_line = ''
    for line in lines:
        if '-->' in line:
            last_line = line
    return get_time_stamp(last_line, 1)

def get_segment_length(script):
    lines = script.split('\n')
    max_len = 0
    time_stamps = []
    for line in lines:
        if '-->' in line:
            time_stamps.append(line)
    return get_time_stamp(time_stamps[-1], 1) - get_time_stamp(time_stamps[0], 0)

def video_stats():
    data = open(r'test_set.jsonlines').readlines()
    video_dict = {}

    for line in data:
        example = json.loads(line)
        v_id = example['v_id']
        tv = '_'.join(v_id.split('_')[:2])
        if tv in video_dict:
            video_dict[tv].append(v_id)
        else:
            video_dict[tv] = [v_id]

    print(len(video_dict))
    lengths = [len(value) for key, value in video_dict.items()]
    print(sum(lengths))
    print(sum(lengths) * 1.0 / len(video_dict))

def time_stats():
    with open(r'data/test_1.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        video_dict = {}
        for row in reader:
            v_id = row['video_id']
            # v_id = '_'.join(v_id.split('_')[:2])
            script = row['video_trans']
            # length = get_video_length(script)
            length = get_segment_length(script)
            if v_id in video_dict:
                video_dict[v_id] = length if length > video_dict[v_id] else video_dict[v_id]
            else:
                video_dict[v_id] = length
    lengths = [value for key, value in video_dict.items()]
    print(sum(lengths))
    print(max(lengths))
    print(min(lengths))
    print(sum(lengths) * 1.0 / len(video_dict))

def ques_stats():
    data = open(r'train_1.jsonlines').readlines() + open(r'test_1.jsonlines').readlines()
    print(len(data))
    dictionary = set([])
    q_lengths = []
    for line in data:
        example = json.loads(line)
        question = example['question']
        q_lengths.append(example['q_len'])
        words = question.split(' ')
        for word in words:
            dictionary.add(word)
    print(len(dictionary))
    print(sum(q_lengths))
    print(sum(q_lengths) * 1.0 / len(data))

def trans_stats():
    data = open(r'test_set.jsonlines').readlines()
    dictionary = set([])
    s_lengths = []
    for line in data:
        example = json.loads(line)
        script = example['script']
        s_lengths.append(example['s_len'])
        words = script.split(' ')
        for word in words:
            dictionary.add(word)
    print(len(dictionary))
    print(sum(s_lengths))
    print(sum(s_lengths) * 1.0 / len(data))
if __name__ == '__main__':
    # time_stats()
    ques_stats()
    # trans_stats()
    # video_stats()
