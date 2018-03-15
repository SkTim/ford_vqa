import sys
import pickle
import csv
import numpy as np

def sort_dict(d):
    return sorted(d.items(), key = lambda x: x[1], reverse = True)

def get_dict(s, e, data):
    slices = data[s:e + 1]# .tolist()
    scores = {x[1]: x[2] for x in slices}
    return scores

def num_nmap(s, e, d, n, v_label):
    hit = 0
    sscores = sort_dict(d)
    rscores = sscores[:n]
    for v, s in rscores:
        if v == v_label:
            hit = 1
            break
    return hit

def evaluate(scores, qv_map, q_index):
    hit_1, hit_5, hit_10 = [0, 0, 0]
    score_list = scores#.tolist()
    questions = q_index.keys()
    num_q = len(questions)
    for q_id in questions:
        v_label = qv_map[q_id]
        # s, e = get_range(q_id, scores)
        s, e = q_index[q_id]
        # print('%s %d %d' % (q_id, s, e))
        d = get_dict(s, e, scores)
        hit_1 += num_nmap(s, e, d, 1, v_label)
        hit_5 += num_nmap(s, e, d, 5, v_label)
        hit_10 += num_nmap(s, e, d, 10, v_label)
    return float(hit_1) / num_q, float(hit_5) / num_q, float(hit_10) / num_q

if __name__ == "__main__":
    scores = pickle.load(open('scores.pkg', 'rb'))
    train_scores, dev_scores, test_scores = scores
    
    if sys.argv[1] == 'train':
        task_scores = train_scores
    elif sys.argv[1] == 'dev':
        task_scores = dev_scores
    elif sys.argv[1] == 'test':
        task_scores = test_scores
    else:
        print('using test set')
        task_score = test_scores
    
    qv_map = {}
    with open('Ford_AMT_Questions_9k_Cleaned_ver2.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            qv_map[row['question_id']] = row['video_id']
    
    q_index = {'': [-1, -1]}
    curr_q = ''
    prev_q = ''
    for i, sample in enumerate(task_scores):
        q, v, s = sample
        curr_q = q
        if curr_q != prev_q:
            q_index[curr_q] = [0, 0]
            q_index[curr_q][0] = i
            q_index[prev_q][1] = i - 1
            prev_q = curr_q
    del(q_index[''])
    
    print(evaluate(task_scores, qv_map, q_index))
