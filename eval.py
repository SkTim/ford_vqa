import json
import pickle

def count_topn(qv_map, sort_map, n):
    match = 0.0
    for q, v in sort_map.items():
        v = set(v[:n])
        if qv_map[q] in v:
            match += 1
    return match

if __name__ == '__main__':
    _, _, test_scores = pickle.load(open('scores.pkg', 'rb'))

    eval_score_map = {}
    for q, v, s in test_scores:
        eval_score_map[(q, v)] = s

    examples = open(r'test.jsonlines').readlines()
    examples = [json.loads(x) for x in examples]
    
    dataset = []
    
    qv_map = {}
    score_map = {}
    
    for i, example in enumerate(examples):
        q = example['q_id']
        v = example['v_id']
        label = example['label']
        score = eval_score_map[(q, v)]
        dataset.append({'v_id': v,
            'q_id': q,
            'label': label,
            'score': score})
        if label == 1:
            qv_map[q] = v
        if q not in score_map:
            score_map[q] = {v: score}
        else:
            score_map[q][v] = score
    
    sort_map = {}
    for q, v in score_map.items():
        sorted_v = sorted(v.iteritems(), key = lambda x: x[1], reverse=True)
        sort_map[q] = [x[0] for x in sorted_v]

    top_1 = count_topn(qv_map, sort_map, 1) / len(qv_map)
    top_5 = count_topn(qv_map, sort_map, 5) / len(qv_map)
    top_10 = count_topn(qv_map, sort_map, 10) / len(qv_map)
    top_100 = count_topn(qv_map, sort_map, 100) / len(qv_map)
    print 'map@1 = %f, map@5 = %f, map@10 = %f, map@100 = %f' % (top_1, top_5, top_10, top_100)

