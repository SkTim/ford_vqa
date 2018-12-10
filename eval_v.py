import sys
import json
import pickle
import numpy as np

if __name__ == '__main__':
    test_set = 'test_set.jsonlines'
    test_fn = 'test_%s.jsonlines' % sys.argv[1]
    
    v_ids = []
    for line in open(test_set, 'r').readlines():
        example = json.loads(line)
        v_ids.append(example['v_id'])
    
    ns = len(open(test_fn, 'r').readlines())
    scores = pickle.load(open('scores_exp%s.pkg' % sys.argv[1], 'rb'))[2][0].tolist()
    # scores = np.array(scores).reshape([ns, ns]).tolist()
    
    n1 = 0.
    n5 = 0.
    n10 = 0.

    test_examples = open(test_fn, 'r').readlines()
    
    for i, line in enumerate(scores):
        
        target = json.loads(test_examples[i])['v_id']
        
        t_video = '_'.join(target.split('_')[:2])
        target = t_video
        scores = line
        
        # for j, score in enumerate(line):
        #     if t_video not in v_ids[j]:
        #         scores[j] = -100000

        # idx = range(ns)
        slist = [[x, v] for x, v in zip(scores, v_ids)]
        sorted_idx = sorted(slist, key=lambda x: x[0], reverse=True)
        
        v_set = set([])
        sorted_v = []
        for x, v in sorted_idx:
            ev = '_'.join(v.split('_')[:2])
            if ev not in v_set:
                v_set.add(ev)
                sorted_v.append(ev)
        
        s1 = sorted_v[:1]
        s5 = sorted_v[:5]
        s10 = sorted_v[:10]

        # print(target)
        # print(s10)
        # print(scores)
        # print('=' * 89)
        
        if target in s1:
            n1 += 1
        if target in s5:
            n5 += 1
        if target in s10:
            n10 += 1
    
    print(n1 / ns)
    print(n5 / ns)
    print(n10 / ns)

