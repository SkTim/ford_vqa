import json

if __name__ == '__main__':
    all_data = open(r'train_1.jsonlines').readlines() + \
            open(r'test_1.jsonlines').readlines()
    video_set = set([])
    test_set = []
    for line in all_data:
        example = json.loads(line)
        if example['v_id'] not in video_set:
            video_set.add(example['v_id'])
            test_set.append(line)
    open('test_set.jsonlines', 'w').write(''.join(test_set))

