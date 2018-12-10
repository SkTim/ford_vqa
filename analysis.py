import json

def UNK_words():
    dataset = open(r'train.jsonlines').readlines()
    # dataset += open(r'test.jsonlines').readlines()
    samples = [json.loads(x) for x in dataset]
    sentences = ' '.join(['%s %s' % (x['question'], x['script']) for x in samples])
    words = set(sentences.split(' '))
    glove = open('glove.840B.300d.txt').readlines()
    n = 0.
    glove_words = set([x.split(' ')[0] for x in glove])
    for word in words:
        if word not in glove_words:
            n += 1
    print n, len(words), n / len(words)

if __name__ == '__main__':
    UNK_words()
