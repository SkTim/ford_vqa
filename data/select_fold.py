import sys
import csv

def get_train_list(n):
    all_fold = range(1, 11)
    train_fold = []
    for i in all_fold:
        if i != n:
            train_fold.append(i)
    return train_fold

def make_test(n):
    text = open('fold_%d_Ford_AMT_Questions_9k_Cleaned_ver2.csv' % n).read()
    open('test_%d.csv' % n, 'w').write(text)

if __name__ == '__main__':
    test_fold = int(sys.argv[1])
    make_test(test_fold)
    train_folds = get_train_list(test_fold)

    head = ''
    data = []
    for i in train_folds:
        lines = open('fold_%d_Ford_AMT_Questions_9k_Cleaned_ver2.csv' % i).readlines()
        if head == '':
            data += lines[0]
            head = lines[0]
        data += lines[1:]
    open('train_%d.csv' % test_fold, 'w').write(''.join(data))
