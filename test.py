import json
from preprocess import process_script

examples = open(r'train.jsonlines').readlines()
ne = len(examples)
nq = 0.
nt = 0.
for e in examples:
    e = json.loads(e)
    q = len(e['question'].split(' '))
    t = len(e['script'].split(' '))
    if q > 20:
        # nq = q
        nq += 1
    if t > 200:
        # nt = t
        nt += 1

print(nq, nt)
# print(nq / ne, nt / ne)
