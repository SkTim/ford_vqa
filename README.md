# ford_vqa

## Training

Run `./setup_training.sh` and it will download and pretrained word embeddings.

Run `python singleton.py EXP_NAME` for training. EXP_NAME should be defined in experiments.conf.

Example: `python singleton.py best`.

## Evaluating

Run `./copy_exp.sh EXP_NAME ITER_NUM`.

For example: `./copy_exp.sh best 10000`.

Run `python test_single.py EXP_NAME`. This generates scores.pkg file.

For example: `python test_single.py best`.

To get final scores, run `python eval.py 0 (or 1)`. 0 for evaluating on video clips and 1 for entire video.

## Format of scores.pkg
Test scores can be loaded by `_, _, test_scores = pickle.load(open('scores.pkg'))`.

test_score = [['q_id', 'v_id', score], ....]
