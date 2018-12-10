#!/usr/bin/env python

import os
import re
import sys
sys.path.append(os.getcwd())
import time
import pickle
import random

import numpy as np
import tensorflow as tf
import ques_trans_model as cm
import util

if __name__ == "__main__":
  # if "GPU" in os.environ:
  #   util.set_gpus(int(os.environ["GPU"]))
  # else:
  #   util.set_gpus()

  if len(sys.argv) > 1:
    name = sys.argv[1]
    print("Running experiment: {} (from command-line argument).".format(name))
  else:
    name = os.environ["EXP"]
    print("Running experiment: {} (from environment variable).".format(name))

  config = util.get_config("experiments.conf")[name]
  config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))
  # config["eval_path"] = "test_100.jsonlines"
  config["eval_path"] = "test_%s.jsonlines" % sys.argv[2]

  util.print_config(config)
  model = cm.CorefModel(config)

  saver = tf.train.Saver()
  log_dir = config["log_dir"]

  with tf.Session() as session:
    checkpoint_path = os.path.join(log_dir, "model.max.ckpt")
    print("Evaluating {}".format(checkpoint_path))
    saver.restore(session, checkpoint_path)
    model.eval_enqueue_thread(session)
    scores = []
    n = len(open(config["eval_path"], 'r').readlines())
    num_samples = 1
    # num_samples = n
    # num_samples = n * n
    for i in range(num_samples):
      # if i % 1000 == 0:
      print('%s example: %d/%d %s' % ('*' * 40, i, num_samples, '*' * 40))
      score = session.run(model.score)
      scores.append(score)
    pickle.dump([[], [], scores], open('scores_%s.pkg' % sys.argv[1], 'wb'))
