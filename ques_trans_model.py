import operator
import random
import math
import json
import threading
import numpy as np
import tensorflow as tf
import pickle
import util
# import coref_ops
# import conll
import metrics

class CorefModel(object):
  def __init__(self, config):
    self.config = config
    self.embedding_info = [(emb["size"], emb["lowercase"]) for emb in config["embeddings"]]
    self.embedding_size = sum(size for size, _ in self.embedding_info)
    self.char_embedding_size = config["char_embedding_size"]
    self.char_dict = util.load_char_dict(config["char_vocab_path"])
    self.embedding_dicts = [util.load_embedding_dict(emb["path"], emb["size"], emb["format"]) for emb in config["embeddings"]]
    self.max_mention_width = config["max_mention_width"]
    self.genres = { g:i for i,g in enumerate(config["genres"]) }
    self.eval_data = None # Load eval data lazily.

    input_props = []
    input_props.append((tf.float32, [None, None, self.embedding_size])) # Question embeddings.
    input_props.append((tf.float32, [None, None, self.embedding_size])) # Transcript embeddings.
    input_props.append((tf.int32, [None])) # Labels.
    input_props.append((tf.bool, [])) # Is training.

    self.queue_input_tensors = [tf.placeholder(dtype, shape) for dtype, shape in input_props]
    dtypes, shapes = zip(*input_props)
    queue = tf.PaddingFIFOQueue(capacity=10, dtypes=dtypes, shapes=shapes)
    self.enqueue_op = queue.enqueue(self.queue_input_tensors)
    self.input_tensors = queue.dequeue()

    self.predictions, self.loss = self.get_predictions_and_loss(*self.input_tensors)
    self.global_step = tf.Variable(0, name="global_step", trainable=False)
    self.reset_global_step = tf.assign(self.global_step, 0)
    learning_rate = tf.train.exponential_decay(self.config["learning_rate"], self.global_step,
                                               self.config["decay_frequency"], self.config["decay_rate"], staircase=True)
    trainable_params = tf.trainable_variables()
    gradients = tf.gradients(self.loss, trainable_params)
    gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_gradient_norm"])
    optimizers = {
      "adam" : tf.train.AdamOptimizer,
      "sgd" : tf.train.GradientDescentOptimizer
    }
    optimizer = optimizers[self.config["optimizer"]](learning_rate)
    self.train_op = optimizer.apply_gradients(zip(gradients, trainable_params), global_step=self.global_step)

  def start_enqueue_thread(self, session):
      with open(self.config["train_path"]) as f:
        train_examples = [json.loads(jsonline) for jsonline in f.readlines()]
      def _enqueue_loop():
        while True:
          random.shuffle(train_examples)
          for example in train_examples:
            tensorized_example = self.tensorize_example(example, is_training=True)
            feed_dict = dict(zip(self.queue_input_tensors, tensorized_example))
            session.run(self.enqueue_op, feed_dict=feed_dict)
      enqueue_thread = threading.Thread(target=_enqueue_loop)
      enqueue_thread.daemon = True
      enqueue_thread.start()

  def tensorize_mentions(self, mentions):
    if len(mentions) > 0:
      starts, ends = zip(*mentions)
    else:
      starts, ends = [], []
    return np.array(starts), np.array(ends)

  def tensorize_example(self, example, is_training, oov_counts=None):
    label = np.array([example['label']])
    question = example['question'].split(' ')
    trans = example['script'].split(' ')

    # clusters = example["clusters"]

    # gold_mentions = sorted(tuple(m) for m in util.flatten(clusters))
    # gold_mention_map = {m:i for i,m in enumerate(gold_mentions)}
    # cluster_ids = np.zeros(len(gold_mentions))
    # for cluster_id, cluster in enumerate(clusters):
    #   for mention in cluster:
    #     cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id

    # sentences = example["sentences"]
    # num_words = sum(len(s) for s in sentences)
    # speakers = util.flatten(example["speakers"])

    # assert num_words == len(speakers)

    # max_sentence_length = max(len(s) for s in sentences)
    # max_word_length = max(max(max(len(w) for w in s) for s in sentences), max(self.config["filter_widths"]))
    ques_emb = np.zeros([1, len(question), self.embedding_size])
    trans_emb = np.zeros([1, len(trans), self.embedding_size])
    # char_index = np.zeros([len(sentences), max_sentence_length, max_word_length])
    ques_len = np.array([len(question)])
    trans_len = np.array([len(trans)])
    for j, word in enumerate(question):
      current_dim = 0
      for k, (d, (s,l)) in enumerate(zip(self.embedding_dicts, self.embedding_info)):
        if l:
          current_word = word.lower()
        else:
          current_word = word
        if oov_counts is not None and current_word not in d:
          oov_counts[k] += 1
        ques_emb[0, j, current_dim:current_dim + s] = util.normalize(d[current_word])
        current_dim += s
      # char_index[i, j, :len(word)] = [self.char_dict[c] for c in word]
    
    for j, word in enumerate(trans):
      current_dim = 0
      for k, (d, (s,l)) in enumerate(zip(self.embedding_dicts, self.embedding_info)):
        if l:
          current_word = word.lower()
        else:
          current_word = word
        if oov_counts is not None and current_word not in d:
          oov_counts[k] += 1
        trans_emb[0, j, current_dim:current_dim + s] = util.normalize(d[current_word])
        current_dim += s
      # char_index[i, j, :len(word)] = [self.char_dict[c] for c in word]

    # speaker_dict = { s:i for i,s in enumerate(set(speakers)) }
    # speaker_ids = np.array([speaker_dict[s] for s in speakers])

    # doc_key = example["doc_key"]
    # genre = self.genres[doc_key[:2]]

    # gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)

    # if is_training and len(sentences) > self.config["max_training_sentences"]:
    #   return self.truncate_example(word_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids)
    # else:
    return ques_emb, trans_emb, label, is_training # char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids


  def get_predictions_and_loss(self, ques_emb, trans_emb, label, is_training):
    self.dropout = 1 - (tf.to_float(is_training) * self.config["dropout_rate"])
    self.lexical_dropout = 1 - (tf.to_float(is_training) * self.config["lexical_dropout_rate"])

    num_sentences = tf.shape(ques_emb)[0]
    # max_sentence_length = tf.shape(ques_emb)[1]

    ques_len = tf.shape(ques_emb)[1]
    trans_len = tf.shape(trans_emb)[1]

    ques_len_lstm = tf.reshape(ques_len, [1])
    trans_len_lstm = tf.reshape(trans_len, [1])

    # text_emb_list = [word_emb]

    # if self.config["char_embedding_size"] > 0:
    #   char_emb = tf.gather(tf.get_variable("char_embeddings", [len(self.char_dict), self.config["char_embedding_size"]]), char_index) # [num_sentences, max_sentence_length, max_word_length, emb]
    #   flattened_char_emb = tf.reshape(char_emb, [num_sentences * max_sentence_length, util.shape(char_emb, 2), util.shape(char_emb, 3)]) # [num_sentences * max_sentence_length, max_word_length, emb]
    #   flattened_aggregated_char_emb = util.cnn(flattened_char_emb, self.config["filter_widths"], self.config["filter_size"]) # [num_sentences * max_sentence_length, emb]
    #   aggregated_char_emb = tf.reshape(flattened_aggregated_char_emb, [num_sentences, max_sentence_length, util.shape(flattened_aggregated_char_emb, 1)]) # [num_sentences, max_sentence_length, emb]
    #   text_emb_list.append(aggregated_char_emb)

    # text_emb = tf.concat(text_emb_list, 2)
    ques_emb = tf.nn.dropout(ques_emb, self.lexical_dropout)
    trans_emb = tf.nn.dropout(trans_emb, self.lexical_dropout)

    ques_len_mask = tf.sequence_mask(ques_len, maxlen=ques_len)
    ques_len_mask = tf.reshape(ques_len_mask, [num_sentences * ques_len])

    trans_len_mask = tf.sequence_mask(trans_len, maxlen=trans_len)
    trans_len_mask = tf.reshape(trans_len_mask, [num_sentences * trans_len])

    # with tf.variable_scope("question_lstm"):
    #   ques_outputs = self.encode_sentences(ques_emb, ques_len_lstm, ques_len_mask)
    #   ques_outputs = tf.nn.dropout(ques_outputs, self.dropout)

    # with tf.variable_scope("transcript_lstm"):
    #   trans_outputs = self.encode_sentences(trans_emb, trans_len_lstm, trans_len_mask)
    #   trans_outputs = tf.nn.dropout(trans_outputs, self.dropout)

    with tf.variable_scope("question_cnn5", reuse=tf.AUTO_REUSE):
      ques_outputs = util.cnn(ques_emb, [5], 200)
    with tf.variable_scope("question_cnn3", reuse=tf.AUTO_REUSE):
      ques_outputs = util.cnn(ques_outputs, [3], 200)[0]
      ques_outputs = tf.nn.dropout(ques_outputs, self.dropout)

    with tf.variable_scope("transcript_cnn5", reuse=tf.AUTO_REUSE):
      trans_outputs = util.cnn(trans_emb, [5], 200)
    with tf.variable_scope("transcript_cnn3", reuse=tf.AUTO_REUSE):
      trans_outputs = util.cnn(trans_outputs, [3], 200)[0]
      trans_outputs = tf.nn.dropout(trans_outputs, self.dropout)

    ques_query = tf.reduce_mean(ques_outputs, 0, keepdims=True)

    # ques_lstm_emb = tf.gather(tf.squeeze(ques_outputs, 0), [ques_len - 1])
    # trans_lstm_emb = tf.squeeze(trans_outputs, 0)

    ques_tiled = tf.tile(ques_query, [trans_len, 1])

    hist_att_emb = tf.concat([ques_tiled, trans_outputs], 1)
    with tf.variable_scope("context_att"):
      att_logits = util.ffnn(hist_att_emb, 1, 150, 1, self.dropout)
    
    context_att = tf.nn.softmax(att_logits, dim=0)
    hist_emb2 = tf.reduce_sum(context_att * trans_outputs, 0, keepdims=True)

    hist_tiled = tf.tile(hist_emb2, [ques_len, 1])
    ques_att_emb = tf.concat([hist_tiled, ques_outputs], 1)
    with tf.variable_scope("question_att"):
      att_logits = util.ffnn(ques_att_emb, 1, 150, 1, self.dropout)

    question_att = tf.nn.softmax(att_logits, dim=0)
    ques_emb2 = tf.reduce_sum(question_att * ques_outputs, 0, keepdims=True)

    pair_emb = tf.concat([ques_emb2, hist_emb2], 1)

    logits = util.ffnn(pair_emb, 2, 150, 1, self.dropout)

    score = tf.nn.sigmoid(logits)
    score = tf.reduce_sum(score)
    label = tf.reduce_sum(label)
    label = tf.cast(label, tf.float32)
    self.label = label
    self.score = score
    loss = tf.reduce_sum((label - score) * (label - score))

    # loss = tf.cond(tf.cast(tf.reshape(label, []), tf.bool), lambda: 1 - score, lambda: score)
    return score, loss

  def softmax_loss(self, antecedent_scores, antecedent_labels):
    gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels)) # [num_mentions, max_ant + 1]
    marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1]) # [num_mentions]
    log_norm = tf.reduce_logsumexp(antecedent_scores, [1]) # [num_mentions]
    return log_norm - marginalized_gold_scores # [num_mentions]

  def encode_sentences(self, text_emb, text_len, text_len_mask):
    num_sentences = tf.shape(text_emb)[0]
    max_sentence_length = tf.shape(text_emb)[1]

    # Transpose before and after for efficiency.
    inputs = tf.transpose(text_emb, [1, 0, 2]) # [max_sentence_length, num_sentences, emb]

    with tf.variable_scope("fw_cell"):
      cell_fw = util.CustomLSTMCell(self.config["lstm_size"], num_sentences, self.dropout)
      preprocessed_inputs_fw = cell_fw.preprocess_input(inputs)
    with tf.variable_scope("bw_cell"):
      cell_bw = util.CustomLSTMCell(self.config["lstm_size"], num_sentences, self.dropout)
      preprocessed_inputs_bw = cell_bw.preprocess_input(inputs)
      preprocessed_inputs_bw = tf.reverse_sequence(preprocessed_inputs_bw,
                                                   seq_lengths=text_len,
                                                   seq_dim=0,
                                                   batch_dim=1)
    state_fw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_fw.initial_state.c, [num_sentences, 1]), tf.tile(cell_fw.initial_state.h, [num_sentences, 1]))
    state_bw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_bw.initial_state.c, [num_sentences, 1]), tf.tile(cell_bw.initial_state.h, [num_sentences, 1]))
    with tf.variable_scope("lstm"):
      with tf.variable_scope("fw_lstm"):
        fw_outputs, fw_states = tf.nn.dynamic_rnn(cell=cell_fw,
                                                  inputs=preprocessed_inputs_fw,
                                                  sequence_length=text_len,
                                                  initial_state=state_fw,
                                                  time_major=True)
      with tf.variable_scope("bw_lstm"):
        bw_outputs, bw_states = tf.nn.dynamic_rnn(cell=cell_bw,
                                                  inputs=preprocessed_inputs_bw,
                                                  sequence_length=text_len,
                                                  initial_state=state_bw,
                                                  time_major=True)

    bw_outputs = tf.reverse_sequence(bw_outputs,
                                     seq_lengths=text_len,
                                     seq_dim=0,
                                     batch_dim=1)

    text_outputs = tf.concat([fw_outputs, bw_outputs], 2)
    return tf.transpose(text_outputs, [1, 0, 2]) # [num_sentences, max_sentence_length, emb]
    # return self.flatten_emb_by_sentence(text_outputs, text_len_mask)

  def evaluate_mentions(self, candidate_starts, candidate_ends, mention_starts, mention_ends, mention_scores, gold_starts, gold_ends, example, evaluators):
    text_length = sum(len(s) for s in example["sentences"])
    gold_spans = set(zip(gold_starts, gold_ends))

    if len(candidate_starts) > 0:
      sorted_starts, sorted_ends, _ = zip(*sorted(zip(candidate_starts, candidate_ends, mention_scores), key=operator.itemgetter(2), reverse=True))
    else:
      sorted_starts = []
      sorted_ends = []

    for k, evaluator in evaluators.items():
      if k == -3:
        predicted_spans = set(zip(candidate_starts, candidate_ends)) & gold_spans
      else:
        if k == -2:
          predicted_starts = mention_starts
          predicted_ends = mention_ends
        elif k == 0:
          is_predicted = mention_scores > 0
          predicted_starts = candidate_starts[is_predicted]
          predicted_ends = candidate_ends[is_predicted]
        else:
          if k == -1:
            num_predictions = len(gold_spans)
          else:
            num_predictions = (k * text_length) / 100
          predicted_starts = sorted_starts[:num_predictions]
          predicted_ends = sorted_ends[:num_predictions]
        predicted_spans = set(zip(predicted_starts, predicted_ends))
      evaluator.update(gold_set=gold_spans, predicted_set=predicted_spans)

  def get_predicted_antecedents(self, antecedents, antecedent_scores):
    predicted_antecedents = []
    for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
      if index < 0:
        predicted_antecedents.append(-1)
      else:
        predicted_antecedents.append(antecedents[i, index])
    return predicted_antecedents

  def get_predicted_clusters(self, mention_starts, mention_ends, predicted_antecedents):
    mention_to_predicted = {}
    predicted_clusters = []
    for i, predicted_index in enumerate(predicted_antecedents):
      if predicted_index < 0:
        continue
      assert i > predicted_index
      predicted_antecedent = (int(mention_starts[predicted_index]), int(mention_ends[predicted_index]))
      if predicted_antecedent in mention_to_predicted:
        predicted_cluster = mention_to_predicted[predicted_antecedent]
      else:
        predicted_cluster = len(predicted_clusters)
        predicted_clusters.append([predicted_antecedent])
        mention_to_predicted[predicted_antecedent] = predicted_cluster

      mention = (int(mention_starts[i]), int(mention_ends[i]))
      predicted_clusters[predicted_cluster].append(mention)
      mention_to_predicted[mention] = predicted_cluster

    predicted_clusters = [tuple(pc) for pc in predicted_clusters]
    mention_to_predicted = { m:predicted_clusters[i] for m,i in mention_to_predicted.items() }

    return predicted_clusters, mention_to_predicted

  def evaluate_coref(self, mention_starts, mention_ends, predicted_antecedents, gold_clusters, evaluator):
    gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
    mention_to_gold = {}
    for gc in gold_clusters:
      for mention in gc:
        mention_to_gold[mention] = gc

    predicted_clusters, mention_to_predicted = self.get_predicted_clusters(mention_starts, mention_ends, predicted_antecedents)
    evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
    return predicted_clusters

  def load_eval_data(self):
    if self.eval_data is None:
      oov_counts = [0 for _ in self.embedding_dicts]
      with open(self.config["eval_path"]) as f:
        self.eval_data = map(lambda example: (self.tensorize_example(example, is_training=False, oov_counts=oov_counts), example), (json.loads(jsonline) for jsonline in f.readlines()))
      # num_words = sum(tensorized_example[2].sum() for tensorized_example, _ in self.eval_data)
      # for emb, c in zip(self.config["embeddings"], oov_counts):
      #   print("OOV rate for {}: {:.2f}%".format(emb["path"], (100.0 * c) / num_words))
      # print("Loaded {} eval examples.".format(len(self.eval_data)))

  def evaluate(self, session, official_stdout=False):
    self.load_eval_data()

    coref_predictions = {}
    coref_evaluator = metrics.CorefEvaluator()

    scores_list = []
    eval_size = len(self.eval_data)

    for example_num, (tensorized_example, example) in enumerate(self.eval_data):
      if example_num % 100 == 0:
        print '%d / %d pairs evaluated' % (example_num, eval_size)
      ques_emb, trans_emb, label, is_training = tensorized_example

      feed_dict = {i:t for i,t in zip(self.input_tensors, tensorized_example)}
      score = session.run(self.predictions, feed_dict=feed_dict)

      scores_list.append([example['q_id'], example['v_id'], score])
    
    pickle.dump([[], [], scores_list], open('scores.pkg', 'wb'))
