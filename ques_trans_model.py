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
    input_props.append((tf.int32, [None])) # q_len.
    input_props.append((tf.int32, [None])) # s_len.
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
          print('*' * 89)
          random.shuffle(train_examples)
          for i in range(0, len(train_examples), 16):
            examples = train_examples[i:i + 16]
            tensorized_example = [self.tensorize_example(x, True, 40, 200) for x in examples]
            ques_emb = np.concatenate([x[0] for x in tensorized_example], axis=0)
            trans_emb = np.concatenate([x[1] for x in tensorized_example], axis=0)
            labels = np.array([x[2] for x in tensorized_example])
            q_len = np.array([x[3] for x in tensorized_example])
            s_len = np.array([x[4] for x in tensorized_example])
            feed_dict = dict(zip(self.queue_input_tensors, [ques_emb, trans_emb, labels, q_len, s_len, True]))
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

  def tensorize_example(self, example, is_training, mlq, mlt, oov_counts=None):
    label = example['label']
    q_len = example['q_len']
    s_len = example['s_len']
    question = example['question'].split(' ')
    trans = example['script'].split(' ')
    
    ques_emb = np.zeros([1, mlq, self.embedding_size])
    trans_emb = np.zeros([1, mlt, self.embedding_size])
    # char_index = np.zeros([len(sentences), max_sentence_length, max_word_length])
    ques_len = np.array([len(question)])
    trans_len = np.array([len(trans)])
    for j, word in enumerate(question):
      if j > mlq - 1:
        break
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
      if j > mlt - 1:
        break
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
    return ques_emb, trans_emb, label, q_len, s_len, is_training # char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids


  def get_predictions_and_loss(self, ques_emb, trans_emb, label, ques_len, trans_len, is_training):
    self.dropout = 1 - (tf.to_float(is_training) * self.config["dropout_rate"])
    self.lexical_dropout = 1 - (tf.to_float(is_training) * self.config["lexical_dropout_rate"])

    num_questions = tf.shape(ques_emb)[0]
    num_sentences = tf.shape(trans_emb)[0]
    # num_sentences = tf.shape(ques_emb)[0]
    # max_sentence_length = tf.shape(ques_emb)[1]

    # ques_len = tf.shape(ques_emb)[1]
    # trans_len = tf.shape(trans_emb)[1]

    # ques_len_lstm = tf.reshape(ques_len, [1])
    # trans_len_lstm = tf.reshape(trans_len, [1])

    # text_emb_list = [word_emb]

    # if self.config["char_embedding_size"] > 0:
    #   char_emb = tf.gather(tf.get_variable("char_embeddings", [len(self.char_dict), self.config["char_embedding_size"]]), char_index) # [num_sentences, max_sentence_length, max_word_length, emb]
    #   flattened_char_emb = tf.reshape(char_emb, [num_sentences * max_sentence_length, util.shape(char_emb, 2), util.shape(char_emb, 3)]) # [num_sentences * max_sentence_length, max_word_length, emb]
    #   flattened_aggregated_char_emb = util.cnn(flattened_char_emb, self.config["filter_widths"], self.config["filter_size"]) # [num_sentences * max_sentence_length, emb]
    #   aggregated_char_emb = tf.reshape(flattened_aggregated_char_emb, [num_sentences, max_sentence_length, util.shape(flattened_aggregated_char_emb, 1)]) # [num_sentences, max_sentence_length, emb]
    #   text_emb_list.append(aggregated_char_emb)

    # text_emb = tf.concat(text_emb_list, 2)
    # ques_emb = tf.matmul(ques_emb, tf.zeros([100, 5]))
    with tf.variable_scope('ques_emb'):
      ques_emb = util.projection(ques_emb, 600)
      # ques_emb = tf.tanh(ques_emb)
    with tf.variable_scope('trans_emb'):
      trans_emb = util.projection(trans_emb, 600)
      # trans_emb = tf.tanh(trans_emb)
    
    ques_emb = tf.nn.dropout(ques_emb, self.lexical_dropout)
    trans_emb = tf.nn.dropout(trans_emb, self.lexical_dropout)

    ques_len_mask = tf.sequence_mask(ques_len, maxlen=40, dtype=tf.float32)
    ques_len_mask = tf.reshape(ques_len_mask, [num_questions, -1, 1])

    trans_len_mask = tf.sequence_mask(trans_len, maxlen=200, dtype=tf.float32)
    trans_len_mask = tf.reshape(trans_len_mask, [num_sentences, -1, 1])

    '''
    with tf.variable_scope("question_lstm"):
      ques_outputs = self.encode_sentences(ques_emb, ques_len_lstm, ques_len_mask)
      ques_outputs = tf.nn.dropout(ques_outputs, self.dropout)

    with tf.variable_scope("transcript_lstm"):
      trans_outputs = self.encode_sentences(trans_emb, trans_len_lstm, trans_len_mask)
      trans_outputs = tf.nn.dropout(trans_outputs, self.dropout)
    '''
    # '''
    # with tf.variable_scope("question_cnn5"):
    ques_outputs = tf.layers.conv1d(ques_emb, 600, 5, padding="same", activation=tf.nn.tanh)
    ques_outputs = tf.nn.dropout(ques_outputs, self.dropout)
    
    trans_outputs = tf.layers.conv1d(trans_emb, 600, 5, padding="same", activation=tf.nn.tanh)
    trans_outputs = tf.nn.dropout(trans_outputs, self.dropout)
    
    # ques_outputs = tf.matmul(ques_outputs, tf.zeros([100, 5]))
    
    # with tf.variable_scope("question_cnn3"):
    # ques_outputs = tf.layers.conv1d(ques_outputs, 600, 3, padding="same", activation=tf.nn.relu)
    # ques_outputs = tf.nn.dropout(ques_outputs, self.dropout)[0]
    
    # ques_outputs = tf.matmul(ques_outputs, tf.zeros([100, 5]))

    # ques_outputs = tf.matmul(ques_outputs, tf.zeros([100, 5]))
    with tf.variable_scope("qh_score"):
      head_scores = util.projection(ques_outputs, 1) + tf.log(ques_len_mask)
      head_att = tf.nn.softmax(head_scores, dim=1)
      ques_emb2 = tf.reduce_sum(head_att * ques_outputs, 1)
      # ques_query = tf.reduce_sum(head_att * ques_outputs, 1, keepdims=True)
    
    with tf.variable_scope("th_score"):
      head_scores = util.projection(trans_outputs, 1) + tf.log(trans_len_mask)
      head_att = tf.nn.softmax(head_scores, dim=1)
      trans_emb2 = tf.reduce_sum(head_att * trans_outputs, 1)
      # trans_query = tf.reduce_sum(head_att * trans_outputs, 1, keepdims=True)

    # ques_lstm_emb = tf.gather(tf.squeeze(ques_outputs, 0), [ques_len - 1])
    # trans_lstm_emb = tf.squeeze(trans_outputs, 0)

    '''
    ques_tiled = tf.tile(ques_query, [1, 200, 1])

    hist_att_emb = tf.concat([ques_tiled, trans_outputs], 2)
    with tf.variable_scope("context_att"):
      att_logits = util.projection(hist_att_emb, 1) + tf.log(trans_len_mask)
    
    context_att = tf.nn.softmax(att_logits, dim=1)
    hist_emb2 = tf.reduce_sum(context_att * trans_outputs, 1, keepdims=True)

    hist_tiled = tf.tile(hist_emb2, [1, 40, 1])
    ques_att_emb = tf.concat([hist_tiled, ques_outputs], 2)
    with tf.variable_scope("question_att"):
      att_logits = util.projection(ques_att_emb, 1) + tf.log(ques_len_mask)

    question_att = tf.nn.softmax(att_logits, dim=1)
    ques_emb2 = tf.reduce_sum(question_att * ques_outputs, 1)

    # pair_emb = tf.concat([ques_emb2, hist_emb2, ques_emb2 * hist_emb2], 1)

    # logits = util.ffnn(pair_emb, 2, 150, 1, self.dropout)
    trans_emb2 = tf.squeeze(hist_emb2, 1)
    '''
    
    logits = tf.matmul(ques_emb2, tf.transpose(trans_emb2, [1, 0]))
    label = tf.eye(num_sentences)
    # loss = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits)
    loss = tf.cond(is_training, lambda: tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits), lambda: 0.)

    # logits = tf.reshape(tf.reduce_sum(ques_emb2 * hist_emb2), [])
    
    # score = tf.nn.sigmoid(logits)
    # score = tf.reduce_sum(score)
    # label = tf.reduce_sum(label)
    # label = tf.cast(label, tf.float32)
    # self.label = label
    # self.score = score
    # loss = tf.reduce_sum((label - score) * (label - score))

    # loss = tf.cond(tf.cast(tf.reshape(label, []), tf.bool), lambda: 1 - score, lambda: score)
    self.score = logits
    loss = tf.reduce_sum(loss)
    return logits, loss

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
        test_examples = [json.loads(jsonline) for jsonline in f.readlines()]
      return test_examples

  def load_test_set(self):
    with open(r'test_set.jsonlines') as f:
      test_set = [json.loads(jsonline) for jsonline in f.readlines()]
    return test_set

  def eval_enqueue_thread(self, session, official_stdout=False):
    oov_counts = [0 for _ in self.embedding_dicts]
    test_samples = self.load_eval_data()
    test_set = self.load_test_set()
    num_samples = len(test_samples)
    num_videos = len(test_set)
    def _enqueue_loop():
      # q, _, _, q_len, _, _ = self.tensorize_example(test_samples[i], False, 40, 200, oov_counts=oov_counts)
      ques = [self.tensorize_example(test_samples[j], False, 40, 200, oov_counts=oov_counts) for j in range(num_samples)]
      ques = [(x[0], x[3]) for x in ques]
      
      trans = [self.tensorize_example(test_set[j], False, 40, 200, oov_counts=oov_counts) for j in range(num_videos)]
      trans = [(x[1], x[4]) for x in trans]
      
      labels = np.zeros(num_videos)
      q_len = np.array([x[1] for x in ques])
      s_len = np.array([x[1] for x in trans])
      q = np.concatenate([x[0] for x in ques], axis=0)
      # q = np.concatenate([q for i in range(num_videos)], axis=0)
      t = np.concatenate([x[0] for x in trans], axis=0)
      print(q.shape)
      print(t.shape)
      # print(q_len.shape)
      # print(s_len.shape)
      feed_dict = dict(zip(self.queue_input_tensors, [q, t, labels, q_len, s_len, False]))
      session.run(self.enqueue_op, feed_dict=feed_dict)
    enqueue_thread = threading.Thread(target=_enqueue_loop)
    enqueue_thread.daemon = True
    enqueue_thread.start()
  
  def eval_enqueue_thread2(self, session, official_stdout=False):
    oov_counts = [0 for _ in self.embedding_dicts]
    test_samples = self.load_eval_data()
    test_set = self.load_test_set()
    num_samples = len(test_samples)
    num_videos = len(test_set)
    def _enqueue_loop():
      for i in range(num_samples):
        q, _, _, q_len, _, _ = self.tensorize_example(test_samples[i], False, 40, 200, oov_counts=oov_counts)
        trans = [self.tensorize_example(test_set[j], False, 40, 200, oov_counts=oov_counts) for j in range(num_videos)]
        trans = [(x[1], x[4]) for x in trans]
        labels = np.zeros(num_videos)
        q_len = np.array([q_len])
        # q_len = np.array([q_len] * num_videos)
        s_len = np.array([x[1] for x in trans])
        q = np.concatenate([q,], axis=0)
        # q = np.concatenate([q for i in range(num_videos)], axis=0)
        t = np.concatenate([x[0] for x in trans], axis=0)
        # print(q.shape)
        # print(t.shape)
        # print(q_len.shape)
        # print(s_len.shape)
        feed_dict = dict(zip(self.queue_input_tensors, [q, t, labels, q_len, s_len, False]))
        session.run(self.enqueue_op, feed_dict=feed_dict)
    enqueue_thread = threading.Thread(target=_enqueue_loop)
    enqueue_thread.daemon = True
    enqueue_thread.start()

  def evaluate_raw(self, session, official_stdout=False):
    ques_emb, trans_emb, labels, q_len, s_len = self.load_eval_data()
    scores_list = []
    for i, t in enumerate(trans_emb):
      feed_dict = dict(zip(self.queue_input_tensors, [ques_emb, t, labels, q_len, s_len, True]))
      score = session.run(self.predictions, feed_dict=feed_dict)
      scores_list.append(score)
    scores_list = np.concatenate(scores_list, axis=1)
    pickle.dump([[], [], scores_list], open('scores.pkg', 'wb'))

    '''
    coref_predictions = {}
    coref_evaluator = metrics.CorefEvaluator()

    scores_list = []
    eval_size = len(self.eval_data)

    for example_num, (tensorized_example, example) in enumerate(self.eval_data):
      if example_num % 100 == 0:
        print('%d / %d pairs evaluated' % (example_num, eval_size))
      ques_emb, trans_emb, label, is_training = tensorized_example

      feed_dict = {i:t for i,t in zip(self.input_tensors, tensorized_example)}
      score = session.run(self.predictions, feed_dict=feed_dict)

      scores_list.append([example['q_id'], example['v_id'], score])
    
    pickle.dump([[], [], scores_list], open('scores.pkg', 'wb'))
    '''
