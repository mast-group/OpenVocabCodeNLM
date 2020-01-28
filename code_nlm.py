# some structure based on  https://github.com/wpm/tfrnnlm/blob/master/tfrnnlm/rnn.py
#https://github.com/tensorflow/tensorflow/pull/2580/files#diff-083dd112b4600ecbaf63b2070951aad8


from __future__ import print_function

import ast

import time
from datetime import timedelta

import inspect
import math
import json
import os.path
import sys
import shutil
import heapq
import pygtrie as trie

from collections import deque
from itertools import chain
from operator import itemgetter

import numpy as np
import tensorflow as tf
import reader

# BPE imports
# import codecs
# from subword_nmt.apply_bpe import BPE, read_vocabulary



flags = tf.flags
# Path options 
flags.DEFINE_string("data_path", None, "Path to folder containing training/test data.")
flags.DEFINE_string("train_dir", None, "Output directory for saving the model.")

# Scenario options. Training is default so, no option for it.
flags.DEFINE_boolean("predict", False, "Set to True for computing predictability.")
flags.DEFINE_boolean("test", False, "Set to True for computing test perplexity.")
flags.DEFINE_boolean("dynamic_test", False, "Set to True for performing dynamic train-testing perplexity calculation (only one train epoch).")
flags.DEFINE_boolean("maintenance_test", False, "Set to True for performing maintenance train-testing perplexity simulation (only one train epoch).")
flags.DEFINE_boolean("completion", False, "Set to True to run code completion experiment.")
flags.DEFINE_boolean("maintenance_completion", False, "Set to True to run maintenance code completion experiment")
flags.DEFINE_boolean("dynamic", False, "Set to True to run dynamic code completion experiment.")

# Filename/path options
flags.DEFINE_string("train_filename", None, "The train file on which to train.")
flags.DEFINE_string("validation_filename", None, "The test file on which to run validation.")
flags.DEFINE_string("test_filename", None, "The test file on which to compute perplexity or predictability.")
flags.DEFINE_string("test_proj_filename", None, "The file that contains the test project name for each test instance.")
flags.DEFINE_string("identifier_map", None, "The file that contains information about which tokens are identifiers.")
flags.DEFINE_boolean("cache_ids", False, "Set to True to cache project identifiers during completion.")
# flags.DEFINE_string("BPE", None, "The file containing the BPE encoding.")
flags.DEFINE_string("subtoken_map", None, "Contains the mapping from heyristic subtokens to tokens.")

# flags.DEFINE_string("output_probs_file", "predictionProbabilities.txt", "The file to store output probabilities.")

# Network architecture/hyper-parameter options
flags.DEFINE_integer("num_layers", 1, "Number of Layers. Using a single layer is advised.")
flags.DEFINE_integer("hidden_size", 512, "Hidden size. Number of dimensions for the embeddings and RNN hidden state.")
flags.DEFINE_float("keep_prob", 0.5, "Keep probability = 1.0 - dropout probability.")
flags.DEFINE_integer("vocab_size", 25000, "Vocabulary size")
flags.DEFINE_boolean("gru", False, "Use a GRU cell. Must be set to True to use a GRU, otherwise an LSTM will be used.")
flags.DEFINE_integer("steps_per_checkpoint", 5000, "Number of steps for printing stats (validation is run) and checkpointing the model. Must be increased by 'a lot' for large training corpora.")
flags.DEFINE_integer("max_epoch", 30, "Max number training epochs to run.")
flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_integer("test_batch_size", 10, "Batch size during predictability test")
flags.DEFINE_integer("num_steps", 200, "Sequence length.")
flags.DEFINE_float("init_scale", 0.05, "Initialization scale.")
flags.DEFINE_float("learning_rate", 0.1, "Learning rate")
flags.DEFINE_float("max_grad_norm", 5.0, "Clip gradients to this norm")
flags.DEFINE_float("lr_decay", 0.5, "Learning rate decay. Default is 0.5 which halves the learning rate.")

# n-gram identifier cache options
flags.DEFINE_float("file_cache_weight", 0.2, "Weight of the file cache.")
flags.DEFINE_integer("cache_order", 6, "n-gram order for the identifier cache")

flags.DEFINE_integer("thresh", 0, "Threshold for vocabulary inclusion.")
flags.DEFINE_boolean("unk", True, "use -UNK- token to model OOV.")
flags.DEFINE_boolean("bidirectional", False, "Bidirectional model.")
flags.DEFINE_boolean("word_level_perplexity", False, "Convert to word level perplexity.")
flags.DEFINE_boolean("cross_entropy", False, "Print cross-entropy for validation/test instead of perplexity.")
flags.DEFINE_boolean("token_model", False, "Whether it is a token level model.")
flags.DEFINE_boolean("completion_unk_wrong", False, "Whether completing -UNK- should contribute in MRR. Set to "
                                                    "True for Allamanis et al. heuristic subtoken model.")
flags.DEFINE_boolean("verbose", False, "Verbose for completion.")


FLAGS = flags.FLAGS

def data_type():
  """
  Returns the TF floating point type used for operations.
  :return: The data type used (tf.float32)
  """
  return tf.float32

def get_gpu_config():
  gconfig = tf.ConfigProto()
  gconfig.gpu_options.per_process_gpu_memory_fraction = 0.975 # Don't take 100% of the memory
  gconfig.allow_soft_placement = True # Does not aggressively take all the GPU memory
  gconfig.gpu_options.allow_growth = True # Take more memory when necessary
  return gconfig

class NLM(object):

  def __init__(self, config):
    """
    Initializes the neural language model based on the specified configation.
    :param config: The configuration to be used for initialization.
    """
    self.num_layers = config.num_layers
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.hidden_size = hidden_size = config.hidden_size
    self.vocab_size = vocab_size = config.vocab_size
    #self.predictions_file = config.output_probs_file
    self.global_step = tf.Variable(0, trainable=False)

    with tf.name_scope("Parameters"):
      # Sets dropout and learning rate.
      self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
      self.keep_probability = tf.placeholder(tf.float32, name="keep_probability")

    with tf.name_scope("Input"):
      self.inputd = tf.placeholder(tf.int64, shape=(batch_size, None), name="inputd")
      self.targets = tf.placeholder(tf.int64, shape=(batch_size, None), name="targets")
      self.target_weights = tf.placeholder(tf.float32, shape=(batch_size, None), name="tgtweights")

    with tf.device("/cpu:0"):
      with tf.name_scope("Embedding"):
        # Initialize embeddings on the CPU and add dropout layer after embeddings.
        self.embedding = tf.Variable(tf.random_uniform((vocab_size, hidden_size), -config.init_scale, config.init_scale), dtype=data_type(), name="embedding")
        self.embedded_inputds = tf.nn.embedding_lookup(self.embedding, self.inputd, name="embedded_inputds")
        self.embedded_inputds = tf.nn.dropout(self.embedded_inputds, self.keep_probability)

    with tf.name_scope("RNN"):
      # Definitions for the different cells that can be used. Either lstm or GRU which will be wrapped with dropout.
      def lstm_cell():
        if 'reuse' in inspect.getargspec(tf.contrib.rnn.BasicLSTMCell.__init__).args:
          return tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
        else:
          return tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)
      def gru_cell():
        if 'reuse' in inspect.getargspec(tf.contrib.rnn.GRUCell.__init__).args:
          return tf.contrib.rnn.GRUCell(hidden_size, reuse=tf.get_variable_scope().reuse)
        else:
          return tf.contrib.rnn.GRUCell(hidden_size)
      def drop_cell():
        if FLAGS.gru:
          return tf.contrib.rnn.DropoutWrapper(gru_cell(), output_keep_prob=self.keep_probability)
        else:
          return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=self.keep_probability)

      # Allows multiple layers to be used. Not advised though.
      rnn_layers = tf.contrib.rnn.MultiRNNCell([drop_cell() for _ in range(self.num_layers)], state_is_tuple=True)
      # Initialize the state to zero.
      self.reset_state = rnn_layers.zero_state(batch_size, data_type())
      self.outputs, self.next_state = tf.nn.dynamic_rnn(rnn_layers, self.embedded_inputds, time_major=False,
                                                        initial_state=self.reset_state)

    with tf.name_scope("Cost"):
      # Output and loss function calculation
      self.output = tf.reshape(tf.concat(axis=0, values=self.outputs), [-1, hidden_size])
      self.softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size], dtype=data_type())
      self.softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
      self.logits = tf.matmul(self.output, self.softmax_w) + self.softmax_b
      self.loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [self.logits], [tf.reshape(self.targets, [-1])], [tf.reshape(self.target_weights, [-1])])
      self.cost = tf.div(tf.reduce_sum(self.loss), batch_size, name="cost")
      self.final_state = self.next_state

      self.norm_logits = tf.nn.softmax(self.logits)

    with tf.name_scope("Train"):
      self.iteration = tf.Variable(0, dtype=data_type(), name="iteration", trainable=False)
      tvars = tf.trainable_variables()
      self.gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                                 config.max_grad_norm, name="clip_gradients")
      optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.train_step = optimizer.apply_gradients(zip(self.gradients, tvars), name="train_step",
                                                  global_step=self.global_step)
      self.validation_perplexity = tf.Variable(dtype=data_type(), initial_value=float("inf"),
                                               trainable=False, name="validation_perplexity")
      tf.summary.scalar(self.validation_perplexity.op.name, self.validation_perplexity)
      self.training_epoch_perplexity = tf.Variable(dtype=data_type(), initial_value=float("inf"),
                                                   trainable=False, name="training_epoch_perplexity")
      tf.summary.scalar(self.training_epoch_perplexity.op.name, self.training_epoch_perplexity)

      self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
      self.initialize = tf.initialize_all_variables()
      self.summary = tf.summary.merge_all()


  def get_parameter_count(self, debug=False):
    """
    Counts the number of parameters required by the model.
    :param debug: Whether debugging information should be printed.
    :return: Returns the number of parameters required for the model.
    """
    params = tf.trainable_variables()
    total_parameters = 0
    for variable in params:
      shape = variable.get_shape()
      variable_parameters = 1
      for dim in shape:
        variable_parameters *= dim.value
      if debug:
          print(variable)
          print(shape + "\t" + str(len(shape)) + "\t" + str(variable_parameters))
      total_parameters += variable_parameters
    return total_parameters

  @property
  def reset_state(self):
      return self._reset_state

  @reset_state.setter
  def reset_state(self, x):
      self._reset_state = x

  @property
  def cost(self):
      return self._cost

  @cost.setter
  def cost(self, y):
      self._cost = y

  @property
  def final_state(self):
      return self._final_state

  @final_state.setter
  def final_state(self, z):
      self._final_state = z

  @property
  def learning_rate(self):
      return self._lr

  @learning_rate.setter
  def learning_rate(self, l):
      self._lr = l

  @property
  def input(self):
      return self.data


  def train(self, session, config, train_data, exit_criteria, valid_data, summary_dir):
    """
    Trains the NLM with the specified configuration, training, and validation data.
    Training is terminated when the specified criteria have been satisfied.
    :param session: The TF session in which operations should be run.
    :param config: The configuration to be used for the model.
    :param train_data: The dataset instance to use for training.
    :param exit_criteria: The training termination criteria.
    :param valid_data: The dataset instance to use for validation.
    :param summary_dir: Directory in which summary information will be stored.
    """
    summary_writer = tf.summary.FileWriter(summary_dir, session.graph)
    previous_valid_log_ppx = []
    nglobal_steps = 0
    epoch = 1
    new_learning_rate = config.learning_rate
    state = session.run(self.reset_state)

    try:
      while True:
        epoch_log_perp_unnorm = epoch_total_weights = 0.0
        print("Epoch %d Learning rate %0.3f" % (epoch, new_learning_rate))
        epoch_start_time = time.time()
        # Runs each training step. A step is processing a minibatch of context-target pairs.
        for step, (context, target, target_weights) in enumerate(
                train_data.batch_producer_memory_efficient(self.batch_size, self.num_steps)):
          # Every steps_per_checkpoint steps run validation and print perplexity/entropy.
          if step % FLAGS.steps_per_checkpoint == 0:
            print('Train steps:', step)
            if step >0:
              validation_perplexity = self.test(session, config, valid_data)
              validation_log_perplexity = math.log(validation_perplexity)
              print("global_steps %d learning_rate %.4f valid_perplexity %.2f" % (nglobal_steps, new_learning_rate, validation_perplexity))
            sys.stdout.flush()
          feed_dict = {self.inputd: context,
                     self.targets: target,
                     self.target_weights: target_weights,
                     self.learning_rate: new_learning_rate,
                     self.keep_probability: config.keep_prob
          }
          if FLAGS.gru:
            for i, h in enumerate(self.reset_state):
              feed_dict[h] = state[i]
          else: # LSTM cell
            for i, (c, h) in enumerate(self.reset_state):
              feed_dict[c] = state[i].c
              feed_dict[h] = state[i].h
          # Run the actual training step.
          _, cost, state, loss, iteration = session.run([self.train_step, self.cost, self.next_state, self.loss, self.iteration], feed_dict)
          nglobal_steps += 1
          # Add step loss and weights to the total.
          epoch_log_perp_unnorm += np.sum(loss)
          epoch_total_weights += np.sum(sum(target_weights))
          # epoch_total_weights += np.sum(sum(sub_target_weights))
        train_log_perplexity = epoch_log_perp_unnorm / epoch_total_weights
        train_perplexity = math.exp(train_log_perplexity) if train_log_perplexity < 300 else float("inf")

        validation_perplexity = self.test(session, config, valid_data)
        validation_log_perplexity = math.log(validation_perplexity)
        # Checkpoint and save the model.
        checkpoint_path = os.path.join(FLAGS.train_dir, "lm.ckpt.epoch" + str(epoch))
        self.saver.save(session, checkpoint_path, global_step=self.global_step)

        train_perplexity_summary = tf.Summary()
        valid_perplexity_summary = tf.Summary()

        train_perplexity_summary.value.add(tag="train_log_ppx", simple_value=train_log_perplexity)
        train_perplexity_summary.value.add(tag="train_ppx", simple_value=train_perplexity)
        summary_writer.add_summary(train_perplexity_summary, nglobal_steps)
        valid_perplexity_summary.value.add(tag="valid_log_ppx", simple_value=validation_log_perplexity)
        valid_perplexity_summary.value.add(tag="valid_ppx", simple_value=validation_perplexity)
        summary_writer.add_summary(valid_perplexity_summary, nglobal_steps)
        # Convert epoch time in minutes and print info on screen.
        epoch_time = (time.time() - epoch_start_time) * 1.0 / 60
        print("END EPOCH %d global_steps %d learning_rate %.4f time(mins) %.4f train_perplexity %.2f valid_perplexity %.2f" %
              (epoch, nglobal_steps, new_learning_rate, epoch_time, train_perplexity, validation_perplexity))
        sys.stdout.flush()

        if exit_criteria.max_epochs is not None and epoch > exit_criteria.max_epochs:
          raise StopTrainingException()

        # Decrease learning rate if valid ppx does not decrease
        if len(previous_valid_log_ppx) > 1 and validation_log_perplexity >= previous_valid_log_ppx[-1]:
          new_learning_rate = new_learning_rate * config.lr_decay

        # If validation perplexity has not improved over the last 5 epochs, stop training
        if new_learning_rate == 0.0 or (len(previous_valid_log_ppx) > 4 and validation_log_perplexity > max(previous_valid_log_ppx[-5:])):
          raise StopTrainingException()

        previous_valid_log_ppx.append(validation_log_perplexity)
        epoch += 1
    except (StopTrainingException, KeyboardInterrupt):
      print("Finished training ........")

  def test(self, session, config, test_data, ignore_padding=False):
    """
    Tests the NLM with the specified configuration and test data.
    :param session: The TF session in which operations should be run.
    :param config: The configuration to be used for the model.
    :param test_data:
    :param ignore_padding:
    :return:
    """
    log_perp_unnorm, total_size = 0.0, 0.0
    batch_number = -1
    state = session.run(self.reset_state)

    for step, (context, target, target_weights, sub_target_weights) in enumerate(
          test_data.batch_producer(self.batch_size, self.num_steps, True)):
      batch_number += 1
      feed_dict = {
          self.inputd: context,
          self.targets: target,
          self.target_weights: target_weights,
          self.keep_probability: 1.0 # No dropout should be used for the test!
      }
      if FLAGS.gru:
        for i, h in enumerate(self.reset_state):
          feed_dict[h] = state[i]
      else:
        for i, (c, h) in enumerate(self.reset_state):
          feed_dict[c] = state[i].c
          feed_dict[h] = state[i].h
      # norm_logits, loss, cost, state = session.run([self.norm_logits, self.loss, self.cost, self.next_state], feed_dict)
      loss, cost, state = session.run([self.loss, self.cost, self.next_state], feed_dict)

      if FLAGS.token_model:
        targets = [t for tar in target for t in tar]
        voc_size = 10500000
        loss = [-math.log(1.0/voc_size, 2) if t == self.train_vocab["-UNK-"] else l
                for l,t in zip(loss, targets) ]

      log_perp_unnorm += np.sum(loss)

      if FLAGS.word_level_perplexity:
        total_size += np.sum(sum(sub_target_weights))
      else:
        total_size += np.sum(sum(target_weights))

    if ignore_padding:
      paddings = 0
      for tok_loss, weight in zip(loss, chain.from_iterable(zip(*target_weights))):
        if weight == 0:
          log_perp_unnorm -= tok_loss
          paddings += 1

    total_size += 1e-12
    log_ppx = log_perp_unnorm / total_size
    ppx = math.exp(float(log_ppx)) if log_ppx < 300 else float("inf")
    if FLAGS.cross_entropy:
      return log_ppx
    return ppx

  def dynamic_train_test_file(self, test_lines, train_vocab, train_vocab_rev, test_projects, config, output_path, session):
    """
    Tests the NLM on the specified test dataset but also updates its parameters by training on each file
    after computing its per token entropy.
    The model is restored back to the global model after testing has been completed for a project.
    This procedure adapts the model to each project resulting in better performance but also prevents the model from
    forgetting information contained in the global model after testing on a plethora of projects.
    The per token entropy/perplexity will be calculated and shown on the screen.
    This mode should always be run with batch_size=1.
    :param test_lines:
    :param train_vocab: The word to id mapping.
    :param train_vocab_rev: The id to word mapping.
    :param test_projects: Names of the projects for each test file instance.
    :param config:  The configuration to be used for the model.
    :param output_path:
    :param session: The TF session in which operations should be run.
    :return: Average loss per file and not per token.
    """
    config.batch_size = 1
    ctr = 0
    nglobal_steps = 0
    state = None
    new_learning_rate = config.learning_rate
    losses = []
    lengths = []

    last_test_project = None
    for test_line, test_project in zip(test_lines, test_projects):
      ctr += 1
      if test_project != last_test_project and last_test_project is not None:
        # New test project so restore the model back to the global one.
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
          self.saver.restore(session, ckpt.model_checkpoint_path)
      last_test_project = test_project

      # Get the ids for this test instance and calculate entropy/perplexity.
      test_line = test_line.replace("\n", (" %s" % "-eod-"))
      ids = [train_vocab[word] if word in train_vocab else train_vocab['-UNK-'] for word in test_line.split(' ')]
      test_dataset = reader.dataset(ids, train_vocab, train_vocab_rev)
      test_loss = self.test(session, config, test_dataset, True)
      if FLAGS.cross_entropy:
        print('line cross_entropy:', test_loss)
      else:
        print('line perplexity:', test_loss)
      sys.stdout.flush()
      losses.append(test_loss)

      # Train.
      state = session.run(self.reset_state)
      try:
        epoch_log_perp_unnorm = epoch_total_weights = 0.0
        epoch_sub_total_weights = 0.0
        # Train on each batch to adapt the model to the new information available.
        for step, (context, target, target_weights, sub_target_weights) in enumerate(
                test_dataset.batch_producer(self.batch_size, self.num_steps)):
          feed_dict = {self.inputd: context,
                       self.targets: target,
                       self.target_weights: target_weights,
                       self.learning_rate: new_learning_rate,
                       self.keep_probability: config.keep_prob
          }
          if FLAGS.gru:
              for i, h in enumerate(self.reset_state):
                feed_dict[h] = state[i]
          else: # LSTM
            for i, (c, h) in enumerate(self.reset_state):
              feed_dict[c] = state[i].c
              feed_dict[h] = state[i].h
          _, cost, state, loss, iteration = session.run([self.train_step, self.cost, self.next_state, self.loss, self.iteration], feed_dict)
          nglobal_steps += 1
          epoch_log_perp_unnorm += np.sum(loss)
          epoch_total_weights += np.sum(sum(target_weights))
          epoch_sub_total_weights += np.sum(sum(sub_target_weights))
        train_log_perplexity = epoch_log_perp_unnorm / epoch_total_weights
        train_perplexity = math.exp(train_log_perplexity) if train_log_perplexity < 300 else float("inf")
        print(train_perplexity)
        lengths.append(int(round(epoch_sub_total_weights, 0)))
      except (StopTrainingException, KeyboardInterrupt):
        print("Finished training ........")

    total_len = float(sum(lengths))
    len_weights = [length / total_len for length in lengths]
    if FLAGS.cross_entropy:
        print('Per token entropy:', sum([perp * weight for perp, weight in zip(losses, len_weights)]))
    else:
        print('Per token perplexity:', sum([perp * weight for perp, weight in zip(losses, len_weights)]))
    return sum(losses)/ctr

  def dynamic_train_test(self, test_lines, train_vocab, train_vocab_rev, test_projects, config, output_path, session):
    """
    Tests the NLM on the specified test dataset but also updates its parameters by training on each batch
    after computing its per token entropy first.
    The model is restored back to the global model after testing has been completed for a project.
    This procedure adapts the model to each project resulting in better performance but also prevents the model from
    forgetting information contained in the global model after testing on a plethora of projects.
    The per token entropy/perplexity will be calculated and shown on the screen.
    This mode should always be run with batch_size=1.
    :param test_lines:
    :param train_vocab:
    :param train_vocab_rev:
    :param test_projects:
    :param config: The configuration to be used for the model.
    :param output_path:
    :param session:
    :return: Average loss per file and not per token.
    """
    config.batch_size = 1
    ctr = 0
    nglobal_steps = 0
    state = None
    new_learning_rate = config.learning_rate
    losses = []
    lengths = []

    last_test_project = None
    for test_line, test_project in zip(test_lines, test_projects):
      ctr += 1
      if ctr % 100 == 0:
        print("\t %d lines" % ctr)
        total_len = float(sum(lengths))
        len_weights = [length / total_len for length in lengths]
        print('Current per token :', sum([perp * weight for perp, weight in zip(losses, len_weights)]))
        print(sum(losses) / ctr)

      file_log_perp_unnorm = file_total_weights = 0.0
      file_sub_total_weights = 0.0

      if test_project != last_test_project and last_test_project is not None:
        # New test project so restore the model
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
          self.saver.restore(session, ckpt.model_checkpoint_path)
      last_test_project = test_project

      test_line = test_line.replace("\n", (" %s" % "-eod-"))
      ids = [train_vocab[word] if word in train_vocab else train_vocab['-UNK-'] for word in test_line.split(' ')]
      test_dataset = reader.dataset(ids, train_vocab, train_vocab_rev)

      # Test, Train
      state = session.run(self.reset_state)
      try:
        epoch_log_perp_unnorm = epoch_total_weights = 0.0
        epoch_sub_total_weights = 0.0
        for step, (context, target, target_weights, sub_target_weights) in enumerate(
                test_dataset.batch_producer(self.batch_size, self.num_steps)):
          feed_dict = {self.inputd: context,
                       self.targets: target,
                       self.target_weights: target_weights,
                       self.keep_probability: 1.0
                      }
          if FLAGS.gru:
            for i, h in enumerate(self.reset_state):
              feed_dict[h] = state[i]
          else: # LSTM
            for i, (c, h) in enumerate(self.reset_state):
              feed_dict[c] = state[i].c
              feed_dict[h] = state[i].h
          loss, cost, state = session.run([self.loss, self.cost, self.next_state], feed_dict)
          if FLAGS.token_model:
            targets = [t for tar in target for t in tar]
            loss = [-math.log(1.0/len(self.train_vocab), 2) if t == self.train_vocab["-UNK-"] else l
                    for l,t in zip(loss, targets) ]
          file_log_perp_unnorm += np.sum(loss)
          file_total_weights += np.sum(sum(target_weights))
          file_sub_total_weights += np.sum(sum(sub_target_weights))
          if True:
            for tok_loss, weight in zip(loss, chain.from_iterable(zip(*target_weights))):
              if weight == 0:
                file_log_perp_unnorm -= tok_loss

          feed_dict = {self.inputd: context,
                       self.targets: target,
                       self.target_weights: target_weights,
                       self.learning_rate: new_learning_rate,
                       self.keep_probability: config.keep_prob
          }
          if FLAGS.gru:
            for i, h in enumerate(self.reset_state):
              feed_dict[h] = state[i]
          else: # LSTM
            for i, (c, h) in enumerate(self.reset_state):
              feed_dict[c] = state[i].c
              feed_dict[h] = state[i].h
          _, cost, state, loss, iteration = session.run([self.train_step, self.cost, self.next_state, self.loss, self.iteration], feed_dict)
          nglobal_steps += 1
          epoch_log_perp_unnorm += np.sum(loss)
          epoch_total_weights += np.sum(sum(target_weights))
          epoch_sub_total_weights += np.sum(sum(sub_target_weights))

        train_log_perplexity = epoch_log_perp_unnorm / epoch_total_weights
        train_perplexity = math.exp(train_log_perplexity) if train_log_perplexity < 300 else float("inf")
        print(train_perplexity)
        lengths.append(int(round(epoch_sub_total_weights, 0)))

        if FLAGS.word_level_perplexity:
          test_loss = file_log_perp_unnorm / file_sub_total_weights
        else:
          test_loss = file_log_perp_unnorm / file_total_weights

        if FLAGS.cross_entropy:
          print('line cross_entropy:', test_loss)
        else:
          print('line perplexity:', test_loss)
        sys.stdout.flush()
        losses.append(test_loss)

      except (StopTrainingException, KeyboardInterrupt):
        print("Finished training ........")

    total_len = float(sum(lengths))
    len_weights = [length / total_len for length in lengths]
    if FLAGS.cross_entropy:
        print('Per token entropy:', sum([perp * weight for perp, weight in zip(losses, len_weights)]))
    else:
        print('Per token perplexity:', sum([perp * weight for perp, weight in zip(losses, len_weights)]))
    return sum(losses)/ctr

  def maintenance_test(self, session, config, test_lines, test_projects, train_vocab, train_vocab_rev):
    """
    Simulates code maintenance scenario.
    For each file in a project the model is first adapted on the rest of the files.
    The model is also adapted on encountered sequences of the test file.
    The test_projects argument informs the model about which instances belong to which project.
    Note: Always use batch_size=1. RNN num_steps=20 is a good value.
    :param session: The TF session in which operations will be run.
    :param config: The configuration to be used for the model.
    :param test_lines: A list containing each test instance as a separate entry. Instances from the same project should be consecutive.
    :param test_projects: To which project does each instance belong to. Should be a list of strings.
    :param train_vocab: The word to id mapping.
    :param train_vocab_rev: The id to word mapping.
    :return:
    """
    # If checkpoint does not exist throw an exception. A global model must have been pretrained.
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if not tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
      raise Exception('Checkpoint does not exist!')

    # Some initializations.
    new_learning_rate = config.learning_rate
    test_losses = []
    test_losses_sum = 0.0
    lengths = []
    ctr = 0

    # First compute which files belong in each project.
    project_sizes = []
    last_project_name = ''
    project_file_size = 0
    for test_project in test_projects:
      if test_project != last_project_name:
        if last_project_name != '':
          project_sizes.append(project_file_size)
          project_file_size = 0
      else:
        project_file_size += 1
      last_project_name = test_project
    print(project_sizes)
    print(sum(project_sizes))
    print()

    # Now distribute test lines based on project size.
    project_test_lines = []
    files_distributed = 0
    for project_file_size in project_sizes:
      project_test_lines.append(test_lines[files_distributed : files_distributed + project_file_size])
      files_distributed += project_file_size

    partitions = 20 # Default number of partitions
    for proj_id, test_lines in enumerate(project_test_lines):
      print(len(test_lines))
      large_project = len(test_lines) > 200
      if len(test_lines) > 2000: # Really big projects can have speed benefits from more partitions.
        partitions = 50
      else:
        partitions = 20

      if large_project:
        # The project is very large so partition it in partition_size parts.
        # For each of the 20/50 parts train a model on all the parts excluding itself.
        partition_size = len(test_lines) / partitions
        for partition in range(partitions):
          partition_words = []
          for i, line in enumerate(test_lines):
            if i / partition_size != partition and not (i / partition_size == partitions + 1 and partition == partitions):
              partition_words.extend([word for word in line.split(' ')])
          # If checkpoint does not exist throw an exception. A global model should have been trained...
          ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
          if not tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            raise Exception('Checkpoint for global model does not exist!')
          self.saver.restore(session, ckpt.model_checkpoint_path)

          # Where to save this partition.
          partition_path = os.path.join(FLAGS.train_dir, "partition%d" % partition)
          if os.path.isdir(partition_path): shutil.rmtree(partition_path)

          partition_ids = [train_vocab[word] if word in train_vocab else train_vocab['-UNK-'] for word in partition_words]
          partition_dataset = reader.dataset(partition_ids, train_vocab, train_vocab_rev)

          # Reset the LSTM state to zeros and train
          state = session.run(self.reset_state)
          try:
            epoch_log_perp_unnorm = epoch_total_weights = 0.0
            epoch_sub_total_weights = 0.0
            for step, (context, target, target_weights, sub_target_weights) in enumerate(
                    partition_dataset.batch_producer(self.batch_size, self.num_steps)):
              feed_dict = {self.inputd: context,
                           self.targets: target,
                           self.target_weights: target_weights,
                           self.learning_rate: new_learning_rate,
                           self.keep_probability: config.keep_prob
                           }
              if FLAGS.gru:
                for i, h in enumerate(self.reset_state):
                  feed_dict[h] = state[i]
              else:
                for i, (c, h) in enumerate(self.reset_state):
                  feed_dict[c] = state[i].c
                  feed_dict[h] = state[i].h
              _, cost, state, loss, iteration = session.run(
                [self.train_step, self.cost, self.next_state, self.loss, self.iteration], feed_dict)
              epoch_log_perp_unnorm += np.sum(loss)
              epoch_total_weights += np.sum(sum(target_weights))
              epoch_sub_total_weights += np.sum(sum(sub_target_weights))

            train_log_perplexity = epoch_log_perp_unnorm / epoch_total_weights
            train_perplexity = math.exp(train_log_perplexity) if train_log_perplexity < 300 else float("inf")
            print(train_perplexity)
            checkpoint_path = os.path.join(partition_path, "model")
            self.saver.save(session, checkpoint_path, global_step=self.global_step)
          except (StopTrainingException, KeyboardInterrupt):
            print("Finished training ........")

      for lines_done, test_line in enumerate(test_lines):
        ctr += 1
        # Restore the global model
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
          self.saver.restore(session, ckpt.model_checkpoint_path)

        if large_project:
          # Load pretrained model and only train on the rest of the files from this partition
          partition = lines_done / partition_size
          partition = min(partitions - 1, partition) # last partition can be bigger
          partition_path =  os.path.join(FLAGS.train_dir, "partition%d" % partition)
          part_ckpt = tf.train.get_checkpoint_state(partition_path)
          self.saver.restore(session, part_ckpt.model_checkpoint_path)

          train_words = []
          if partition < partitions - 1:
            partition_lines = zip(range(partition * partition_size, (partition + 1) * partition_size),
                                  test_lines[partition * partition_size : (partition + 1) * partition_size])
          else:
            partition_lines = zip(range(partition * partition_size, len(test_lines)),
                                        test_lines[partition * partition_size : (partition + 1) * partition_size])
          for i, line in partition_lines:
            if i != (lines_done % partition_size):
              train_words.extend([word for word in line.split(' ')])
        else:
          # Now for each file in the current test project use the rest as train data
          train_words = []
          for i, line in enumerate(test_lines):
            if i != lines_done:
              train_words.extend([word for word in line.split(' ')])

        # Convert the train data words to ids
        ids = [train_vocab[word] if word in train_vocab else train_vocab['-UNK-'] for word in train_words]
        train_dataset = reader.dataset(ids, train_vocab, train_vocab_rev)

        # Reset the LSTM state to zeros and train
        state = session.run(self.reset_state)
        try:
          epoch_log_perp_unnorm = epoch_total_weights = 0.0
          epoch_sub_total_weights = 0.0
          for step, (context, target, target_weights, sub_target_weights) in enumerate(
                  train_dataset.batch_producer(self.batch_size, self.num_steps)):
            feed_dict = {self.inputd: context,
                          self.targets: target,
                          self.target_weights: target_weights,
                          self.learning_rate: new_learning_rate,
                          self.keep_probability: config.keep_prob
            }
            if FLAGS.gru:
              for i, h in enumerate(self.reset_state):
                feed_dict[h] = state[i]
            else:
              for i, (c, h) in enumerate(self.reset_state):
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h
                # print("state number " + str(i))
            _, cost, state, loss, iteration = session.run([self.train_step, self.cost, self.next_state, self.loss, self.iteration], feed_dict)
            epoch_log_perp_unnorm += np.sum(loss)
            epoch_total_weights += np.sum(sum(target_weights))
            epoch_sub_total_weights += np.sum(sum(sub_target_weights))
          train_log_perplexity = epoch_log_perp_unnorm / epoch_total_weights
          train_perplexity = math.exp(train_log_perplexity) if train_log_perplexity < 300 else float("inf")
          # lengths.append(int(round(epoch_sub_total_weights, 0)))

          # Training done. Now test on test file
          ids = [train_vocab[word] if word in train_vocab else train_vocab['-UNK-']
                  for word in test_lines[lines_done].split(' ')]
          # Test on each sequence of tokens and then train on it until the file is done
          subtokens_done = 0
          tokens_done = 0
          instance_losses = []
          # Reset the LSTM state to zeros and train
          state = session.run(self.reset_state)
          test_state = session.run(self.reset_state)
          while subtokens_done + config.num_steps < len(ids):
            step_end = subtokens_done + config.num_steps
            unfinished_token = train_vocab_rev[ids[step_end]].endswith('@@')
            while unfinished_token:
              step_end += 1
              unfinished_token = train_vocab_rev[ids[step_end]].endswith('@@')

            try:
              context = ids[subtokens_done : step_end]
              target = ids[subtokens_done + 1 : step_end + 1]
              target_weights = [1] * len(context)
              for id in context:
                if not train_vocab_rev[id].endswith('@@'):
                  tokens_done += 1

              feed_dict = {self.inputd: np.tile(context, (self.batch_size, 1)),
                           self.targets: np.tile(target, (self.batch_size, 1)),
                           self.target_weights: np.tile(target_weights, (self.batch_size, 1)),
                           self.keep_probability: 1.0
                          }
              if FLAGS.gru:
                for i, h in enumerate(self.reset_state):
                  feed_dict[h] = test_state[i]
              else:
                for i, (c, h) in enumerate(self.reset_state):
                  feed_dict[c] = test_state[i].c
                  feed_dict[h] = test_state[i].h
              cost, test_state, loss = session.run([self.cost, self.next_state, self.loss], feed_dict)
              if tokens_done > 0:
                instance_losses.append((tokens_done, len(context), sum(loss)/tokens_done))
              feed_dict = {self.inputd: np.tile(context, (self.batch_size, 1)),
                           self.targets: np.tile(target, (self.batch_size, 1)),
                           self.target_weights: np.tile(target_weights, (self.batch_size, 1)),
                           self.learning_rate: new_learning_rate,
                           self.keep_probability: config.keep_prob
                          }
              if FLAGS.gru:
                for i, h in enumerate(self.reset_state):
                  feed_dict[h] = state[i]
              else: # LSTM
                for i, (c, h) in enumerate(self.reset_state):
                  feed_dict[c] = state[i].c
                  feed_dict[h] = state[i].h
              _, cost, state, loss, iteration = session.run(
                [self.train_step, self.cost, self.next_state, self.loss, self.iteration], feed_dict)
            except (StopTrainingException, KeyboardInterrupt):
              print("Finished training ........")
            subtokens_done = step_end

          # Test and train on leftover part of the sequence, which has length < step_size.
          try:
            tokens_done = 0
            context = ids[subtokens_done:-1]
            target = ids[subtokens_done + 1:]
            target_weights = [1] * len(context)
            if len(context) == 0 or len(target) == 0: # if there are no actual leftovers stop.
              continue

            for id in context:
              if not train_vocab_rev[id].endswith('@@'):
                tokens_done += 1
            feed_dict = {self.inputd: np.tile(context, (self.batch_size, 1)),
                         self.targets: np.tile(target, (self.batch_size, 1)),
                         self.target_weights: np.tile(target_weights, (self.batch_size, 1)),
                         self.keep_probability: 1.0
                        }
            if FLAGS.gru:
              for i, h in enumerate(self.reset_state):
                feed_dict[h] = test_state[i]
            else:
              for i, (c, h) in enumerate(self.reset_state):
                feed_dict[c] = test_state[i].c
                feed_dict[h] = test_state[i].h
            cost, test_state, loss = session.run([self.cost, self.next_state, self.loss], feed_dict)
            if tokens_done > 0:
              instance_losses.append((tokens_done, len(context), sum(loss)/tokens_done))
            feed_dict = {self.inputd: np.tile(context, (self.batch_size, 1)),
                         self.targets: np.tile(target, (self.batch_size, 1)),
                         self.target_weights: np.tile(target_weights, (self.batch_size, 1)),
                         self.learning_rate: new_learning_rate,
                         self.keep_probability: config.keep_prob
                        }
            if FLAGS.gru:
              for i, h in enumerate(self.reset_state):
                feed_dict[h] = state[i]
            else: # LSTM
              for i, (c, h) in enumerate(self.reset_state):
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h
                # print("state number " + str(i))
            _, cost, state, loss, iteration = session.run(
              [self.train_step, self.cost, self.next_state, self.loss, self.iteration], feed_dict)
          except (StopTrainingException, KeyboardInterrupt):
            print("Finished training ........")

          tokens_done = float(sum([toks for toks, _, _ in instance_losses]))
          lengths.append(tokens_done)
          weighted_losses = [loss * toks / tokens_done for toks, _, loss in instance_losses]
          test_loss = sum(weighted_losses)
          test_losses_sum += test_loss
          print('line cross_entropy:', test_loss, '--- current average:', test_losses_sum / ctr)
          # if FLAGS.cross_entropy:
          #   print('line cross_entropy:', test_loss, '--- current average:', test_losses_sum / ctr)
          # else:
          #   print('line perplexity:', test_loss)
          sys.stdout.flush()
          test_losses.append(test_loss)
        except (StopTrainingException, KeyboardInterrupt):
          print("Finished training ........")

      print("Projects Done:", proj_id + 1)

    total_len = float(sum(lengths))
    len_weights = [length / total_len for length in lengths]
    print('Is it correct perplexity:', sum([perp * weight for perp, weight in zip(test_losses, len_weights)]))
    return test_losses_sum / ctr

  def completion(self, session, config, test_dataset, test_projects, beam_size, dynamic=False, id_map=None, cache_ids=False, token_map=None):
    """
    Runs code the code completion scenario. Dynamic update can be performed but by default is turned off.
    :param session: The TF session in which operations should be run.
    :param config: The configuration to be used for the model.
    :param test_dataset:
    :param test_projects: To which project does each instance belong to. Should be a list of strings.
    :param beam_size: The size of the beam to be used by the search algorithm.
    :param dynamic: Whether dynamic adaptation should be performed.
    :return:
    """
    mrr = 0.0
    id_mrr = 0.0
    id_acc1 = 0.0
    id_acc3 = 0.0
    id_acc5 = 0.0
    id_acc10 = 0.0

    ids_in_cache = 0.0
    ids_in_project_cache = 0.0
    context_history_in_ngram_cache = 0.0
    context_history_in_ngram_project_cache = 0.0

    satisfaction_prob = 0.8
    top_needed = 10
    verbose = FLAGS.verbose
    last_test_project = None
    train_every = config.num_steps
    tokens_done = 0
    files_done = 0
    identifiers = 0
    file_identifiers = 0
    state = session.run(self.reset_state)
    
    context_size = FLAGS.cache_order - 1
    context_history = deque([None] * context_size, context_size)
    ngram_cache = dict()
    ngram_project_cache = dict()
    # project_context_history = []
    id_cache = trie.CharTrie()
    project_id_cache = trie.CharTrie()
    CACHE_WEIGHT = FLAGS.file_cache_weight
    PROJECT_CACHE_WEIGHT = FLAGS.file_cache_weight * 0.5
    SKIP_CACHE_PROB_THRESHOLD = 0.0

    raw_data = test_dataset.data  # is just one long array
    data_len = len(raw_data)
    print('Data Length:', data_len)
    data_covered = 0
    end_file_id = test_dataset.vocab["-eod-"]
    start_index = 0
    file_start_index = 0
    while data_covered < data_len:
      # Stop when 1000000 test tokens have been scored.
      if tokens_done > 1000000:
        break

      # Create minibatches for the next file
      while raw_data[data_covered] != end_file_id:
        data_covered += 1
      data_covered += 1 # eod symbol
      file_identifiers = 0
      
      # Reset identifier cache for each file.
      if cache_ids:
          id_cache.clear()
          ids_in_cache = 0.0
          ids_in_project_cache = 0.0
          context_history = deque([None] * context_size, context_size)
          ngram_cache = dict()
          context_history_in_ngram_cache = 0.0
          context_history_in_ngram_project_cache = 0.0

      if dynamic:
        test_project = test_projects[files_done]
        if test_project != last_test_project:
          # New test project so restore the model
          ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
          if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver.restore(session, ckpt.model_checkpoint_path)
          
          # Reset the project's cache of identifiers if one is used.
          if cache_ids:
            print('clearing project cache')
            ids_in_project_cache = 0.0
            project_id_cache.clear()
            ngram_project_cache = dict()
        last_test_project = test_project

      file_data = raw_data[file_start_index:data_covered]
      file_start_index = data_covered
      print('Completion Length:', len(file_data))

      if not id_map is None: file_ids = id_map[files_done] + [0]
      else: file_ids = [0] * (len(file_data) - 1)
      
      if not token_map is None: file_tokens = token_map[files_done]
      else: file_tokens = [0] * (len(file_data) - 1)
      
      tokens_before = deque([None, test_dataset.rev_vocab[file_data[0]]], 2)

      state = session.run(self.reset_state)
      remember_state = state
      train_state = session.run(self.reset_state)
      in_token = False

      correct_token = ''
      train_start = 0
      train_end = 0
      # to_add = []
      for subtoken_id, context_target_is_id in enumerate(zip(file_data[:-1], file_data[1:], file_ids)):
        context, target, is_id = context_target_is_id
        # print(test_dataset.rev_vocab[context], test_dataset.rev_vocab[target], is_id)
        train_end += 1

        # to_add.append(test_dataset.rev_vocab[context])
        if (subtoken_id - 1) - train_start > train_every and not test_dataset.rev_vocab[context].endswith('@@'):
          # train
          if dynamic:
            tr_context, tr_target, tr_target_weights = file_data[train_start : subtoken_id - 1], \
                                                       file_data[train_start + 1 : subtoken_id], \
                                                       [1.0] * ((subtoken_id - 1) - train_start)
            # start_train_at = subtoken_id
            feed_dict = {self.inputd: np.tile(tr_context, (self.batch_size, 1)),
                          self.targets: np.tile(tr_target, (self.batch_size, 1)),
                          self.target_weights: np.tile(tr_target_weights, (self.batch_size, 1)),
                          self.learning_rate: config.learning_rate,
                          self.keep_probability: config.keep_prob
                        }
            if FLAGS.gru:
              for i, h in enumerate(self.reset_state):
                feed_dict[h] = train_state[i]
            else: # LSTM
              for i, (c, h) in enumerate(self.reset_state):
                feed_dict[c] = train_state[i].c
                feed_dict[h] = train_state[i].h
            _, cost, train_state, loss, iteration = session.run(
              [self.train_step, self.cost, self.next_state, self.loss, self.iteration], feed_dict)
          train_start = train_end

        feed_dict = {self.inputd: np.array([[context]] * self.batch_size),
                     self.targets: np.array([[target]] * self.batch_size),
                     self.target_weights: np.array([[1.0]] * self.batch_size),
                     self.keep_probability: 1.0
                    }
        if FLAGS.gru:
          for i, h in enumerate(self.reset_state):
            feed_dict[h] = state[i]
        else: # LSTM
          for i, (c, h) in enumerate(self.reset_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        norm_logits, loss, cost, state = session.run([self.norm_logits, self.loss, self.cost, self.next_state], feed_dict)

        correct_word = test_dataset.rev_vocab[target]
        if verbose: print('Correct:', correct_word)
        # if train_start > 0 and is_id:
        #   self._score_cache_contents(session, config, beam_size, test_dataset, list(norm_logits[0]), id_cache, context, state)
        
        if correct_word.endswith('@@') or (token_map is not None and not file_tokens[subtoken_id]):
          if not in_token:
            correct_subtokens = []
            remember_state = state
            logits = norm_logits[0]
            # correct_token = correct_word[:-2]
            correct_token = correct_word
          else:
            # correct_token += correct_word[:-2]
            correct_token += correct_word
          correct_subtokens.append(correct_word)
          in_token = True
          continue
        else:
          tokens_done += 1
          if not id_map is None and is_id:
            identifiers += 1
            file_identifiers += 1
          if not in_token:
            correct_subtokens = []
            remember_state = state
            logits = norm_logits[0]
            correct_token = correct_word
          else:
            correct_token += correct_word
            in_token = False
          correct_subtokens.append(correct_word)

        full_tokens_found = 0
        full_tokens = []

        project_cache_preds = None
        if is_id and id_cache.has_key(correct_token):
          ids_in_cache += 1.0
        if is_id and project_id_cache.has_key(correct_token):
          ids_in_project_cache += 1.0
        if is_id and tuple(context_history) in ngram_cache:
          context_history_in_ngram_cache += 1
          # print('file context hit:', correct_token in ngram_cache[tuple(context_history)], len(ngram_cache[tuple(context_history)]))
          file_cache_preds = self._score_cache_contents(session, config, beam_size, test_dataset, \
              list(norm_logits[0]), ngram_cache[tuple(context_history)], context, remember_state)
          # print(file_cache_preds, correct_token.replace('@@', ''))
        if is_id and tuple(context_history) in ngram_project_cache:
          context_history_in_ngram_project_cache += 1
          # print('project context hit:', correct_token in ngram_project_cache[tuple(context_history)], len(ngram_project_cache[tuple(context_history)]))
          project_cache_preds = self._score_cache_contents(session, config, beam_size, test_dataset, \
              list(norm_logits[0]), ngram_project_cache[tuple(context_history)], context, remember_state)
          # print(project_cache_preds, correct_token.replace('@@', ''))
          # print()

        # Rank single subtoken long predictions and keep top_needed (usually 10) best complete token ones
        sorted = list(enumerate(logits))
        sorted.sort(key=itemgetter(1), reverse=True)
        complete_done = 0
        prob_mass = 0.0
        counted = 0
        for id, prob in sorted:
          if cache_ids and is_id and False:
            word = test_dataset.rev_vocab[id]
            counted += 1
            if not word.endswith('@@'):
              if id_cache.has_key(word) or prob >= SKIP_CACHE_PROB_THRESHOLD:
                complete_done += 1
                full_tokens.append((prob, word))
                prob_mass += prob
                if complete_done >= top_needed:
                  break
                  if verbose: print(full_tokens)
          else:
            counted += 1
            word = test_dataset.rev_vocab[id]
            if not word.endswith('@@'):
              complete_done += 1
              full_tokens.append((prob, word))
              prob_mass += prob
              if complete_done >= top_needed:
                break
                if verbose: print(full_tokens)

        # Probability mass greater than satisfaction_prob so output this prediction
        if prob_mass > satisfaction_prob or counted == top_needed:
          rank = 0
          correct_found = False
          # if verbose: print('correct_token:', correct_token)
          if verbose: print('correct_token:', correct_token.replace('@@', ''))
          
          # if cache_ids and is_id:
          #   # cache_predictions = self._score_cache_contents(session, config, beam_size, test_dataset, \
          #   #   list(norm_logits[0]), id_cache, context, remember_state)
          #   pred_scores = dict()
          #   for prob, pred in cache_predictions:
          #     pred_scores[pred] = CACHE_WEIGHT * prob
          #   for prob, prediction in full_tokens:
          #     prediction = prediction.replace('@@', '')
          #     if prediction in pred_scores:
          #       pred_scores[prediction] = pred_scores[prediction] + (1.0 - CACHE_WEIGHT) * prob
          #     else:
          #       prediction = (1.0 - CACHE_WEIGHT) * prob
          #   # sort
          #   f_tokens = []
          #   for pred in pred_scores:
          #     f_tokens.append((pred_scores[pred], pred))
          #   f_tokens.sort(reverse=True)
          #   full_tokens = f_tokens[: 10]
          if cache_ids and is_id and project_cache_preds is not None:
            pred_scores = dict()
            for prob, pred in project_cache_preds:
              pred_scores[pred] = CACHE_WEIGHT * prob
            for prob, prediction in full_tokens:
              prediction = prediction.replace('@@', '')
              if prediction in pred_scores:
                pred_scores[prediction] = pred_scores[prediction] + (1.0 - CACHE_WEIGHT) * prob
              else:
                prediction = (1.0 - CACHE_WEIGHT) * prob
            # sort
            f_tokens = []
            for pred in pred_scores:
              f_tokens.append((pred_scores[pred], pred))
            f_tokens.sort(reverse=True)
            full_tokens = f_tokens[: 10]
          
          for prob, prediction in full_tokens:
            if FLAGS.token_model and correct_token == '-UNK-':
              break
            if (correct_token == '-UNK-' or '-UNK-' in correct_subtokens) and FLAGS.completion_unk_wrong:
                break
            if verbose: print(prob, prediction)
            if not correct_found:
              rank += 1
            # if prediction == correct_token:
            if prediction.replace('@@', '') == correct_token.replace('@@', ''):
              mrr += 1.0 / rank
              correct_found = True
              if verbose: print('MRR:', mrr / tokens_done)
              if verbose: print()
              
              if is_id:
                id_mrr += 1.0 / rank
                if rank <= 1:
                  id_acc1 += 1.0
                if rank <= 3:
                  id_acc3 += 1.0
                if rank <= 5:
                  id_acc5 += 1.0
                if rank <= 10:
                  id_acc10 += 1.0
          if not correct_found:
              rank += 1
          # if is_id: print(correct_token.replace('@@', ''), full_tokens, '\n')
          # if cache_ids and is_id: 
          #   print(rank)
          #   if rank > 10:
          #     print(correct_token, full_tokens, cache_predictions)
          
          if cache_ids and is_id and correct_token != '-UNK-':
            id_cache[correct_token] = True
            project_id_cache[correct_token] = True
            if tuple(context_history) in ngram_cache:
              ngram_cache[tuple(context_history)].add(correct_token)
            else:
              ngram_cache[tuple(context_history)] = set()
              ngram_cache[tuple(context_history)].add(correct_token)
            if tuple(context_history) in ngram_project_cache:
              ngram_project_cache[tuple(context_history)].add(correct_token)
            else:
              ngram_project_cache[tuple(context_history)] = set()
              ngram_project_cache[tuple(context_history)].add(correct_token)
            context_history.append(correct_token)
          continue
        if FLAGS.token_model: print('???')

        # Remember the score of the worst one out of the top_needed (usually 10) full_token candidates
        if len(full_tokens) > 0: worst_full_score = full_tokens[-1][0]
        else:  worst_full_score = 0.0
        # Create a priority queue to rank predictions and continue the search
        heapq.heapify(full_tokens)
        # Now find beam_size best candidates to initialize the search
        candidates_pq = []
        for id, prob in sorted:
          word = test_dataset.rev_vocab[id]
          if verbose: print(word, prob)
          if word.endswith('@@'):
            if cache_ids and is_id and False:
              # if id_cache.has_subtrie(word[-2]) or prob >= SKIP_CACHE_PROB_THRESHOLD:
              if id_cache.has_subtrie(word) or prob >= SKIP_CACHE_PROB_THRESHOLD:
                # All the initial state vectors are the same so the first is used
                # candidates_pq.append((-prob, Candidate(remember_state[0][0], id, word[:-2], -prob,
                #                                       tuple(tokens_before) + (word,))))
                candidates_pq.append((-prob, Candidate(remember_state[0][0], id, word, -prob,
                                                      tuple(tokens_before) + (word,))))
            else:
              # All the initial state vectors are the same so the first is used
              # candidates_pq.append((-prob, Candidate(remember_state[0][0], id, word[:-2], -prob,
              #                                       tuple(tokens_before) + (word,))))
              candidates_pq.append((-prob, Candidate(remember_state[0][0], id, word, -prob,
                                                    tuple(tokens_before) + (word,))))
          if len(candidates_pq) >= beam_size:
            break
        heapq.heapify(candidates_pq)
        full_tokens_scored = 0

        # Keep creating candidates until 5000 have been created or total probability mass has exceeded satisfaction_prob
        # Search can stop earlier if the best current candidate has score worst than that
        # of the worst one of the initial full_tokens since it would be pointless to further continue the search
        search_iterations = 0
        while full_tokens_scored < 5000 and prob_mass <= satisfaction_prob and search_iterations < 8:
          search_iterations += 1
          # Create a beam of new candidates until 500 full tokens have been produced
          to_expand = []
          new_state = (np.empty([beam_size, config.hidden_size]), )
          for c_id in range(beam_size):
            if len(candidates_pq) == 0:
              break
            to_expand.append(heapq.heappop(candidates_pq))
            new_state[0][c_id] = to_expand[-1][1].get_state_vec()

          if len(to_expand) < beam_size: break
          if -to_expand[0][1].get_parent_prob() < worst_full_score:
            break

          feed_dict = {self.inputd: np.array([[candidate.get_id()] for (score, candidate) in to_expand]),
                        self.keep_probability: 1.0
                      }
          if FLAGS.gru:
            for i, h in enumerate(self.reset_state):
              feed_dict[h] = new_state[i]
          else:
            for i, (c, h) in enumerate(self.reset_state):
              feed_dict[c] = new_state[i].c
              feed_dict[h] = new_state[i].h
          norm_logits, new_state = session.run([self.norm_logits, self.next_state], feed_dict)
          for c_id in range(beam_size):
            _, candidate = to_expand[c_id]
            logits = norm_logits[c_id]
            sorted = list(enumerate(logits))
            sorted.sort(key=itemgetter(1), reverse=True)

            for i in range(beam_size):
              id, prob = sorted[i]
              new_prob = candidate.get_parent_prob() * prob
              if cache_ids and is_id and False:
                if not test_dataset.rev_vocab[id].endswith('@@'):
                  if id_cache.has_key(candidate.get_text() + test_dataset.rev_vocab[id]) or new_prob >= SKIP_CACHE_PROB_THRESHOLD:
                    full_tokens_scored += 1
                    prob_mass += -new_prob
                    heapq.heappushpop(full_tokens, (-new_prob, candidate.get_text() + test_dataset.rev_vocab[id]))
                    worst = heapq.nsmallest(1, full_tokens)
                    worst_full_score = worst[0][0]
                  else:
                    # word = test_dataset.rev_vocab[id][:-2]
                    word = test_dataset.rev_vocab[id]
                    if id_cache.has_subtrie(candidate.get_text() + word) or new_prob >= SKIP_CACHE_PROB_THRESHOLD:
                      heapq.heappush(candidates_pq, (new_prob, Candidate(new_state[0][c_id], id, candidate.get_text() + word,
                                                                        new_prob, tuple(candidate.get_subtoken_history()) +
                                                                        (test_dataset.rev_vocab[id],))))
              else:
                if not test_dataset.rev_vocab[id].endswith('@@'):
                  full_tokens_scored += 1
                  prob_mass += -new_prob
                  heapq.heappushpop(full_tokens, (-new_prob, candidate.get_text() + test_dataset.rev_vocab[id]))
                  worst = heapq.nsmallest(1, full_tokens)
                  worst_full_score = worst[0][0]
                else:
                  # word = test_dataset.rev_vocab[id][:-2]
                  word = test_dataset.rev_vocab[id]
                  heapq.heappush(candidates_pq, (new_prob, Candidate(new_state[0][c_id], id, candidate.get_text() + word,
                                                                    new_prob, tuple(candidate.get_subtoken_history()) +
                                                                    (test_dataset.rev_vocab[id],))))

        # Get top and count rank of correct answer
        # if verbose: print('Correct_token:', correct_token)
        if verbose: print('Correct_token:', correct_token.replace('@@', ''), correct_token)
        if cache_ids: 
          if is_id:
            id_cache[correct_token] = True
            project_id_cache[correct_token] = True
            if tuple(context_history) in ngram_cache:
              ngram_cache[tuple(context_history)].add(correct_token)
            else:
              ngram_cache[tuple(context_history)] = set()
              ngram_cache[tuple(context_history)].add(correct_token)
            if tuple(context_history) in ngram_project_cache:
              ngram_project_cache[tuple(context_history)].add(correct_token)
            else:
              ngram_project_cache[tuple(context_history)] = set()
              ngram_project_cache[tuple(context_history)].add(correct_token)
            context_history.append(correct_token)
        
        full_tokens.sort(reverse=True)
        
        # if cache_ids and is_id:
        #   # print('full_tokens:', full_tokens)
        #   # cache_predictions = self._score_cache_contents(session, config, beam_size, test_dataset, \
        #   #   list(norm_logits[0]), id_cache, context, remember_state)
        #   pred_scores = dict()
        #   for prob, pred in cache_predictions:
        #     pred_scores[pred] = CACHE_WEIGHT * prob
        #   for prob, prediction in full_tokens:
        #     prediction = prediction.replace('@@', '')
        #     if prediction in pred_scores:
        #       pred_scores[prediction] = pred_scores[prediction] + (1.0 - CACHE_WEIGHT) * prob
        #     else:
        #       prediction = (1.0 - CACHE_WEIGHT) * prob
        #   # sort
        #   f_tokens = []
        #   for pred in pred_scores:
        #     f_tokens.append((pred_scores[pred], pred))
        #   f_tokens.sort(reverse=True)
        #   full_tokens = f_tokens[: 10]
        #   # print('new full tokens:', full_tokens)
        #   # print(correct_token)
        #   # print()

        if cache_ids and is_id and project_cache_preds is not None:
            pred_scores = dict()
            for prob, pred in project_cache_preds:
              pred_scores[pred] = CACHE_WEIGHT * prob
            for prob, prediction in full_tokens:
              prediction = prediction.replace('@@', '')
              if prediction in pred_scores:
                pred_scores[prediction] = pred_scores[prediction] + (1.0 - CACHE_WEIGHT) * prob
              else:
                prediction = (1.0 - CACHE_WEIGHT) * prob
            # sort
            f_tokens = []
            for pred in pred_scores:
              f_tokens.append((pred_scores[pred], pred))
            f_tokens.sort(reverse=True)
            full_tokens = f_tokens[: 10]
        
        correct_found = False
        for i, answer in enumerate(full_tokens):
          if (correct_token == '-UNK-' or '-UNK-' in correct_subtokens) and FLAGS.completion_unk_wrong:
                break
          prob, prediction = answer
          if verbose: print(-prob, prediction)
          if prediction.replace('@@', '') == correct_token.replace('@@', ''):
            correct_found = True
            mrr += 1.0 / (i + 1)
            if verbose: print('MRR:', mrr / tokens_done)
            if verbose: print()
            
            if is_id:
              id_mrr += 1.0 / (i + 1)
              if (i + 1) <= 1:
                id_acc1 += 1.0
              if (i + 1) <= 3:
                id_acc3 += 1.0
              if (i + 1) <= 5:
                id_acc5 += 1.0
              if (i + 1) <= 10:
                id_acc10 += 1.0
            break
        if not correct_found: i += 1
        # if cache_ids and is_id: 
        #   print(i + 1)
        #   if i + 1 > 10:
        #     print(correct_token, full_tokens, cache_predictions)
        # if is_id: print(correct_token.replace('@@', ''), full_tokens, '\n')
      files_done += 1
      if cache_ids:
        print(id_cache)

      # Train on remainder
      if dynamic and len(file_data) - train_start > 1:
        tr_context, tr_target, tr_target_weights = file_data[train_start : -1], file_data[train_start + 1:], \
                                                   [1.0] * len(file_data[train_start : -1])
        feed_dict = {self.inputd: np.tile(tr_context, (self.batch_size, 1)),
                     self.targets: np.tile(tr_target, (self.batch_size, 1)),
                     self.target_weights: np.tile(tr_target_weights, (self.batch_size, 1)),
                     self.learning_rate: config.learning_rate,
                     self.keep_probability: config.keep_prob
                    }
        if FLAGS.gru:
          for i, h in enumerate(self.reset_state):
            feed_dict[h] = train_state[i]
        else: # LSTM
          for i, (c, h) in enumerate(self.reset_state):
            feed_dict[c] = train_state[i].c
            feed_dict[h] = train_state[i].h
        _, cost, train_state, loss, iteration = session.run(
          [self.train_step, self.cost, self.next_state, self.loss, self.iteration], feed_dict)

      print(files_done, 'MRR:', mrr / tokens_done)
      if not id_map is None :
        print(id_mrr / identifiers, id_acc1 / identifiers, id_acc3 / identifiers, \
          id_acc5 / identifiers, id_acc10 / identifiers)
        if file_identifiers > 0:
          print('File cache recall', ids_in_cache / file_identifiers)
          print('Project cache recall', ids_in_project_cache / file_identifiers)
          print('File context recall', context_history_in_ngram_cache / file_identifiers )
          print('Project context recall', context_history_in_ngram_project_cache / file_identifiers )

    print('Tokens scored:', tokens_done)
    return mrr / tokens_done
  

  def _score_cache_contents(self, session, config, beam_size, test_dataset, logits, id_cache, context, state):
    """[summary]
    
    Arguments:
        session {[type]} -- [description]
        config {[type]} -- [description]
        beam_size {[type]} -- [description]
        test_dataset {[type]} -- [description]
        logits {[type]} -- [description]
        id_cache {[type]} -- [description]
        context {[type]} -- [description]
        state {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    ranked_pred = []
    heapq.heapify(ranked_pred)
    candidates_pq = []
    heapq.heapify(candidates_pq)
    for identifier in id_cache:
      # print(identifier)
      if not '@@' in identifier:
        index = test_dataset.vocab[identifier]
        prob = logits[index]
        # print(prob)
        if len(ranked_pred) < 10: heapq.heappush(ranked_pred, (prob, identifier))
        else: heapq.heappushpop(ranked_pred, (prob, identifier))
      else:
        identifier_parts = identifier.split('@@')
        index = test_dataset.vocab[identifier_parts[0] + '@@']
        prob = logits[index]
        candidates_pq.append((-prob, Candidate(state[0][0], index, identifier_parts, -prob, [0])))
        # unscored.append((identifier_parts, logits[index]))
    # print(ranked_pred)
    # print(candidates_pq)
    
    while len(candidates_pq) > 0:
      to_expand = []
      new_state = (np.empty([beam_size, config.hidden_size]), )
      for c_id in range(beam_size):
        if len(candidates_pq) == 0:
          break
        to_expand.append(heapq.heappop(candidates_pq))
        new_state[0][c_id] = to_expand[-1][1].get_state_vec()

      missing = beam_size - len(to_expand)
      for m in range(missing):
        to_expand.append((0.0, Candidate(state[0][0], 0, [], 0.0, [0])))

      # if len(to_expand) < beam_size: break

      feed_dict = {self.inputd: np.array([[candidate.get_id()] for (score, candidate) in to_expand]),
                    self.keep_probability: 1.0
                  }
      if FLAGS.gru:
        for i, h in enumerate(self.reset_state):
          feed_dict[h] = new_state[i]
      else:
        for i, (c, h) in enumerate(self.reset_state):
          feed_dict[c] = new_state[i].c
          feed_dict[h] = new_state[i].h
      norm_logits, new_state = session.run([self.norm_logits, self.next_state], feed_dict)
      for c_id in range(beam_size):
        _, candidate = to_expand[c_id]
        if candidate.get_parent_prob() == 0.0:
          continue
        
        logits = list(norm_logits[c_id])
        identifier_parts = candidate.get_text()
        # print(identifier_parts)
        next_part_index = candidate.get_subtoken_history()[-1] + 1
        if next_part_index == len(identifier_parts) - 1:
          index = test_dataset.vocab[identifier_parts[next_part_index]]
          prob = logits[index]
          new_prob = candidate.get_parent_prob() * prob
          candidate_text = ''.join(candidate.get_text())
          if len(ranked_pred) < 10:
            # print('pushing full token:', (-new_prob, candidate_text))
            heapq.heappush(ranked_pred, (-new_prob, candidate_text))
          else:
            # print('pushing full token:', (-new_prob, candidate_text))
            heapq.heappushpop(ranked_pred, (-new_prob, candidate_text))
        else:
          index = test_dataset.vocab[identifier_parts[next_part_index] + '@@']
          prob = logits[index]
          new_prob = candidate.get_parent_prob() * prob
          # print('Pushing new candidate:', (new_prob, Candidate(new_state[0][c_id], index, candidate.get_text(),
          #                                                       new_prob, list(candidate.get_subtoken_history()) + [next_part_index])))
          heapq.heappush(candidates_pq, (new_prob, Candidate(new_state[0][c_id], index, candidate.get_text(),
                                                                new_prob, list(candidate.get_subtoken_history()) + [next_part_index])))
    
    ranked_pred.sort(reverse=True)
    scores = np.asarray([prob for prob, token in ranked_pred])
    scores_sum = sum(scores)
    scores = [score / scores_sum for score in scores]
    # print(ranked_pred)
    norm_pred = []
    for i in range(len(ranked_pred)):
      # print(scores[i])
      norm_pred.append( (scores[i], ranked_pred[i][1]) )
    # print(candidates_pq)
    # print('cache norm pred:', norm_pred)
    # sys.exit(0)
    return norm_pred


  def maintenance_completion(self, session, config, test_lines, test_projects, train_vocab, train_vocab_rev, beam_size):
    """
    Runs the code completion for the maintenance scenario.
    :param session: The TF session in which operations should be run.
    :param config: The configuration to be used for the model.
    :param test_lines: A list containing each test file.
    :param test_projects: A list containing the test project name for each file.
    :param train_vocab: The word to id mapping.
    :param train_vocab_rev: The id to word mapping.
    :param beam_size: The size of the beam to be used by the search algorithm.
    :return: MRR for the test set.
    """
    # If checkpoint does not exist throw an exception
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if not tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
      raise Exception('Checkpoint does not exist!')

    new_learning_rate = config.learning_rate
    ctr = 0

    mrr = 0.0
    satisfaction_prob = 0.8
    top_needed = 10
    verbose = False
    train_every = 20
    tokens_done = 0

    # First compute which files belong to each project.
    project_sizes = []
    last_project_name = ''
    project_file_size = 0
    for test_project in test_projects:
      if test_project != last_project_name:
        if last_project_name != '':
          project_sizes.append(project_file_size)
          project_file_size = 0
      else:
        project_file_size += 1
      last_project_name = test_project
    print(project_sizes)
    print(sum(project_sizes))
    print()

    # Now distribute test lines based on project size
    project_test_lines = []
    files_distributed = 0
    for project_file_size in project_sizes:
      project_test_lines.append(test_lines[files_distributed : files_distributed + project_file_size])
      files_distributed += project_file_size

    partitions = 20
    for proj_id, test_lines in enumerate(project_test_lines):
      print(len(test_lines))
      large_project = len(test_lines) > 200

      if large_project:
        # The project is very large so partition in partition_size parts.
        # For each of the 20 parts train a model on all the parts excluding itself.
        partition_size = len(test_lines) / partitions
        for partition in range(partitions):
          partition_words = []
          for i, line in enumerate(test_lines):
            if i / partition_size != partition and not (i / partition_size == partitions + 1 and partition == partitions):
              partition_words.extend([word for word in line.split(' ')])
          # If checkpoint does not exist throw an exception
          ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
          if not tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            raise Exception('Checkpoint does not exist!')
          self.saver.restore(session, ckpt.model_checkpoint_path)

          partition_path = os.path.join(FLAGS.train_dir, "partition%d" % partition)
          if os.path.isdir(partition_path): shutil.rmtree(partition_path)

          partition_ids = [train_vocab[word] if word in train_vocab else train_vocab['-UNK-'] for word in partition_words]
          partition_dataset = reader.dataset(partition_ids, train_vocab, train_vocab_rev)

          # Reset the LSTM state to zeros and train
          state = session.run(self.reset_state)
          try:
            epoch_log_perp_unnorm = epoch_total_weights = 0.0
            epoch_sub_total_weights = 0.0
            for step, (context, target, target_weights, sub_target_weights) in enumerate(
                    partition_dataset.batch_producer(1, self.num_steps)):
              feed_dict = {self.inputd: np.tile(context, (self.batch_size, 1)),
                           self.targets: np.tile(target, (self.batch_size, 1)),
                           self.target_weights: np.tile(target_weights, (self.batch_size, 1)),
                           self.learning_rate: new_learning_rate,
                           self.keep_probability: config.keep_prob
                           }
              if FLAGS.gru:
                for i, h in enumerate(self.reset_state):
                  feed_dict[h] = state[i]
              else:
                for i, (c, h) in enumerate(self.reset_state):
                  feed_dict[c] = state[i].c
                  feed_dict[h] = state[i].h
              _, cost, state, loss, iteration = session.run(
                [self.train_step, self.cost, self.next_state, self.loss, self.iteration], feed_dict)
              epoch_log_perp_unnorm += np.sum(loss)
              epoch_total_weights += np.sum(sum(target_weights))
              epoch_sub_total_weights += np.sum(sum(sub_target_weights))
              # break
            train_log_perplexity = epoch_log_perp_unnorm / epoch_total_weights
            train_perplexity = math.exp(train_log_perplexity) if train_log_perplexity < 300 else float("inf")
            print(train_perplexity)
            checkpoint_path = os.path.join(partition_path, "model")
            self.saver.save(session, checkpoint_path, global_step=self.global_step)
          except (StopTrainingException, KeyboardInterrupt):
            print("Finished training ........")

      for lines_done, test_line in enumerate(test_lines):
        ctr += 1
        if tokens_done > 1000000:
          return mrr / tokens_done

        # Restore the global model
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
          self.saver.restore(session, ckpt.model_checkpoint_path)

        if large_project:
          # Load pretrained model and only train on the rest of the files from this partition
          partition = lines_done / partition_size
          partition = min(partitions - 1, partition) # last partition can be bigger
          partition_path =  os.path.join(FLAGS.train_dir, "partition%d" % partition)
          part_ckpt = tf.train.get_checkpoint_state(partition_path)
          self.saver.restore(session, part_ckpt.model_checkpoint_path)

          train_words = []
          if partition < partitions - 1:
            partition_lines = zip(range(partition * partition_size, (partition + 1) * partition_size),
                                  test_lines[partition * partition_size : (partition + 1) * partition_size])
          else:
            partition_lines = zip(range(partition * partition_size, len(test_lines)),
                                        test_lines[partition * partition_size : (partition + 1) * partition_size])
          for i, line in partition_lines:
            if i != (lines_done % partition_size):
              train_words.extend([word for word in line.split(' ')])
        else:
          # Now for each file in the current test project use the rest as train data
          train_words = []
          for i, line in enumerate(test_lines):
            if i != lines_done:
              train_words.extend([word for word in line.split(' ')])

        # Convert the train data words to ids
        ids = [train_vocab[word] if word in train_vocab else train_vocab['-UNK-'] for word in train_words]
        train_dataset = reader.dataset(ids, train_vocab, train_vocab_rev)

        # Reset the LSTM state to zeros and train
        state = session.run(self.reset_state)
        try:
          epoch_log_perp_unnorm = epoch_total_weights = 0.0
          epoch_sub_total_weights = 0.0
          for step, (context, target, target_weights, sub_target_weights) in enumerate(
                  train_dataset.batch_producer(1, self.num_steps)):
            feed_dict = {self.inputd: np.tile(context, (self.batch_size, 1)),
                          self.targets: np.tile(target, (self.batch_size, 1)),
                          self.target_weights: np.tile(target_weights, (self.batch_size, 1)),
                          self.learning_rate: new_learning_rate,
                          self.keep_probability: config.keep_prob
            }
            if FLAGS.gru:
              for i, h in enumerate(self.reset_state):
                feed_dict[h] = state[i]
            else:
              for i, (c, h) in enumerate(self.reset_state):
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h
                # print("state number " + str(i))
            _, cost, state, loss, iteration = session.run([self.train_step, self.cost, self.next_state, self.loss, self.iteration], feed_dict)
            epoch_log_perp_unnorm += np.sum(loss)
            epoch_total_weights += np.sum(sum(target_weights))
            epoch_sub_total_weights += np.sum(sum(sub_target_weights))
          train_log_perplexity = epoch_log_perp_unnorm / epoch_total_weights
          train_perplexity = math.exp(train_log_perplexity) if train_log_perplexity < 300 else float("inf")
          # lengths.append(int(round(epoch_sub_total_weights, 0)))

          # Training done. Now test on test file
          ids = [train_vocab[word] if word in train_vocab else train_vocab['-UNK-']
                  for word in test_lines[lines_done].split(' ')]

          state = session.run(self.reset_state)
          remember_state = state
          train_state = session.run(self.reset_state)
          in_token = False
          correct_token = ''
          train_start = 0
          train_end = 0

          # Reset the LSTM state to zeros and train
          for subtoken_id, context_target in enumerate(zip(ids[:-1], ids[1:])):
            context, target = context_target
            train_end += 1

            if (subtoken_id - 1) - train_start > train_every and not train_vocab_rev[context].endswith('@@'):
              # train
              tr_context, tr_target, tr_target_weights = ids[train_start: subtoken_id - 1], \
                                                         ids[train_start + 1: subtoken_id], \
                                                         [1.0] * ((subtoken_id - 1) - train_start)
              start_train_at = subtoken_id
              feed_dict = {self.inputd: np.tile(tr_context, (self.batch_size, 1)),
                           self.targets: np.tile(tr_target, (self.batch_size, 1)),
                           self.target_weights: np.tile(tr_target_weights, (self.batch_size, 1)),
                           self.learning_rate: config.learning_rate,
                           self.keep_probability: config.keep_prob
                           }
              if FLAGS.gru:
                for i, h in enumerate(self.reset_state):
                  feed_dict[h] = train_state[i]
              else:
                for i, (c, h) in enumerate(self.reset_state):
                  feed_dict[c] = train_state[i].c
                  feed_dict[h] = train_state[i].h
              _, cost, train_state, loss, iteration = session.run(
                [self.train_step, self.cost, self.next_state, self.loss, self.iteration], feed_dict)
            train_start = train_end

            feed_dict = {self.inputd: np.array([[context]] * self.batch_size),
                         self.targets: np.array([[target]] * self.batch_size),
                         self.target_weights: np.array([[1.0]] * self.batch_size),
                         self.keep_probability: 1.0
                        }
            if FLAGS.gru:
              for i, h in enumerate(self.reset_state):
                feed_dict[h] = state[i]
            else:
              for i, (c, h) in enumerate(self.reset_state):
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h
            norm_logits, loss, cost, state = session.run([self.norm_logits, self.loss, self.cost, self.next_state],
                                                         feed_dict)
            correct_word = train_vocab_rev[target]

            if correct_word.endswith('@@'):
              if not in_token:
                correct_subtokens = []
                remember_state = state
                logits = norm_logits[0]
                correct_token = correct_word[:-2]
              else:
                correct_token += correct_word[:-2]
              correct_subtokens.append(correct_word)
              in_token = True
              continue
            else:
              tokens_done += 1
              if not in_token:
                correct_subtokens = []
                remember_state = state
                logits = norm_logits[0]
                correct_token = correct_word
              else:
                correct_token += correct_word
                in_token = False
              correct_subtokens.append(correct_word)

            full_tokens_found = 0
            full_tokens = []

            # Rank single subtoken long predictions and keep top_needed (usually 10) best complete token ones
            sorted = list(enumerate(logits))
            sorted.sort(key=itemgetter(1), reverse=True)
            complete_done = 0
            prob_mass = 0.0
            counted = 0
            for id, prob in sorted:
              counted += 1
              word = train_vocab_rev[id]
              if not word.endswith('@@'):
                complete_done += 1
                full_tokens.append((prob, word))
                prob_mass += prob
                if complete_done >= top_needed:
                  break

            # Probability mass greater than satisfaction_prob so output this prediction
            if prob_mass > satisfaction_prob or counted == top_needed:
              rank = 0
              correct_found = False
              if verbose: print(correct_token)
              for prob, prediction in full_tokens:
                if verbose: print(prob, prediction)
                if not correct_found:
                  rank += 1
                if prediction == correct_token:
                  mrr += 1.0 / rank
                  correct_found = True
                  if verbose: print('MRR:', mrr / tokens_done)
                  if verbose: print()
              continue

            # Remember the score of the worst one out of the top_needed (usually 10) full_token candidates
            worst_full_score = full_tokens[-1][0]
            # Create a priority queue to rank predictions and continue the search
            heapq.heapify(full_tokens)
            # Now find beam_size best candidates to initialize the search
            candidates_pq = []
            for id, prob in sorted:
              word = train_vocab_rev[id]
              if word.endswith('@@'):
                # All state vectors are the same so the first is used
                candidates_pq.append((-prob, Candidate(remember_state[0][0], id, word[:-2], -prob, '')))
              if len(candidates_pq) >= beam_size:
                break
            heapq.heapify(candidates_pq)
            full_tokens_scored = 0

            # Keep creating candidates until 5000 have been created or total probability mass has exceeded satisfaction_prob
            # Search can stop earlier if the best current candidate has score worst than that
            # of the worst one of the initial full_tokens since it would be pointless to further continue the search
            search_iterations = 0
            while full_tokens_scored < 5000 and prob_mass <= satisfaction_prob and search_iterations < 8:
              search_iterations += 1
              # Create a beam of new candidates until 500 full tokens have been produced
              to_expand = []
              new_state = (np.empty([beam_size, config.hidden_size]),)
              for c_id in range(beam_size):
                if len(candidates_pq) == 0:
                  break
                to_expand.append(heapq.heappop(candidates_pq))
                new_state[0][c_id] = to_expand[-1][1].get_state_vec()

              if len(to_expand) < beam_size: break
              if -to_expand[0][1].get_parent_prob() < worst_full_score:
                break

              feed_dict = {self.inputd: np.array([[candidate.get_id()] for (score, candidate) in to_expand]),
                           self.keep_probability: 1.0
                           }
              if FLAGS.gru:
                for i, h in enumerate(self.reset_state):
                  feed_dict[h] = new_state[i]
              else:
                for i, (c, h) in enumerate(self.reset_state):
                  feed_dict[c] = new_state[i].c
                  feed_dict[h] = new_state[i].h
              norm_logits, new_state = session.run([self.norm_logits, self.next_state], feed_dict)
              for c_id in range(beam_size):
                if len(to_expand) <= c_id: break
                _, candidate = to_expand[c_id]
                logits = norm_logits[c_id]
                sorted = list(enumerate(logits))
                sorted.sort(key=itemgetter(1), reverse=True)
                for i in range(beam_size):
                  id, prob = sorted[i]
                  new_prob = candidate.get_parent_prob() * prob
                  if not train_vocab_rev[id].endswith('@@'):
                    full_tokens_scored += 1
                    prob_mass += -new_prob
                    heapq.heappushpop(full_tokens, (-new_prob, candidate.get_text() + train_vocab_rev[id]))
                    worst = heapq.nsmallest(1, full_tokens)
                    worst_full_score = worst[0][0]
                  else:
                    word = train_vocab_rev[id][:-2]
                    heapq.heappush(candidates_pq,
                                   (new_prob, Candidate(new_state[0][c_id], id, candidate.get_text() + word,
                                                        new_prob, '')))

            # Get top and count rank of correct answer
            if verbose: print(correct_token)
            full_tokens.sort(reverse=True)
            for i, answer in enumerate(full_tokens):
              prob, prediction = answer
              if verbose: print(-prob, prediction)
              if prediction == correct_token:
                mrr += 1.0 / (i + 1)
                if verbose: print('MRR:', mrr / tokens_done)
                if verbose: print()

          pass
          # Train on remainder
          if len(ids) - start_train_at > 1:
            tr_context, tr_target, tr_target_weights = ids[train_start : -1], \
                                                       ids[train_start + 1 : ], \
                                                       [1.0] * len(ids[train_start : -1])
            feed_dict = {self.inputd: np.tile(tr_context, (self.batch_size, 1)),
                         self.targets: np.tile(tr_target, (self.batch_size, 1)),
                         self.target_weights: np.tile(tr_target_weights, (self.batch_size, 1)),
                         self.learning_rate: config.learning_rate,
                         self.keep_probability: config.keep_prob
                        }
            if FLAGS.gru:
              for i, h in enumerate(self.reset_state):
                feed_dict[h] = train_state[i]
            else:
              for i, (c, h) in enumerate(self.reset_state):
                feed_dict[c] = train_state[i].c
                feed_dict[h] = train_state[i].h
            _, cost, train_state, loss, iteration = session.run(
              [self.train_step, self.cost, self.next_state, self.loss, self.iteration], feed_dict)

          print(ctr, 'MRR:', mrr / tokens_done)
          sys.stdout.flush()
        except (StopTrainingException, KeyboardInterrupt):
          print("Finished training ........")

      print("Projects Done:", proj_id + 1)
    return mrr / tokens_done


  def write_model_parameters(self, model_directory):
    """
    Saves basic model information.
    :param model_directory:
    :return:
    """
    parameters = {
      "num_layers": str(self.num_layers),
      "vocab_size": str(self.vocab_size),
      "hidden_size": str(self.hidden_size),
      "keep_probability": str(self.keep_probability),
      "total_parameters": str(self.get_parameter_count())
    }
    with open(self.parameters_file(model_directory), "w") as f:
      json.dump(parameters, f, indent=4)

  @staticmethod
  def parameters_file(model_directory):
    return os.path.join(model_directory, "parameters.json")

  @staticmethod
  def model_file(model_directory):
    return os.path.join(model_directory, "model")

def calculate_predictability(test_lines, train_vocab, train_vocab_rev, config, output_path, model, session):
  """
  Computes predictability for each test file instance and returs the average.
  :param test_lines:
  :param train_vocab:
  :param train_vocab_rev:
  :param config:
  :param output_path:
  :param model:
  :param session: The TF session in which operations should be run.
  :return:
  """
  config.batch_size = config.test_batch_size
  ctr = 0
  predictabilities = []
  for test_line in test_lines:
    ctr += 1
    if ctr % 1000 == 0:
      print("\t %d lines" % ctr)

    test_line = test_line.replace("\n", (" %s" % "-eod-"))
    ids = [train_vocab[word]
              if word in train_vocab else train_vocab["-UNK-"]
            for word in test_line.split(' ')]
    test_dataset = reader.dataset(ids, train_vocab, train_vocab_rev)
    file_predictability = model.test(session, config, test_dataset)
    predictabilities.append(file_predictability)
  average_predictability = sum(predictabilities) / ctr
  return average_predictability

def do_test(test_path, train_vocab, train_vocab_rev, config):
  test_wids = reader._file_to_word_ids(test_path, train_vocab)
  test_dataset = reader.dataset(test_wids, train_vocab, train_vocab_rev)
  with tf.Graph().as_default():
    with tf.Session(config=get_gpu_config()) as session:
      model = create_model(session, config)
      model.train_vocab = train_vocab
      test_perplexity = model.test(session, config, test_dataset)
      print("\n\nTest perplexity is " + str(test_perplexity) + "\n")

def create_model(session, config):
  """
  Creates the NLM and restores its parameters if there is a saved checkpoint.
  :param session: The TF session in which operations will be run.
  :param config: The configuration to be used.
  :return:
  """
  model = NLM(config)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters:")
    session.run(tf.global_variables_initializer())
  print("*Number of parameters* = " + str(model.get_parameter_count()))
  return model


class Config(object):
  """Configuration"""

  def __init__(self, inits, lr, mgrad, nlayers, nsteps, hsize, mepoch, kp, decay, bsize, tbsize, vsize):
    self.init_scale = inits
    self.learning_rate = lr
    self.max_grad_norm = mgrad
    self.num_layers = nlayers
    self.num_steps = nsteps
    self.hidden_size = hsize
    self.max_epoch = mepoch
    self.keep_prob = kp
    self.lr_decay = decay
    self.batch_size = bsize
    self.test_batch_size = tbsize
    self.vocab_size = vsize



def main(_):
    """
    Handles argument parsing and runs the chosen scenario.
    """
    if not FLAGS.data_path:
      raise ValueError("Must set --data_path to directory with train/valid/test")

    config = Config(FLAGS.init_scale, FLAGS.learning_rate, FLAGS.max_grad_norm, FLAGS.num_layers, FLAGS.num_steps,
                    FLAGS.hidden_size, FLAGS.max_epoch, FLAGS.keep_prob, FLAGS.lr_decay, FLAGS.batch_size,
                    FLAGS.test_batch_size, FLAGS.vocab_size)

    exit_criteria = ExitCriteria(config.max_epoch)

    if FLAGS.predict:
      # Runs the predictability scenario (per file entropy/perplexity).
      vocab_path = FLAGS.train_dir + "/vocab.txt"
      train_vocab, train_vocab_rev = reader._read_vocab(vocab_path)
      config.vocab_size = len(train_vocab)
      start_time = time.time()

      test_lines = []
      valid_file = FLAGS.data_path + '/' + FLAGS.validation_filename
      with open(valid_file, 'r') as f:
        test_lines = [line for line in f]
      # test_lines = ['class Car { public static void main ( String [ ] args) }\n']
      with tf.Graph().as_default():
        with tf.Session(config=get_gpu_config()) as session:
          model = create_model(session, config)
          model.train_vocab = train_vocab
          model.train_vocab_rev = train_vocab_rev
          ppl = calculate_predictability(test_lines, train_vocab, train_vocab_rev, config, '', model, session)
      print("Average:", ppl)
      print("Total time %s" % timedelta(seconds=time.time() - start_time))
      print("Done computing predictability scores!")
    elif FLAGS.dynamic_test:
      # Runs the dynamic adaptation scenario.
      vocab_path = FLAGS.train_dir + "/vocab.txt"
      train_vocab, train_vocab_rev = reader._read_vocab(vocab_path)
      config.vocab_size = len(train_vocab)
      start_time = time.time()

      test_lines = []
      test_file = FLAGS.data_path + '/' + FLAGS.test_filename
      test_proj_file = FLAGS.data_path + '/' + FLAGS.test_proj_filename
      with open(test_file, 'r') as f:
        test_lines = [line for line in f]
      with open(test_proj_file, 'r') as f:
        test_proj_lines = [line for line in f]
      # test_lines = ['class Car { public static void main ( String [ ] args) }\n']
      with tf.Graph().as_default():
        with tf.Session(config=get_gpu_config()) as session:
          model = create_model(session, config)
          model.train_vocab = train_vocab
          model.train_vocab_rev = train_vocab_rev
          perplexity = model.dynamic_train_test(test_lines, train_vocab, train_vocab_rev,
                                                test_proj_lines, config, '', session)
          print('Average perplexity:', perplexity)
      print("Total time %s" % timedelta(seconds=time.time() - start_time))
      print("Done computing predictability scores!")
    elif FLAGS.maintenance_test:
      # Runs the code maintenance scenario and calculates entropy/perplexity.
      vocab_path = FLAGS.train_dir + "/vocab.txt"
      train_vocab, train_vocab_rev = reader._read_vocab(vocab_path)
      config.vocab_size = len(train_vocab)
      start_time = time.time()

      test_lines = []
      test_file = FLAGS.data_path + '/' + FLAGS.test_filename
      test_proj_file = FLAGS.data_path + '/' + FLAGS.test_proj_filename
      with open(test_file, 'r') as f:
        test_lines = [line.replace("\n", (" %s" % "-eod-")) for line in f]
      with open(test_proj_file, 'r') as f:
        test_proj_lines = [line.rstrip('\n') for line in f]
      # test_lines = ['class Car { public static void main ( String [ ] args) }\n']
      with tf.Graph().as_default():
        with tf.Session(config=get_gpu_config()) as session:
          model = create_model(session, config)
          perplexity = model.maintenance_test(session, config, test_lines, test_proj_lines, train_vocab, train_vocab_rev)
          print('Average perplexity:', perplexity)
      print("Total time %s" % timedelta(seconds=time.time() - start_time))
      print("Done computing predictability scores!")
    elif FLAGS.test:
      # Default test scenario. Essentially entropy/perplexity calculation.
      vocab_path = FLAGS.train_dir + "/vocab.txt"
      train_vocab, train_vocab_rev = reader._read_vocab(vocab_path)
      print(len(train_vocab))
      config.vocab_size = len(train_vocab)
      start_time = time.time()
      do_test(FLAGS.data_path + "/" + FLAGS.test_filename, train_vocab, train_vocab_rev, config)
      print("Total time %s" % timedelta(seconds=time.time() - start_time))
      print("Done testing!")
    elif FLAGS.completion:
      # Runs the code completion scenario and calculates MRR if dynamic adaptation is on it also adapts the model.
      vocab_path = FLAGS.train_dir + "/vocab.txt"
      train_vocab, train_vocab_rev = reader._read_vocab(vocab_path)
      config.vocab_size = len(train_vocab)
      start_time = time.time()
      test_wids = reader._file_to_word_ids(FLAGS.data_path + "/" + FLAGS.test_filename, train_vocab)
      test_dataset = reader.dataset(test_wids, train_vocab, train_vocab_rev)
      if FLAGS.dynamic:
        test_proj_file = FLAGS.data_path + '/' + FLAGS.test_proj_filename
        with open(test_proj_file, 'r') as f:
          test_proj_lines = [line for line in f]
      else:
        test_proj_lines = []
      with tf.Graph().as_default():
        with tf.Session(config=get_gpu_config()) as session:
          model = create_model(session, config)
          model.train_vocab = train_vocab
          model.train_vocab_rev = train_vocab_rev

          id_map = None
          if FLAGS.identifier_map:
            id_map = []
            with open(FLAGS.identifier_map, 'r') as f:
              for line in f:
                id_map.append(ast.literal_eval(line.rstrip('\n')))
          
          token_map = None
          if FLAGS.subtoken_map:
            token_map = []
            with open(FLAGS.subtoken_map, 'r') as f:
              for line in f:
                token_map.append(ast.literal_eval(line.rstrip('\n')))

          mrr = model.completion(session, config, test_dataset, test_proj_lines, config.batch_size, \
            FLAGS.dynamic, id_map, FLAGS.cache_ids, token_map)
          print(mrr)
      print("Total time %s" % timedelta(seconds=time.time() - start_time))
      print("Done completion!")
    elif FLAGS.maintenance_completion:
      # Runs the maintenance code completion scenario and calculates MRR.
      vocab_path = FLAGS.train_dir + "/vocab.txt"
      train_vocab, train_vocab_rev = reader._read_vocab(vocab_path)
      config.vocab_size = len(train_vocab)
      start_time = time.time()
      test_lines = []
      test_file = FLAGS.data_path + '/' + FLAGS.test_filename
      test_proj_file = FLAGS.data_path + '/' + FLAGS.test_proj_filename
      with open(test_file, 'r') as f:
        test_lines = [line.replace("\n", (" %s" % "-eod-")) for line in f]
      with open(test_proj_file, 'r') as f:
        test_proj_lines = [line.rstrip('\n') for line in f]

      with tf.Graph().as_default():
        with tf.Session(config=get_gpu_config()) as session:
          model = create_model(session, config)
          mrr = model.maintenance_completion(session, config, test_lines, test_proj_lines, train_vocab,
                                       train_vocab_rev, config.batch_size)
          print('Final MRR:', mrr)
      print("Total time %s" % timedelta(seconds=time.time() - start_time))
      print("Done computing mean reciprocal rank!")
    else:
      # Default scenario. Trains on training set and calculates entropy/perplexity for each epoch on the validation set.
      train_file = FLAGS.data_path + '/' + FLAGS.train_filename #"/java_10M_train_bpe"
      valid_file = FLAGS.data_path + '/' + FLAGS.validation_filename #"/java_validation_10%_sample_bpe"
      train_vocab, train_vocab_rev = reader._build_vocab(train_file, FLAGS.thresh)
      print("Vocabulary size:", len(train_vocab))
      config.vocab_size = len(train_vocab) # change so that vocab also reflects UNK, EMPTY, EOS etc
      reader._write_vocab(train_vocab, FLAGS.train_dir + "/vocab.txt")

      train_wids = reader._file_to_word_ids(train_file, train_vocab)
      train_dataset = reader.dataset(train_wids, train_vocab, train_vocab_rev)
      val_wids = reader._file_to_word_ids(valid_file, train_vocab)
      valid_dataset = reader.dataset(val_wids, train_vocab, train_vocab_rev)
      del train_wids
      del val_wids

      start_time = time.time()
      with tf.Graph().as_default():
        with tf.Session(config=get_gpu_config()) as session:
          md = create_model(session, config)
          md.train_vocab = train_vocab
          md.train_vocab_rev = train_vocab_rev
          md.write_model_parameters(FLAGS.train_dir)
          md.train(session, config, train_dataset, exit_criteria, valid_dataset, FLAGS.train_dir)
      print("Total time %s" % timedelta(seconds=time.time() - start_time))
      print("Done training!")

class StopTrainingException(Exception):
  pass


class ExitCriteria(object):
  """
  Defines the criteria needed for training termination.
  """
  def __init__(self, max_epochs):
    self.max_epochs = max_epochs

class Candidate(object):
  """
  Represents a code completion search candidate.
  """
  def __init__(self, state_vec, id, token_text, parent_prob, subtoken_history):
    self._state_vec = state_vec
    self._id = id
    self._token_text = token_text
    self._parent_prob = parent_prob
    self._subtoken_history = subtoken_history

  def get_state_vec(self):
    return self._state_vec

  def get_id(self):
    return self._id

  def get_text(self):
    return self._token_text

  def get_parent_prob(self):
    return self._parent_prob

  def get_subtoken_history(self):
    return tuple(self._subtoken_history)
  
  def __eq__(self, other):
    return self._token_text == other._token_text
  
  def __lt__(self, other):
    return self._token_text < other._token_text

  def __gt__(self, other):
    return self._token_text > other._token_text

  def __le__(self, other):
    return self._token_text <= other._token_text

  def __ge__(self, other):
    return self._token_text >= other._token_text


if __name__=="__main__":
    tf.app.run()
