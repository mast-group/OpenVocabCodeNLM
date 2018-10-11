from __future__ import absolute_import
from __future__ import division
from itertools import chain

import collections
import getopt
import numpy as np
import sys
import tensorflow as tf

from collections import deque

UNKNOWN_WORD = "-UNK-"
EMPTY_WORD = "-EMP-"
END_SENT = "-eos-"
START_SENT = "-bos-"
END_DOC = "-eod-"
SUBWORD_END = "@@"


def _build_vocab(filename, threshold=5, debug=True):
  """
  Builds a vocabulary containing all the subword_units/words/tokens that appear at least #threshold times
  in file filename. All subword_units with frequency < #threshold are converted to a special -UNK- token.
  For the BPE NLM the threshold used should typically be 0.
  The vocabulary is represented as a pair of mappings from subword_units to ids and vice-versa.
  :param filename: The path of the file to be used for vocabulary creation.
  :param threshold: The frequency threshold for vocabulary inclusion.
  :param debug: Whether debugging information should be printed.
  :return: A pair of mappings from subword_units to ids and vice-versa.
  """
  with open(filename, 'r') as f:
    linewords = (line.replace("\n", " %s" % END_DOC).split() for line in f)
    counter = collections.Counter(chain.from_iterable(linewords))
  if debug: print('Read data for vocabulary!')

  counter[UNKNOWN_WORD] = 0
  unk_counts = 0
  for word, freq in counter.items():
    if freq < threshold:
        unk_counts += freq
        del counter[word] # Cleans up resources. Absolutely necessary for large corpora!
  if unk_counts > 0:
    counter[UNKNOWN_WORD] += unk_counts
  if debug: print('UNKS:', unk_counts)
  counter[EMPTY_WORD] = threshold
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words = [word for word, freq in count_pairs if freq >= threshold]
  word_to_id = dict(zip(words, range(len(words))))
  id_to_word = {v: k for k, v in word_to_id.items()}
  return word_to_id, id_to_word


def _file_to_word_ids(filename, word_to_id):
  """
  Creates the sequence of ids for a file based on the specified mapping.
  If a unit/word/token is not contained in the vocabulary then it is convert into the id of the special -UNK- token.
  Each line of the file is considered a different instance (sentence, code file, etc.)
  :param filename: The path of the file to be converted into a sequence of ids.
  :param word_to_id: Contains the mapping of vocabulary entries to their respective ids.
  :return: The mapped sequence of ids.
  """
  with open(filename, 'r') as f:
    ids = []
    for line in f:
      line = line.replace("\n", (" %s" % END_DOC))
      ids.extend([word_to_id[word]
                  if word in word_to_id else word_to_id[UNKNOWN_WORD] for word in line.split()])
  return ids

def _read_words(filename):
  """
  Reads a whitespace tokenized version of the specified file (in UTF-8 encoding) using Tensorflow's API.
  All whitespace characters at the beginning and ending of the file are trimmed.
  :param filename: The path of the file to be read.
  :return: The whitespace tokenized version of the specified file.
  """
  with tf.device('/cpu:0'):
    with tf.gfile.GFile(filename, "r") as f:
      return f.read().decode("utf-8").strip().split()


def _read_lines(filename):
  """
  Creates a list of the specified file's lines using Tensorflow API's (in UTF-8 encoding).
  Each line of the file is a separate list entry and all whitespace characters at its beginning and ending are trimmed.
  :param filename: The path of the file to be read.
  :return: A list of the specified file's lines.
  """
  with tf.device('/cpu:0'):
    with tf.gfile.GFile(filename, "r") as f:
      ret = []
      for l in f:
        ret.append(l.decode("utf8").strip())
      return ret

def _read_vocab(filename):
  """
  Reads the vocabulary from the specified file.
  The file should contain one vocabulary entry per line.
  Each line contains a word, id pair separated by a tab ('\t') character.
  :param filename: Path to the file that the vocabulary was stored into.
  :return: A pair of mappings from subword_units to ids and vice-versa.
  """
  with tf.device('/cpu:0'):
    word_to_id = {}
    id_to_word = {}
    with tf.gfile.GFile(filename, "r") as ff:
      for line in ff:
        word, iden = line.strip().split('\t')
        iden = int(iden)
        word_to_id[word] = iden
        id_to_word[iden] = word
    return word_to_id, id_to_word


def _write_vocab(vocab, filename):
  """
  Exports the vocabulary (mapping of subword_units/tokens to ids) to the specified file.
  :param vocab: A dictionary containing the mapping.
  :param filename: Path to the file in which the vocabulary will be saved.
  """
  with tf.device('/cpu:0'):
    with tf.gfile.GFile(filename, "w") as ff:
      for w, wid in vocab.items():
        ff.write(w + "\t" + str(wid) + "\n")

def _get_ids_for_wordlist(list_words, word_to_id):
  """
  Converts the specified list of subword_units/tokens/words to a list of ids based on the specified mapping.
  If a subword_unit/token/word is out of vocabulary then it is converted into the special -UNK- symbol.
  :param list_words: A list of subword_units/tokens/words.
  :param word_to_id: The mapping (vocabulary) to be used.
  :return:
  """
  ret = []
  for k in list_words:
    if k in word_to_id:
      ret.append(word_to_id[k])
    else:
      ret.append(word_to_id[UNKNOWN_WORD])
  return ret

def _get_id(word, word_to_id):
  if word in word_to_id:
    return word_to_id[word]
  return word_to_id[UNKNOWN_WORD]

def _get_empty_id(vocab):
  """
  Returns the id of the special symbol for empty words (-EMP-) for the specified vocabulary.
  :param vocab: The vocabulary to be used.
  :return: The id of the special symbol for empty words.
  """
  return vocab[EMPTY_WORD]

def _get_unknown_id(vocab):
  """
  Returns the id of the special symbol for unknown words (-UNK-) for the specified vocabulary.
  :param vocab: The vocabulary to be used.
  :return: The id of the special symbol for unknown words.
  """
  return vocab[UNKNOWN_WORD]

# Not used anymore. I think so at least. CHECKKKKKKKKKKKK!!!!!!!!
# def _create_dataset(wordlist, vocabulary):
#   """
#   Converts a list of subword_units/tokens/words in
#   :param wordlist: A tokenized list of a dataset's contents. The vocabulary is used to assign ids to the tokens.
#   :param vocabulary:
#   :return: A dataset: containing the words encoded as ids
#   """
#   encoded = _get_ids_for_wordlist(wordlist, vocabulary)
#   return dataset(encoded, vocabulary)


class dataset(object):
  """
  Represents a set of instances. Each instance is a code file but could also be a sentence in natural language.
  """
  def __init__(self, rdata, vocab, rev_vocab):
    """
    Creates and returns dataset instance. It contains a numpy array of ids and vocabulary mappings
    from subword_units to ids and vice-versa
    :param rdata:
    :param vocab:
    :param rev_vocab:
    """
    self.data = np.array(rdata, dtype=np.int32)
    self.vocab = vocab
    self.rev_vocab = rev_vocab

  def batch_producer_memory_efficient(self, batch_size, num_steps):
    """
    Generates batches of context and target pairs (the target is context time-shifted by one) in a memory efficient way based
    on the specified parameters (batch size and number of RNN steps).
    Each batch contains batch_size * num_steps ids.
    This method should be preferred when training on large corpora because the training corpus might not fit in memory.
    The only downside of this strategy is that the minibatches must be produced again for each epoch.
    :param batch_size: Minibatch size. Increases this parameter results in more parallelization.
    Making this parameter too big is not advised though since the average gradient of the batch
    is calculated during learning.
    :param num_steps: The lenght of the RNN sequence.
    :return: Yields minibatches of numpy arrays containing ids with dimensions: [batch_size, num_steps].
    """
    raw_data = self.data # is just one long array
    data_len = len(raw_data)
    nbatches = data_len // (num_steps * batch_size)
    max_index_covered = nbatches * num_steps * batch_size
    remaining_tok_cnt = data_len - max_index_covered
    to_fill = num_steps * batch_size - remaining_tok_cnt + 1 # the last +1 is for target of last times ep
    to_fill_array = np.full(to_fill, self.vocab[EMPTY_WORD], dtype=int)
    for i in range(nbatches - 1):
      ip_range = raw_data[(i * batch_size * num_steps): (i + 1) * (batch_size * num_steps)]
      tar_range = raw_data[(i * batch_size * num_steps) + 1: (i + 1) * (batch_size * num_steps) + 1]
      ip_wt_range = np.ones(len(tar_range), dtype=float)
      contexts = np.stack((np.array_split(ip_range, batch_size)), axis=0)
      targets = np.stack((np.array_split(tar_range, batch_size)), axis=0)
      weights = np.stack((np.array_split(ip_wt_range, batch_size)), axis=0)
      yield(contexts, targets, weights)
    # Yield to fill
    ip_range = np.concatenate((raw_data[max_index_covered:], to_fill_array[:-1]))
    tar_range = np.concatenate((raw_data[max_index_covered+1:], to_fill_array))
    ip_wt_range = np.concatenate((np.ones(remaining_tok_cnt - 1, dtype=float), np.zeros(len(to_fill_array), dtype=float)))
    contexts = np.stack((np.array_split(ip_range, batch_size)), axis=0)
    targets = np.stack((np.array_split(tar_range, batch_size)), axis=0)
    weights = np.stack((np.array_split(ip_wt_range, batch_size)), axis=0)
    yield (contexts, targets, weights)

  def batch_producer(self, batch_size, num_steps, subword_weights=True, debug=False):
    """
    Generates batches of context and target pairs (the target is context time-shifted by one)
    based on the specified parameters (batch size and number of RNN steps).
    Each batch contains batch_size * num_steps ids.
    This variation converts all the data once in huge numpy array.
    This method should be preferred when training on smaller corpora or during test time.
    Memory requirements scale with the data size.
    :param batch_size: Minibatch size. Increases this parameter results in more parallelization.
    Making this parameter too big is not advised though since the average gradient of the batch
    is calculated during learning.
    :param num_steps: The lenght of the RNN sequence.
    :return: Yields minibatches of numpy arrays containing ids with dimensions: [batch_size, num_steps].
    """
    # raw_data = np.array(self.data, dtype=np.int32)# is just one long array
    raw_data = self.data # is just one long array
    data_len = len(raw_data)
    if debug: print('data_len:', data_len)
    nbatches = data_len // (num_steps * batch_size)
    if debug: print('nbatches', nbatches)
    remaining_tok_cnt = data_len - nbatches * num_steps * batch_size
    if debug:  print('remaning:', remaining_tok_cnt)

    to_fill = num_steps * batch_size - remaining_tok_cnt + 1 # the last +1 is for target of last epoch
    to_fill_array = np.full(to_fill, self.vocab[EMPTY_WORD], dtype=int)
    padded_data = np.concatenate((raw_data, to_fill_array))
    if subword_weights:
      data_weights = np.concatenate((np.ones(len(raw_data) - 1, dtype=float), np.zeros(len(to_fill_array) + 1, dtype=float)))
      raw_weights = self._create_weights()
      subword_weights = np.concatenate((np.array(raw_weights[1:], dtype=float), np.zeros(len(to_fill_array) + 1, dtype=float)))
    else:
      data_weights = np.concatenate((np.ones(len(raw_data) - 1, dtype=float), np.zeros(len(to_fill_array) + 1, dtype=float)))
    if to_fill > 0:
      nbatches += 1

    if debug: print('actual batches:', nbatches)
    for i in range(nbatches):
      ip_range = padded_data[(i * batch_size * num_steps): (i + 1) * (batch_size * num_steps)]
      tar_range = padded_data[(i * batch_size * num_steps) + 1: (i + 1) * (batch_size * num_steps) + 1]
      ip_wt_range = data_weights[(i * batch_size * num_steps): (i + 1) * (batch_size * num_steps)]
      sub_wt_range = subword_weights[(i * batch_size * num_steps): (i + 1) * (batch_size * num_steps)]
      x = np.stack((np.array_split(ip_range, batch_size)), axis=0)
      y = np.stack((np.array_split(tar_range, batch_size)), axis=0)
      z = np.stack((np.array_split(ip_wt_range, batch_size)), axis=0)
      sub_wt = np.stack((np.array_split(sub_wt_range, batch_size)), axis=0)
      yield (x, y, z, sub_wt)


  def _create_weights(self):
    """
    Creates weights for entropy calculation.
    If a word has #NUMBER subword units then each subword unit will have weight 1.0/(#NUMBER subword units).
    :return:
    """
    # print(self.rev_vocab)
    raw_weights = []
    subwords = 1
    for wid in self.data:
      subword = self.rev_vocab[wid]
      if subword.endswith(SUBWORD_END):
        subwords += 1
      else:
        for _ in range(subwords):
          raw_weights.append(1.0 / subwords)
        subwords = 1
    return raw_weights

def main(argv):
  """ primarily to test if the readers work"""
  if len(argv) < 4:
      print("read_langmodel_comments.py -i <input_file> -v <vocab_limit_count> -b <batch_size> -n <num_steps>")
      sys.exit(2)
  ipfile = ""
  num_steps = 0
  batch_size  = 0
  vlimit = 0
  try:
    opts, args = getopt.getopt(argv, "i:b:n:v:")
  except getopt.GetoptError:
    print("read_langmodel_comments.py -i <input_file> -v <vocab_limit_count> -b <batch_size> -n <num_steps>")
    sys.exit(2)
  for opt, arg in opts:
    if opt == "-i":
      ipfile = arg
    if opt == "-v":
      vlimit = int(arg)
    if opt == "-b":
      batch_size = int(arg)
    if opt == "-n":
      num_steps = int(arg)

  # ip_conts = _read_words(ipfile)
  # print ip_conts
  ip_vocab, rev_ip_vocab = _build_vocab(ipfile, 0)
  wids = _file_to_word_ids(ipfile, ip_vocab)
  ip_dataset = dataset(wids, ip_vocab, rev_ip_vocab)
  # print(ip_vocab)
  print(ip_vocab['</s>'])

  print("Batches from data")
  # for (ip, tar, wts, mem) in ip_dataset.batch_producer_memory_efficient_mem_net(batch_size, num_steps, 10):
  for (ip, tar, wts) in ip_dataset.batch_producer_memory_efficient_per_file(batch_size, num_steps):
    pass
    print("-- Batch --", len(ip), len(tar), len(wts))
    print(len(ip[0]), len(tar[0]), len(wts[0]))
    print(len(ip[10]), len(tar[10]), len(wts[10]))
    print(len(ip[63]), len(tar[63]), len(wts[63]))
    # print ip
    # print tar
    # print wts

if __name__=="__main__":
  main(sys.argv[1:])

