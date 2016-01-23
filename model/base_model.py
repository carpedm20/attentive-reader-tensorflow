import os
from glob import glob
import tensorflow as tf

from data_utils import initialize_vocabulary

class Model(object):
  """Abstract object representing an Reader model."""
  def __init__(self):
    self.vocab = None
    self.data = None

  def load_vocab(self, data_dir, dataset_name, vocab_size):
    vocab_fname = os.path.join(data_dir, dataset_name, "%s.vocab%d" % (dataset_name, vocab_size))
    self.vocab, self.rev_vocab = initialize_vocabulary(vocab_fname)

  def load_dataset(self, data_dir, dataset_name, vocab_size):
    self.train_files = os.path.join(data_dir, dataset_name, "questions",
                                    "training", "*.question.ids%s" % (vocab_size))
    for fname in glob(self.train_files):
      with open(fname) as f:
        url, context, question, answer, candidates = f.read().split("\n\n")

  def train(self, epoch=25, batch_size=64,
            learning_rate=0.0002, data_dir="data",
            dataset_name="cnn", vocab_size=1000000):
    if not self.vocab:
      self.load_vocab(data_dir, dataset_name, vocab_size)
      self.load_dataset(data_dir, dataset_name, vocab_size)

    x = self.build_model(10)

  def save(self, checkpoint_dir, dataset_name):
    self.saver = tf.train.Saver()

    print(" [*] Saving checkpoints...")
    model_name = type(self).__name__ or "Reader"
    if self.batch_size:
      model_dir = "%s_%s" % (dataset_name, self.batch_size)
    else:
      model_dir = dataset_name

    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name))

  def load(self, checkpoint_dir, dataset_name):
    self.saver = tf.train.Saver()

    print(" [*] Loading checkpoints...")
    if self.batch_size:
      model_dir = "%s_%s" % (dataset_name, self.batch_size)
    else:
      model_dir = dataset_name
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
    else:
      raise Exception(" [!] Loading cehckpoints, but %s not found" % checkpoint_dir)
