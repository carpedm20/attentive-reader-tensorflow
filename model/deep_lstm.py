import time
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell

from utils import array_pad
from base_model import Model
from cells import LSTMCell, MultiRNNCellWithSkipConn
from data_utils import load_vocab, load_dataset

class DeepLSTM(Model):
  """Deep LSTM model."""
  def __init__(self, vocab_size, size=256, depth=3, batch_size=32,
               keep_prob=0.1, max_nsteps=2000,
               #keep_prob=0.1, max_nsteps=10,
               checkpoint_dir="checkpoint", forward_only=False):
    """Initialize the parameters for an Deep LSTM model.
    
    Args:
      vocab_size: int, The dimensionality of the input vocab
      size: int, The dimensionality of the inputs into the Deep LSTM cell [32, 64, 256]
      learning_rate: float, [1e-3, 5e-4, 1e-4, 5e-5]
      batch_size: int, The size of a batch [16, 32]
      keep_prob: unit Tensor or float between 0 and 1 [0.0, 0.1, 0.2]
      max_nsteps: int, The max time unit [1500]
    """
    super(DeepLSTM, self).__init__()

    self.vocab_size = int(vocab_size)
    self.size = int(size)
    self.depth = int(depth)
    self.batch_size = int(batch_size)
    self.output_size = self.depth * self.size
    self.keep_prob = float(keep_prob)
    self.max_nsteps = int(max_nsteps)
    self.checkpoint_dir = checkpoint_dir

    start = time.clock()
    print(" [*] Building Deep LSTM...")
    self.cell = LSTMCell(size, forget_bias=0.0)
    if not forward_only and self.keep_prob < 1:
      self.cell = rnn_cell.DropoutWrapper(self.cell, output_keep_prob=keep_prob)
    self.stacked_cell = MultiRNNCellWithSkipConn([self.cell] * depth)

    self.initial_state = self.stacked_cell.zero_state(batch_size, tf.float32)

    with tf.device("/cpu:0"):
      self.emb = tf.get_variable("emb", [vocab_size, size])

    # inputs
    self.inputs = tf.placeholder(tf.int32, [self.batch_size, self.max_nsteps])
    with tf.device("/cpu:0"):
      embed_inputs = tf.nn.embedding_lookup(self.emb, tf.transpose(self.inputs))

    # output states
    _, states = rnn.rnn(self.stacked_cell,
                        tf.unpack(embed_inputs),
                        dtype=tf.float32,
                        initial_state=self.initial_state)
    self.batch_states = tf.pack(states)

    self.nstarts = tf.placeholder(tf.int32, [self.batch_size, 3])
    outputs = tf.pack([tf.slice(self.batch_states, nstarts, [1, 1, self.output_size])
        for idx, nstarts in enumerate(tf.unpack(self.nstarts))])

    self.outputs = tf.reshape(outputs, [self.batch_size, self.output_size])

  def prepare_model(self, data_dir, dataset_name, vocab_size):
    if not self.vocab:
      self.vocab, self.rev_vocab = load_vocab(data_dir, dataset_name, vocab_size)
      print(" [*] Loading vocab finished.")

    self.W = tf.get_variable("W", [vocab_size, self.output_size])

    self.targets = tf.placeholder(tf.float32, [self.batch_size, vocab_size])
    self.loss = tf.nn.softmax_cross_entropy_with_logits(
        tf.matmul(self.outputs, self.W, transpose_b=True), self.targets)

    print(" [*] Preparing model finished.")

  def train(self, sess, epoch=25, learning_rate=0.0002, momentum=0.9,
            decay=0.95, data_dir="data", dataset_name="cnn", vocab_size=100000):
    self.prepare_model(data_dir, dataset_name, vocab_size)

    start = time.clock()
    print(" [*] Calculating gradient and loss...")
    self.optim = tf.train.AdamOptimizer(learning_rate, 0.9).minimize(self.loss)
    print(" [*] Calculating gradient and loss finished. Take %.2fs" % (time.clock() - start))

    # Could not use RMSPropOptimizer because the sparse update of RMSPropOptimizer
    # is not implemented yet (2016.01.24).
    # self.optim = tf.train.RMSPropOptimizer(learning_rate,
    #                                        decay=decay,
    #                                        momentum=momentum).minimize(self.loss)

    sess.run(tf.initialize_all_variables())

    if self.load(self.checkpoint_dir, dataset_name):
      print(" [*] Builing Deep LSTM is loaded.")
    else:
      print(" [*] There is no checkpoint for this model.")

    targets = np.zeros([self.batch_size, self.vocab_size])

    start_time = time.time()
    for epoch_idx in xrange(epoch):
      data_loader = load_dataset(data_dir, dataset_name, vocab_size)

      counter = 0
      batch_stop = False
      while True:
        targets.fill(0)
        inputs, nstarts, answers = [], [], []
        for batch_idx in xrange(self.batch_size):
          try:
            (_, document, question, answer, _), data_idx, data_max_idx = data_loader.next()
          except StopIteration:
            batch_stop = True
            break
          inputs.append([int(d) for d in document.split()] + [0] + \
                        [int(q) for q in question.split()]) # [0] means splitter between d and q
          nstarts.append(len(inputs[-1]))
          targets[batch_idx][int(answer)] = 1

        if batch_stop:
          break

        FORCE=False
        if FORCE:
          inputs = array_pad(inputs, self.max_nsteps, pad=-1, force=FORCE)
          nstarts = np.where(inputs==-1)[1]
          inputs[inputs==-1]=0
        else:
          inputs = array_pad(inputs, self.max_nsteps, pad=0)
        nstarts = [[nstart, idx, 0] for idx, nstart in enumerate(nstarts)]

        cost = sess.run([self.loss], feed_dict={self.inputs: inputs,
                                                self.nstarts: nstarts,
                                                self.targets: targets})

        if counter % 10 == 0:
          print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.8f" \
              % (epoch_idx, data_idx, data_max_idx, time.time() - start_time, np.mean(cost)))
        counter += 1

  def test(self, voab_size):
    self.prepare_model(data_dir, dataset_name, vocab_size)
