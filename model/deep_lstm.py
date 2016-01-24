import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell

from base_model import Model
from cells import LSTMCell, MultiRNNCellWithSkipConn, DropoutWrapper
from data_utils import load_vocab, load_dataset

class DeepLSTM(Model):
  """Deep LSTM model."""
  def __init__(self, vocab_size, size=256, depth=3,
               learning_rate=1e-4, batch_size=32,
               keep_prob=0.1, max_nsteps=200,
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
    self.learning_rate = float(learning_rate)
    self.batch_size = int(batch_size)
    self.keep_prob = float(keep_prob)
    self.max_nsteps = int(max_nsteps)
    self.checkpoint_dir = checkpoint_dir

    print(" [*] Building Deep LSTM...")
    self.cell = LSTMCell(size, forget_bias=0.0)
    if not forward_only and self.keep_prob < 1:
      self.cell = DropoutWrapper(self.cell, output_keep_prob=keep_prob)
    self.stacked_cell = MultiRNNCellWithSkipConn([self.cell] * depth)

    self.initial_state = self.stacked_cell.zero_state(batch_size, tf.float32)

    with tf.device("/cpu:0"):
      self.emb = tf.get_variable("emb", [vocab_size, size])

    # inputs
    self.inputs = tf.placeholder(tf.int32, [batch_size, max_nsteps])
    with tf.device("/cpu:0"):
      embed_inputs = tf.nn.embedding_lookup(self.emb, tf.transpose(self.inputs))
    self.nsteps = tf.placeholder(tf.int32, [3]) # should be [0, nstep, 0]
    self.actual_batch_size = tf.placeholder(tf.int32, [3]) # should be [actual_batch_size, 1, size]

    # output states
    _, states = rnn.rnn(self.stacked_cell,
                        tf.unpack(embed_inputs),
                        dtype=tf.float32,
                        initial_state=self.initial_state)

    self.batch_states = tf.transpose(tf.pack(states), [1, 0, 2])
    self.output = tf.gather(self.batch_states, self.nsteps)

    actual_output = tf.slice(self.output, self.nsteps, self.actual_batch_size)
    self.actual_output = tf.reshape(actual_output, [-1, self.size * self.depth])

  def prepare_model(self, data_dir, dataset_name, vocab_size):
    if not self.vocab:
      self.vocab, self.rev_vocab = load_vocab(data_dir, dataset_name, vocab_size)
      print(" [*] Loading vocab finished.")

    self.W = tf.get_variable("W", [vocab_size, self.size * self.depth])

    self.target = tf.placeholder(tf.float32, [None, vocab_size])
    self.loss = tf.nn.softmax_cross_entropy_with_logits(
        tf.matmul(self.actual_output, self.W, transpose_b=True), self.target)

    print(" [*] Preparing model finished.")

  def train(self, sess, epoch=25, learning_rate=0.0002, momentum=0.9,
            decay=0.95, data_dir="data", dataset_name="cnn", vocab_size=100000):
    self.prepare_model(data_dir, dataset_name, vocab_size)

    print(" [*] Calculating gradient and loss...")
    self.optim = tf.train.AdamOptimizer(learning_rate, 0.9).minimize(self.loss)
    print(" [*] Calculating gradient and loss finished.")

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

    for epoch_idx in xrange(epoch):
      data_loader = load_dataset(data_dir, dataset_name, vocab_size)

      contexts, questions, answers = [], [], []
      for batch_idx in xrange(self.batch_size):
        _, context, question, answer, _ = data_loader.next()
        contexts.append([int(c) for c in context.split()])
        questions.append([int(q) for q in question.split()])
        answers.append(answers)

      import ipdb; ipdb.set_trace() 
      cost = sess.run([loss], feed_dict={})

      #self.model.

  def test(self, voab_size):
    self.prepare_model(data_dir, dataset_name, vocab_size)
